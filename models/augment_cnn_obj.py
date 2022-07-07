""" CNN for network augmentation """
import torch
import torch.nn as nn
from models.augment_cells import AugmentCell
from models import ops
from models.search_cnn_obj import get_roi, get_rpn

from torchvision.models.detection import FasterRCNN
# from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
# from anchor_utils import AnchorGenerator
from rpn36 import AnchorGenerator
from collections import OrderedDict
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.roi_heads_hardness import RoIHeads
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torch.hub import load_state_dict_from_url


class AuxiliaryHead(nn.Module):
    """ Auxiliary head in 2/3 place of network to let the gradient flow well """
    def __init__(self, input_size, C, n_classes):
        """ assuming input size 7x7 or 8x8 """
        second_kernel_size = 2
        if input_size in [7, 8]:
            kernel_size = 5
            stride = input_size-5
        elif input_size == 56:
            kernel_size = 10
            stride = 4
            second_kernel_size = 4
        else:
            kernel_size = 10
            stride = 4
        # else:
        #     raise AssertionError("input size not appropriate", input_size)
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size, stride=stride, padding=0, count_include_pad=False), # 2x2 out
            nn.Conv2d(C, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, kernel_size=second_kernel_size, bias=False), # 1x1 out
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.linear = nn.Linear(768, n_classes)

    def forward(self, x):
        out = x
        for module in self.net:
            out = module(out)
            # print(out.shape)
        # out = self.net(x)
        raise AttributeError(out.shape)
        out = out.view(out.size(0), -1) # flatten
        logits = self.linear(out)
        return logits


def get_multihot(labels, num_classes):
    multihot = [0] * num_classes
    for lab in labels:
        multihot[lab] = 1
    return torch.tensor(multihot).cuda()


class AugmentCNN(nn.Module):
    """ Augmented CNN model """
    def __init__(self, input_size, C_in, C, n_classes, n_layers, genotype, aux_criterion,
                 aux_weight, use_kendall, stem_multiplier=3):
        """
        Args:
            input_size: size of height and width (assuming height = width)
            C_in: # of input channels
            C: # of starting model channels
        """
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.genotype = genotype
        # aux head position
        self.aux_criterion = aux_criterion
        self.aux_weight = aux_weight
        self.use_aux = self.aux_weight > 0.
        self.aux_pos = 2*n_layers//3 if self.use_aux else -1

        self.use_kendall = use_kendall

        C_cur = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_cur)
        )

        C_pp, C_p, C_cur = C_cur, C_cur, C

        self.cells = nn.ModuleList()
        reduction_p = False
        last_concat_length = 0
        for i in range(n_layers):
            if i in [n_layers//3, 2*n_layers//3]:
                C_cur *= 2
                reduction = True
            else:
                reduction = False

            cell = AugmentCell(genotype, C_pp, C_p, C_cur, reduction_p, reduction)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * len(cell.concat)
            C_pp, C_p = C_p, C_cur_out
            last_concat_length = len(cell.concat)
            print(f"cell{i} shape is {cell.preproc1.net[1]}")

            if i == self.aux_pos:
                # [!] this auxiliary head is ignored in computing parameter size
                #     by the name 'aux_head'
                self.aux_head = AuxiliaryHead(input_size//4, C_p, n_classes)

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.linear = nn.Linear(C_p, n_classes)
        self.additional_pipe = None
        if last_concat_length == 8: # nodes=8
            self.additional_pipe = nn.Conv2d(C_p, 256, kernel_size=(1,1))

        out_channels = 256
        # out_channels = 512
        # out_channels = 1280
        # self.linear = nn.Linear(C_p, out_channels)

        # self.backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        # self.backbone.out_channels = out_channels

        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(
            anchor_sizes, aspect_ratios
        )

        self.rpn = get_rpn(anchor_generator, None, out_channels)
        # self.rpn.head.conv.register_forward_hook(get_activation(f'cellhead'))
        roi_pooler = MultiScaleRoIAlign(featmap_names=['0'],
                                        output_size=7,
                                        sampling_ratio=2)
        self.roi_heads = get_roi(roi_pooler, out_channels, n_classes)

        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        self.transform = GeneralizedRCNNTransform(min_size=800, max_size=1333, image_mean=image_mean, image_std=image_std)

        # load pretrain
        pretrained_backbone = resnet_fpn_backbone('resnet50', False, trainable_layers=0)
        pretrained = FasterRCNN(pretrained_backbone,
                                num_classes=91,
                                # rpn_anchor_generator=anchor_generator,
                                box_roi_pool=roi_pooler,
                                )
        state_dict = load_state_dict_from_url(
            'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth', progress=True)
        pretrained.load_state_dict(state_dict)

        # Copy network weights from pretrained.rpn + freeze them
        # note not worth doing this (this way) even for a directly copied backbone as those
        # use cell format, which adopts alpha/weights normal/reduce, instantiated in controller.
        self.rpn.load_state_dict(pretrained.rpn.state_dict())
        # self.rpn.requires_grad_(False)

        self.roi_heads.load_state_dict(pretrained.roi_heads.state_dict())
        # self.roi_heads.requires_grad_(False)

        del pretrained

        num_log_vars = 5 if self.use_aux else 4
        self.log_vars = nn.Parameter(torch.zeros((num_log_vars)))

    def forward(self, x, y, full_ret=False):

        original_image_sizes = []
        for img in x:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(x, [{k: v.cuda() for k,v in label.items() if not isinstance(v, str)} for label in y])


        s0 = s1 = self.stem(x)

        aux_logits = None
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            if i == self.aux_pos and self.training:
                try:
                    aux_logits = self.aux_head(s1)
                except RuntimeError:
                    raise AttributeError(i, s1.shape, self.aux_head)

        # out = self.gap(s1)
        # out = out.view(out.size(0), -1) # flatten
        # logits = self.linear(out)
        features = s1
        if self.additional_pipe is not None:
            features = self.additional_pipe(features)
        # raise AttributeError(features.shape)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        try:
            proposals, proposal_losses = self.rpn(images, features, targets)
        except ValueError:
            raise AttributeError(targets, len(targets), len(targets[0]), len(images.tensors), len(images.tensors[0]))#, len(targets[1]))
        try:
            detections, detector_losses, hardness = self.roi_heads(features, proposals, images.image_sizes, targets)
        except AttributeError:
            raise AttributeError(len(proposals), len(features), len(proposals[0]), len(features['0']), len(targets), len(targets[0]), len(targets[1]), len(images.tensors))#, len(images[0]))

        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.use_aux and self.training:
            y_classes = [get_multihot(labels=label["labels"], num_classes=91) for label in y]
            y_classes = torch.stack(y_classes, dim=0)
            sigmoid = torch.sigmoid(aux_logits)
            rounded = torch.round(sigmoid)
            aux_loss = self.aux_criterion(rounded, y_classes.float())
            aux_loss = self.aux_weight * aux_loss
            losses.update({'aux_loss': aux_loss})

        if self.use_kendall:
            for q, (key, loss) in enumerate(losses.items()):
                precision = torch.exp(-self.log_vars[q])
                new_loss = torch.sum(precision * loss ** 2. + self.log_vars[q], -1)
                losses[key] = new_loss
        return self.eager_outputs(losses, detections, hardness, full_ret)

    def eager_outputs(self, losses, detections, hardness, full_ret):
        if self.training:
            if full_ret:
                return losses, detections, hardness
            return losses

        return detections


    def drop_path_prob(self, p):
        """ Set drop path probability """
        for module in self.modules():
            if isinstance(module, ops.DropPath_):
                module.p = p
