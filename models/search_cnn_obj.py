""" CNN for architecture search """
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.search_cells import SearchCell
import genotypes as gt
from torch.nn.parallel._functions import Broadcast
import logging
import sys
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from collections import OrderedDict
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform

def broadcast_list(l, device_ids):
    """ Broadcasting list """
    l_copies = Broadcast.apply(device_ids, *l)
    l_copies = [l_copies[i:i+len(l)] for i in range(0, len(l_copies), len(l))]

    return l_copies


class SearchCNN(nn.Module):
    """ Search CNN model """
    def __init__prev(self, C_in, C, n_classes, n_layers, n_nodes=4, stem_multiplier=3):
        """
        Args:
            C_in: # of input channels
            C: # of starting model channels
            n_classes: # of classes
            n_layers: # of layers
            n_nodes: # of intermediate nodes in Cell
            stem_multiplier
        """
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.n_classes = n_classes
        self.n_layers = n_layers

        C_cur = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_cur)
        )

        # for the first cell, stem is used for both s0 and s1
        # [!] C_pp and C_p is output channel size, but C_cur is input channel size.
        C_pp, C_p, C_cur = C_cur, C_cur, C

        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):
            # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
            if i in [n_layers//3, 2*n_layers//3]:
                C_cur *= 2
                reduction = True
            else:
                reduction = False

            cell = SearchCell(n_nodes, C_pp, C_p, C_cur, reduction_p, reduction)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * n_nodes
            C_pp, C_p = C_p, C_cur_out

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(C_p, n_classes)


    def __init__(self, C_in, C, n_classes, n_layers, n_nodes=4, stem_multiplier=3):
        super().__init__()
        self.backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        self.backbone.out_channels = 1280
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))

        self.rpn = get_rpn(anchor_generator, None, self.backbone.out_channels)
        roi_pooler = MultiScaleRoIAlign(featmap_names=['0'],
                                        output_size=7,
                                        sampling_ratio=2)
        self.roi_heads = get_roi(roi_pooler, self.backbone.out_channels, n_classes)

        # TODO currently using cifar, change mean/std to pure_det
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        self.transform = GeneralizedRCNNTransform(min_size=800, max_size=1333, image_mean=image_mean, image_std=image_std)

        # self.model = FasterRCNN(backbone,
        #            num_classes=200,
        #            rpn_anchor_generator=anchor_generator,
        #            box_roi_pool=roi_pooler)

    def forward_prev(self, x, weights_normal, weights_reduce):
        s0 = s1 = self.stem(x)

        for cell in self.cells:
            weights = weights_reduce if cell.reduction else weights_normal
            s0, s1 = s1, cell(s0, s1, weights)

        out = self.gap(s1)
        out = out.view(out.size(0), -1) # flatten
        logits = self.linear(out)
        return logits

    def forward(self, x, y, weights_normal, weights_reduce):
        # using torchvision.models.detection.generalized_rcnn
        # skipping sanity checks (read: pray)

        original_image_sizes = []
        for img in x:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # images, targets = self.transform(x, [{k: v.cuda() for k,v in label.items() if not isinstance(v, str)} for label in y])
        images, targets = self.transform(x, y)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return self.eager_outputs(losses, detections)
        # return self.model(x, targets=[{"labels": label["labels"].cuda(), "boxes": label["boxes"].cuda()} for label in y])

    def eager_outputs(self, losses, detections):
        if self.training:
            return losses

        return detections

def get_rpn(rpn_anchor_generator, rpn_head, out_channels):
    if rpn_anchor_generator is None:
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(
            anchor_sizes, aspect_ratios
        )
    if rpn_head is None:
        rpn_head = RPNHead(
            out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
        )

    rpn_pre_nms_top_n = dict(training=2000, testing=1000)
    rpn_post_nms_top_n = dict(training=2000, testing=1000)

    rpn = RegionProposalNetwork(
        rpn_anchor_generator, rpn_head,
        fg_iou_thresh=0.7, bg_iou_thresh=0.3,
        batch_size_per_image=256, positive_fraction=0.5,
        pre_nms_top_n=rpn_pre_nms_top_n, post_nms_top_n=rpn_post_nms_top_n, nms_thresh=0.7)

    return rpn


def get_roi(box_roi_pool, out_channels, num_classes):
    if box_roi_pool is None:
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2)

    resolution = box_roi_pool.output_size[0]
    representation_size = 1024
    box_head = TwoMLPHead(
        out_channels * resolution ** 2,
        representation_size)

    representation_size = 1024
    box_predictor = FastRCNNPredictor(
        representation_size,
        num_classes)

    roi_heads = RoIHeads(
        # Box
        box_roi_pool, box_head, box_predictor,
        fg_iou_thresh=0.5, bg_iou_thresh=0.5,
        batch_size_per_image=512, positive_fraction=0.25,
        bbox_reg_weights=None,
        score_thresh=0.05, nms_thresh=0.5, detections_per_img=100)

    return roi_heads

class SearchCNNControllerObj(nn.Module):
    """ SearchCNN controller supporting multi-gpu """
    def __init__(self, C_in, C, n_classes, n_layers, criterion, n_nodes=4, stem_multiplier=3,
                 device_ids=None, class_loss=None, weight_dict=None):
        super().__init__()
        self.n_nodes = n_nodes
        self.criterion = criterion
        self.class_loss = class_loss
        self.weight_dict = weight_dict # for detr loss
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids

        # initialize architect parameters: alphas
        n_ops = len(gt.PRIMITIVES)

        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()

        for i in range(n_nodes):
            self.alpha_normal.append(nn.Parameter(1e-3*torch.randn(i+2, n_ops)))
            self.alpha_reduce.append(nn.Parameter(1e-3*torch.randn(i+2, n_ops)))

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))

        self.net = SearchCNN(C_in, C, n_classes, n_layers, n_nodes, stem_multiplier)

    def forward(self, x, y):
        weights_normal = [F.softmax(alpha, dim=-1) for alpha in self.alpha_normal]
        weights_reduce = [F.softmax(alpha, dim=-1) for alpha in self.alpha_reduce]

        if len(self.device_ids) == 1:
            return self.net(x, y, weights_normal, weights_reduce)

        # scatter x
        xs = nn.parallel.scatter(x, self.device_ids)

        # scatter x
        ys = nn.parallel.scatter(y, self.device_ids)
        # broadcast weights
        wnormal_copies = broadcast_list(weights_normal, self.device_ids)
        wreduce_copies = broadcast_list(weights_reduce, self.device_ids)

        # replicate modules
        replicas = nn.parallel.replicate(self.net, self.device_ids)
        outputs = nn.parallel.parallel_apply(replicas,
                                             list(zip(xs, ys, wnormal_copies, wreduce_copies)),
                                             devices=self.device_ids)
        return nn.parallel.gather(outputs, self.device_ids[0])

    def loss(self, X, y, is_multi):
        logits = self.forward(X)
        try:
            # raise AttributeError(y.shape, logits.shape, y)
            if is_multi:
                y = y.float()
            return self.criterion(logits, y)
        except (RuntimeError, ValueError) as e:
            print(e)
            raise AttributeError(y.shape, logits.shape, y, self.criterion, logits)
            sys.exit()

    def print_alphas(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### ALPHA #######")
        logger.info("# Alpha - normal")
        for alpha in self.alpha_normal:
            logger.info(F.softmax(alpha, dim=-1))

        logger.info("\n# Alpha - reduce")
        for alpha in self.alpha_reduce:
            logger.info(F.softmax(alpha, dim=-1))
        logger.info("#####################")

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def genotype(self):
        gene_normal = gt.parse(self.alpha_normal, k=2)
        gene_reduce = gt.parse(self.alpha_reduce, k=2)
        concat = range(2, 2+self.n_nodes) # concat all intermediate nodes

        return gt.Genotype(normal=gene_normal, normal_concat=concat,
                           reduce=gene_reduce, reduce_concat=concat)

    def weights(self):
        return self.net.parameters()

    def named_weights(self):
        return self.net.named_parameters()

    def alphas(self):
        for n, p in self._alphas:
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p
