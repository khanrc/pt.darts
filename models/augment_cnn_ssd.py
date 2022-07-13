""" CNN for network augmentation """
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.augment_cells import AugmentCell
from models import ops
from collections import OrderedDict
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import boxes as box_ops
import torchvision.models.detection._utils as det_utils
from anchor_utils import DefaultBoxGenerator
from torchvision.models.vgg import VGG, vgg16
import warnings
from ssd_torchvision import _vgg_extractor, SSD, SSDMatcher, retrieve_out_channels, SSDHead, _topk_min, _xavier_init
import detection_utils as utils


class AuxiliaryHead(nn.Module):
    """ Auxiliary head in 2/3 place of network to let the gradient flow well """
    def __init__(self, input_size, C, n_classes):
        """ assuming input size 7x7 or 8x8 """
        second_kernel_size = 2
        if input_size in [7, 8]:
            kernel_size = 5
            stride = input_size-5
        elif input_size == 56: # TODO too brutal? even if its for just aux.
            kernel_size = 10
            stride = 6
            second_kernel_size = 8
        elif input_size == 32: # TODO too brutal? even if its for just aux.
            kernel_size = 10
            stride = 4
            second_kernel_size = 6
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
        C = C*2  # double such that after cells we have 512 dim size rather than 256 for ssd piping purposes
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

        self.anchor_generator = DefaultBoxGenerator(
            [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
            scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
            steps=[8, 16, 32, 64, 100, 300],
        )
        self.box_coder = det_utils.BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        defaults = {
            # Rescale the input in a way compatible to the backbone
            "image_mean": image_mean,
            "image_std": image_std
        }
        kwargs = {**defaults}

        out_channels = [512, 1024, 512, 256, 256, 256]
        num_anchors = self.anchor_generator.num_anchors_per_location()
        self.head = SSDHead(out_channels, num_anchors, 91)
        self.proposal_matcher = SSDMatcher(0.5)

        self.transform = GeneralizedRCNNTransform(
            300, 300, image_mean, image_std#, size_divisible=1, fixed_size=size, **kwargs
        )

        self.score_thresh = 0.01
        self.nms_thresh = 0.45
        self.detections_per_img = 200
        self.topk_candidates = 400
        self.neg_to_pos_ratio = (1.0 - 0.25) / 0.25
        self._has_warned = False

        pretrained_backbone = vgg16(pretrained=False, progress=True)
        pretrained_backbone = _vgg_extractor(pretrained_backbone, False, trainable_layers=0)
        pretrained = SSD(pretrained_backbone, self.anchor_generator, (300, 300), 91, **kwargs)
        state_dict = torch.load('/hdd/PhD/nas/pt.darts/ssd30016.pth')
        pretrained.load_state_dict(state_dict)
        self.anchor_generator.load_state_dict(pretrained.anchor_generator.state_dict())
        self.head.load_state_dict(pretrained.head.state_dict())
        del pretrained

        num_log_vars = 3 if self.use_aux else 2
        self.log_vars = nn.Parameter(torch.zeros((num_log_vars)))


    def forward(self, x, y):

        original_image_sizes = []
        for img in x:
            val = img.shape[-2:]
        assert len(val) == 2, f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}"

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
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        features = list(features.values())

        # compute the ssd heads outputs using the features
        head_outputs = self.head(features)

        # create the set of anchors
        anchors = self.anchor_generator(images, features)

        losses = {}
        detections = []
        if self.training:
            matched_idxs = []
            if targets is None:
                assert False, "targets should not be none when in training mode"
            else:
                for anchors_per_image, targets_per_image in zip(anchors, targets):
                    if targets_per_image["boxes"].numel() == 0:
                        matched_idxs.append(
                            torch.full(
                                (anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device
                            )
                        )
                        continue

                    match_quality_matrix = box_ops.box_iou(targets_per_image["boxes"], anchors_per_image)
                    matched_idxs.append(self.proposal_matcher(match_quality_matrix))

                losses = self.compute_loss(targets, head_outputs, anchors, matched_idxs)
        else:
            detections = self.postprocess_detections(head_outputs, anchors, images.image_sizes)
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = utils.reduce_dict(losses)

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

        return self.eager_outputs(losses, detections)

    def eager_outputs(self, losses, detections):
        if self.training:
            return losses

        return detections

    def postprocess_detections(self, head_outputs, image_anchors, image_shapes):
        bbox_regression = head_outputs["bbox_regression"]
        pred_scores = F.softmax(head_outputs["cls_logits"], dim=-1)

        num_classes = pred_scores.size(-1)
        device = pred_scores.device

        detections = []

        for boxes, scores, anchors, image_shape in zip(bbox_regression, pred_scores, image_anchors, image_shapes):
            boxes = self.box_coder.decode_single(boxes, anchors)
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            image_boxes = []
            image_scores = []
            image_labels = []
            for label in range(1, num_classes):
                score = scores[:, label]

                keep_idxs = score > self.score_thresh
                score = score[keep_idxs]
                box = boxes[keep_idxs]

                # keep only topk scoring predictions
                num_topk = _topk_min(score, self.topk_candidates, 0)
                score, idxs = score.topk(num_topk)
                box = box[idxs]

                image_boxes.append(box)
                image_scores.append(score)
                image_labels.append(torch.full_like(score, fill_value=label, dtype=torch.int64, device=device))

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            # non-maximum suppression
            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[: self.detections_per_img]

            detections.append(
                {
                    "boxes": image_boxes[keep],
                    "scores": image_scores[keep],
                    "labels": image_labels[keep],
                }
            )
        return detections

    def compute_loss(
        self,
        targets,
        head_outputs,
        anchors,
        matched_idxs):

        bbox_regression = head_outputs["bbox_regression"]
        cls_logits = head_outputs["cls_logits"]

        # Match original targets with default boxes
        num_foreground = 0
        bbox_loss = []
        cls_targets = []
        for (
            targets_per_image,
            bbox_regression_per_image,
            cls_logits_per_image,
            anchors_per_image,
            matched_idxs_per_image,
        ) in zip(targets, bbox_regression, cls_logits, anchors, matched_idxs):
            # produce the matching between boxes and targets
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            foreground_matched_idxs_per_image = matched_idxs_per_image[foreground_idxs_per_image]
            num_foreground += foreground_matched_idxs_per_image.numel()

            # Calculate regression loss
            matched_gt_boxes_per_image = targets_per_image["boxes"][foreground_matched_idxs_per_image]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]
            target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
            bbox_loss.append(
                torch.nn.functional.smooth_l1_loss(bbox_regression_per_image, target_regression, reduction="sum")
            )

            # Estimate ground truth for class targets
            gt_classes_target = torch.zeros(
                (cls_logits_per_image.size(0),),
                dtype=targets_per_image["labels"].dtype,
                device=targets_per_image["labels"].device,
            )
            gt_classes_target[foreground_idxs_per_image] = targets_per_image["labels"][
                foreground_matched_idxs_per_image
            ]
            cls_targets.append(gt_classes_target)


        bbox_loss = torch.stack(bbox_loss)
        cls_targets = torch.stack(cls_targets)

        # Calculate classification loss
        num_classes = cls_logits.size(-1)
        cls_loss = F.cross_entropy(cls_logits.view(-1, num_classes), cls_targets.view(-1), reduction="none").view(
            cls_targets.size()
        )

        # Hard Negative Sampling
        foreground_idxs = cls_targets > 0
        num_negative = self.neg_to_pos_ratio * foreground_idxs.sum(1, keepdim=True)
        # num_negative[num_negative < self.neg_to_pos_ratio] = self.neg_to_pos_ratio
        negative_loss = cls_loss.clone()
        negative_loss[foreground_idxs] = -float("inf")  # use -inf to detect positive values that creeped in the sample
        values, idx = negative_loss.sort(1, descending=True)
        # background_idxs = torch.logical_and(idx.sort(1)[1] < num_negative, torch.isfinite(values))
        background_idxs = idx.sort(1)[1] < num_negative

        N = max(1, num_foreground)
        return {
            "bbox_regression": bbox_loss.sum() / N,
            "classification": (cls_loss[foreground_idxs].sum() + cls_loss[background_idxs].sum()) / N,
        }

    def drop_path_prob(self, p):
        """ Set drop path probability """
        for module in self.modules():
            if isinstance(module, ops.DropPath_):
                module.p = p