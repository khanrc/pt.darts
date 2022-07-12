from torchvision.models.vgg import VGG, vgg16
from ssd_torchvision import _vgg_extractor, SSD, SSDMatcher, retrieve_out_channels, SSDHead, _topk_min, _xavier_init
from anchor_utils import DefaultBoxGenerator
import torchvision.models.detection._utils as det_utils
import torch
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from collections import OrderedDict
from torchvision.ops import boxes as box_ops
import warnings
import torch.nn as nn
import torch.nn.functional as F

from simple_cell import Simple_Cell
import genotypes as gt

class test_class(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
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
        # state_dict = torch.load('/home/matt/Documents/nas/darts/ssd30016.pth')
        pretrained.load_state_dict(state_dict)
        self.anchor_generator.load_state_dict(pretrained.anchor_generator.state_dict())
        self.head.load_state_dict(pretrained.head.state_dict())
        del pretrained

    def forward(self, images, targets=None):
        if self.training:
            if targets is None:
                assert False, "targets should not be none when in training mode"
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        assert (len(boxes.shape) == 2 and boxes.shape[-1] == 4), f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}."

                    else:
                        assert False, f"Expected target boxes to be of type Tensor, got {type(boxes)}."

        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
        assert len(val) == 2, f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}"

        original_image_sizes.append((val[0], val[1]))

        # transform the input
        # raise AttributeError(images.shape, targets[0])
        images, targets = self.transform(images, targets)
        # get the features from the backbone
        features = self.backbone(images.tensors)
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

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("SSD always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
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

class backbonevgg(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vgg16(pretrained=False)
        self.backbone = _vgg_extractor(self.backbone, False, trainable_layers=None)

    def forward(self, x):
        return self.backbone(x)

class backbonecell(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Simple_Cell(C_in=3, C=16, n_classes=91, n_layers=4)
        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()
        n_ops = len(gt.PRIMITIVES)

        for i in range(4):
            self.alpha_normal.append(nn.Parameter(1e-3*torch.randn(i+2, n_ops)))
            self.alpha_reduce.append(nn.Parameter(1e-3*torch.randn(i+2, n_ops)))

    def forward(self, x):
        weights_normal = [F.softmax(alpha, dim=-1) for alpha in self.alpha_normal]
        weights_reduce = [F.softmax(alpha, dim=-1) for alpha in self.alpha_reduce]
        return self.backbone(x, weights_normal, weights_reduce)