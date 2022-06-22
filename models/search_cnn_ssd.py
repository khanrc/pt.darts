""" CNN for architecture search """
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.search_cells import SearchCell
import genotypes as gt
from torch.nn.parallel._functions import Broadcast
import logging
from collections import OrderedDict
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torch.hub import load_state_dict_from_url
from torchvision.ops import boxes as box_ops
import torchvision.models.detection._utils as det_utils
from anchor_utils import DefaultBoxGenerator
from torchvision.models.vgg import VGG, vgg16
import warnings
from ssd_torchvision import _vgg_extractor, SSD, SSDMatcher, retrieve_out_channels, SSDHead, _topk_min


def broadcast_list(l, device_ids):
    """ Broadcasting list """
    l_copies = Broadcast.apply(device_ids, *l)
    l_copies = [l_copies[i:i+len(l)] for i in range(0, len(l_copies), len(l))]

    return l_copies


class SearchCNN(nn.Module):
    """ Search CNN model """
    def __init__(self, C_in, C, n_classes, n_layers, n_nodes=4, stem_multiplier=3):
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

        # Visualize feature maps
        self.activation = {}
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output.detach()
            return hook


        self.C_in = C_in
        self.C = C
        self.n_classes = n_classes
        self.n_layers = n_layers
        num_classes = 91

        C_cur = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_cur)
        )

        # for the first cell, stem is used for both s0 and s1
        # [!] C_pp and C_p is output channel size, but C_cur is input channel size.
        C_pp, C_p, C_cur = C_cur, C_cur, C

        # self.stem = torchvision.models.mobilenet_v2(pretrained=True).features
        # C_pp, C_p, C_cur = 1280, 1280, 128

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
            cell.preproc1.net[1].register_forward_hook(get_activation(f'cell{i}')) # cell preproc1 is necessarily ops.stdconv
            print(f"cell{i} shape is {cell.preproc1.net[1]}")
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * n_nodes
            C_pp, C_p = C_p, C_cur_out

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.test_gap = nn.Conv2d(256, 512, 1, 1, 1, bias=False)
        # out_channels = 256
        # out_channels = 1280
        # self.linear = nn.Linear(C_p, out_channels)

        # self.backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        # self.backbone.out_channels = out_channels

        self.anchor_generator = DefaultBoxGenerator(
            [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
            scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
            steps=[8, 16, 32, 64, 100, 300],
        )
        self.box_coder = det_utils.BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))


        # out_channels = retrieve_out_channels(backbone, (300,300))
        out_channels = [512, 1024, 512, 256, 256, 256]
        # out_channels = [256] # ? ... list of 1 since not using pyramid network w/in backbone

        if len(out_channels) != len(self.anchor_generator.aspect_ratios):
            raise ValueError(
                f"The length of the output channels from the backbone ({len(out_channels)}) do not match the length of the anchor generator aspect ratios ({len(self.anchor_generator.aspect_ratios)})"
            )

        num_anchors = self.anchor_generator.num_anchors_per_location()
        head = SSDHead(out_channels, num_anchors, num_classes)
        self.head = head

        self.proposal_matcher = SSDMatcher(0.5)

        # image_mean = [0.48235, 0.45882, 0.40784]
        # image_std = [(1.0 / 255.0), (1.0 / 255.0), (1.0 / 255.0)],  # undo the 0-1 scaling of toTensor

        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        self.transform = GeneralizedRCNNTransform(
            300, 300, image_mean, image_std#, size_divisible=1, fixed_size=size, **kwargs
        )

        self.score_thresh = 0.01
        self.nms_thresh = 0.45
        self.detections_per_img = 200
        self.topk_candidates = 400
        self.neg_to_pos_ratio = (1.0 - 0.25) / 0.25

        # used only on torchscript mode
        self._has_warned = False

        #################################################################
        # load a pretrained_model, and transfer across the weights of the non-backbone portion
        defaults = {
            # Rescale the input in a way compatible to the backbone
            "image_mean": image_mean,
            "image_std": image_std
        }
        kwargs = {**defaults}
        pretrained_backbone = vgg16(pretrained=False, progress=True)
        pretrained_backbone = _vgg_extractor(pretrained_backbone, False, trainable_layers=0)
        pretrained = SSD(pretrained_backbone, self.anchor_generator, (300, 300), num_classes, **kwargs)
        state_dict = torch.load('/hdd/PhD/nas/pt.darts/ssd30016.pth')
        # state_dict = torch.load('/home/matt/Documents/nas/darts/ssd30016.pth')
        pretrained.load_state_dict(state_dict)

        # Copy network weights from pretrained.rpn + freeze them?
        # note not worth doing this (this way) even for a directly copied backbone as those
        # use cell format, which adopts alpha/weights normal/reduce, instantiated in controller.
        self.anchor_generator.load_state_dict(pretrained.anchor_generator.state_dict())
        self.head.load_state_dict(pretrained.head.state_dict())

        del pretrained


    def eager_outputs(self, losses, detections):
        if self.training:
            return losses

        return detections

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):

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

    def forward(self, x, y, weights_normal, weights_reduce, full_ret):
        if self.training:
            if y is None:
                assert False, "targets should not be none when in training mode"
            else:
                for target in y:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        assert (len(boxes.shape) == 2 and boxes.shape[-1] == 4), f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}."

                    else:
                        assert False, f"Expected target boxes to be of type Tensor, got {type(boxes)}."

        # get the original image sizes
        original_image_sizes = []
        for img in x:
            val = img.shape[-2:]
            assert len(val) == 2, f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}"

            original_image_sizes.append((val[0], val[1]))

        # transform the input
        # images, targets = self.transform(x, y)
        images, targets = self.transform(x, [{k: v.cuda() for k,v in label.items() if not isinstance(v, str)} for label in y])

        # Check for degenerate boxes
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb = boxes[bb_idx].tolist()
                    raise AssertionError(f"All bounding boxes should have positive height and width. "
                                         f"Found invalid box {degen_bb} for target at index {target_idx}.")

        # get the features from the backbone
        s0 = s1 = self.stem(x) # use tensor form of images not transformed form
        # s0 = s1 = self.stem(images.tensors)

        for cell in self.cells:
            weights = weights_reduce if cell.reduction else weights_normal
            s0, s1 = s1, cell(s0, s1, weights)

        # features = s1
        features = self.test_gap(s1)

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

class SearchCNNControllerSSD(nn.Module):
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

    def forward(self, x, y, full_ret=False):
        weights_normal = [F.softmax(alpha, dim=-1) for alpha in self.alpha_normal]
        weights_reduce = [F.softmax(alpha, dim=-1) for alpha in self.alpha_reduce]

        if len(self.device_ids) == 1:
            return self.net(x, y, weights_normal, weights_reduce, full_ret)

        # # scatter x
        # xs = nn.parallel.scatter(x, self.device_ids)
        #
        # # scatter x
        # ys = nn.parallel.scatter(y, self.device_ids)
        # # broadcast weights
        # wnormal_copies = broadcast_list(weights_normal, self.device_ids)
        # wreduce_copies = broadcast_list(weights_reduce, self.device_ids)
        #
        # # replicate modules
        # replicas = nn.parallel.replicate(self.net, self.device_ids)
        # outputs = nn.parallel.parallel_apply(replicas,
        #                                      list(zip(xs, ys, wnormal_copies, wreduce_copies)),
        #                                      devices=self.device_ids)
        # return nn.parallel.gather(outputs, self.device_ids[0])

    def loss(self, X, y, is_multi):
        losses = self.forward(X, y)
        return sum(_loss for _loss in losses.values())

        # return self.criterion(logits, y)

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
