import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import detection_utils as utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
import torchvision.transforms as tf
from PIL import ImageDraw
import numpy as np

import sys
sys.path.insert(0, "/home2/lgfm95/hem/perceptual")
sys.path.insert(0, "/hdd/PhD/hem/perceptual")
sys.path.insert(0, "/home/matt/Documents/hem/perceptual")
from coco_obj import get_dict

class_dict, rev_class_dict = get_dict()


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        # lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        #     optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        # )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 10, eta_min=0.001)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):

        # assert len(images) == len(targets)
        # for i in range(len(images)):
        #     image, target = images[i], targets[i]
        #     image = tf.ToPILImage()(image)
        #     draw = ImageDraw.Draw(image)
        #     for i in range(len(target["boxes"])):
        #         draw.rectangle(np.array(target["boxes"][i]))
        #         draw.text((target['boxes'][i][0].item() + 2, target['boxes'][i][1].item() + 2), str(rev_class_dict[target['labels'][i].item()]))
        #     image.save(f"tempSave/validate_obj/coco/{target['image_id']}.png")

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items() if not isinstance(v, str)} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            # loss_dict = model(images, targets)
            try:
                loss_dict, detections = model(images, targets)
            except ValueError:
                raise AttributeError(model(images, targets), model.training)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def train_one_epoch_ssd(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        # lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        #     optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        # )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 10, eta_min=0.001)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):

        # for i in range(len(images)): # working
        #     image, target = images[i], targets[i]
        #     image = tf.ToPILImage()(image)
        #     draw = ImageDraw.Draw(image)
        #     for i in range(len(target["boxes"])):
        #         draw.rectangle(np.array(target["boxes"][i]))
        #         draw.text((target['boxes'][i][0].item() + 2, target['boxes'][i][1].item() + 2), str(rev_class_dict[target['labels'][i].item()]))
        #     image.save(f"tempSave/validate_obj/coco/{target['image_id']}.png")

        images = torch.stack([image.to(device) for image in images])
        targets = [{k: v.to(device) for k, v in t.items() if not isinstance(v, str)} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # loss_dict = model(images)
            # loss_dict = tuple(loss.to(device) for loss in loss_dict)

            # needs labels in form batchsize x num_objects x 5 (first 4 is bbox, 5th is class label)
            # boxlabels = [torch.cat((targets[i]['boxes'], targets[i]['labels'].unsqueeze(1)), 1) for i in range(len(images))]
            # loss_l, loss_c = criterion(loss_dict, boxlabels)
            # losses = loss_l + loss_c
            # loss_dict = {"loss_l": loss_l, "loss_c": loss_c}

        # # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


def evaluate(model, data_loader, device, epoch=0, augment=False):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    step = 0
    zero_preds = True
    with torch.no_grad():
        for images, targets in metric_logger.log_every(data_loader, 100, header):
            # images = list(img.to(device) for img in images)
            # old_images_shape = images.shape
            images = torch.stack([img.to(device) for img in images], dim=0)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_time = time.time()
            outputs = model(images, targets)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            if step < 10:
                for i in range(len(outputs)): # batch size
                    to_save = f"./tempSave/validate_obj/{epoch}-{step}-{i}.png"
                    utils.draw_bounding_boxes(images[i].to(cpu_device), outputs[i]["boxes"], to_save, labels=None, fill=True)
            model_time = time.time() - model_time

            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            if len(res) == 0: # do not try to update w/ 0 predictions
                step += 1
                continue

            zero_preds = False

            # res = {target["image_id"]: output for target, output in zip(targets, outputs)}
            evaluator_time = time.time()
            coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
            step += 1

    if zero_preds and not augment:
        raise AttributeError("using zero pred system for search. are you sure?")
    elif zero_preds and augment: # create artificial pred (top left, tiny box equivalent, score will still be 0)
        artificial_res = {targets[0]["image_id"].item(): {"boxes": torch.empty((0,4)),
                                                      "labels": torch.tensor([], dtype=torch.int64),
                                                      "scores": torch.tensor([])
                                                      }}
        coco_evaluator.update(artificial_res)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
