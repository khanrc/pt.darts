import torchvision
import torchvision.transforms as tf
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

import torch
import sys
import copy

import ssd_torchvision
import utils
import preproc
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/hdd/PhD/hem/perceptual")
from det_dataset import Imagenet_Det as Pure_Det
from coco_obj import COCODetLoader as Coco_Det
from subloader import SubDataset
from detectionengine import train_one_epoch_ssd, evaluate
from ssd_torchvision import SSD300_VGG16 as ssd300

from new_test_class import test_class, backbonevgg, backbonecell
# from torchvision._internally_replaced_utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

is_pretrained = eval(sys.argv[sys.argv.index('--obj_pretrained')+1])
is_retrained = eval(sys.argv[sys.argv.index('--obj_retrained')+1])
is_fixed = eval(sys.argv[sys.argv.index('--obj_fixed')+1])
dataset = sys.argv[sys.argv.index('--dataset')+1]
batch_size = eval(sys.argv[sys.argv.index('--batch_size')+1])


def collate_fn(batch):
    data, labels = zip(*batch)
    stacked_data = torch.stack(data, dim=0)
    return stacked_data, labels


def main():
    num_classes = 200
    if dataset == "pure_det":
        # train_transforms, _ = preproc.data_transforms("pure_det", cutout_length=0)
        # full_set = Pure_Det(train_path, train_transforms)
        # _, _, _, train_data, _ = utils.get_data(
        #     "pure_det", "", cutout_length=0, validation=True, search=True,
        #     bede=False, is_concat=False)
        trn_transform = tf.Compose([
                tf.Resize((64, 64)),
                # tf.RandomHorizontalFlip(),
                tf.ToTensor(),
                # normalize,
            ])
        train_path = '/hdd/PhD/data/imagenet2017detection/' # TODO revert
        # train_path = '/data/mining/imageobjdata/'
        train_data = Pure_Det(root=train_path, transforms=trn_transform)
    elif dataset == "coco_det":
        # train_transforms, _ = preproc.data_transforms("coco_det", cutout_length=0)
        # _, _, _, train_data, _ = utils.get_data(
        #     "coco_det", "", cutout_length=0, validation=True, search=True,
        #     bede=False, is_concat=False)
        trn_transform = tf.Compose([
                tf.Resize((64, 64)),
                # tf.RandomHorizontalFlip(),
                tf.ToTensor(),
                # normalize,
            ])
        train_path = '/hdd/PhD/data/coco/'
        # train_path = '/home/matt/Documents/coco/'
        # train_data = SubDataset(transforms=trn_transform, dataset_name="coco_det", convert_to_paths=True)
        train_data = Coco_Det(train_path=train_path, transforms=trn_transform)
        val_data = Coco_Det(train_path=train_path, transforms=trn_transform, train=False)
        num_classes = 91
    else:
        raise AttributeError("bad dataset")

    print(f"batch size {batch_size}")
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size, # needs to be > 1
                                               # sampler=train_sampler,
                                               num_workers=0,
                                               pin_memory=True,
                                               collate_fn=collate_fn
                                               )
    val_loader = torch.utils.data.DataLoader(val_data,
                                               batch_size=batch_size, # needs to be > 1
                                               # sampler=train_sampler,
                                               num_workers=0,
                                               pin_memory=True,
                                               collate_fn=collate_fn
                                               )

    device = torch.device("cuda")
    # model = ssd300()
    # backbone = backbonevgg()
    backbone = backbonecell()
    model = test_class(backbone)

    if is_pretrained:
        state_dict = torch.load('/hdd/PhD/nas/pt.darts/ssd30016.pth')
        # state_dict = torch.load('/home/matt/Documents/nas/darts/ssd30016.pth')
        model.load_state_dict(state_dict)
        if is_retrained:
            if is_fixed:
                # fix non backbone weights
                for param in model.parameters():
                    param.requires_grad = False

            # load back in new (untrained) backbone
            backbone = torchvision.models.mobilenet_v2(pretrained=True).features
            backbone[-1][0] = torch.nn.Conv2d(320, 512, kernel_size=(3,3), stride=(1,1), bias=False)
            backbone[-1][1] = torch.nn.BatchNorm2d(512)
            model.backbone = backbone

            for param in model.backbone.parameters():
                param.requires_grad = True

        if dataset == "pure_det":
            # set to 200 class.
            model.roi_heads.box_predictor.cls_score = torch.nn.Linear(1024, num_classes, bias=True)
            model.roi_heads.box_predictor.cls_score.requires_grad_(True)
            model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(1024, num_classes*4, bias=True)
            model.roi_heads.box_predictor.bbox_pred.requires_grad_(True)

    model = model.to(device)

    # Visualize feature maps
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    # if is_pretrained and not is_retrained:
    #     model.backbone.body.conv1.register_forward_hook(get_activation(f'conv1'))
    #     for i, module in enumerate(model.backbone.body):
    #         if i <= 3:  # non layer
    #             print(module)
    #         else:
    #             for j, bottleneck in enumerate(model.backbone.body[module]):  # necessarily bottleneck module
    #                 bottleneck.conv1.register_forward_hook(get_activation(f'conv{i}-{j}'))
    #     model.backbone.fpn.inner_blocks[0].register_forward_hook(get_activation(f'fpn1'))
    #     model.backbone.fpn.layer_blocks[0].register_forward_hook(get_activation(f'fpn2'))
    # else:
    #     model.backbone[0][0].register_forward_hook(get_activation(f'cell{0}'))  # cell preproc1 is necessarily ops.stdconv
    #     for i, module in enumerate(model.backbone):
    #         if isinstance(module, torchvision.models.mobilenet.InvertedResidual):
    #             module.conv[0].register_forward_hook(get_activation(f'cell{i}'))
    #     model.rpn.head.conv.register_forward_hook(get_activation(f'cellhead'))


    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0005,
                                momentum=0.9, weight_decay=0.0005)

    visualization_stem = "mobile"
    if is_fixed:
        visualization_stem = "mobile_rf"
    elif is_retrained:
        visualization_stem = "mobile_r"
    elif is_pretrained:
        visualization_stem = "resnet"
    for i in range(200):
        # for step, (image, targets) in enumerate(train_loader):
        #     targets = [{k: v.cuda() for k,v in label.items() if not isinstance(v, str)} for label in targets]
        #     output = model(image.to(device), targets)
        #     output = model(image, targets)
        X, y = next(iter(train_loader))
        X = torch.stack([image.to(device) for image in X])
        y = [{k: v.to(device) for k, v in t.items() if not isinstance(v, str)} for t in y]
        v_net = copy.deepcopy(model)
        losses = model.forward(X, y)
        virtual_step = sum(_loss for _loss in losses.values())
        gradients = torch.autograd.grad(virtual_step, model.parameters(), allow_unused=True)
        with torch.no_grad():
            bad_count = 0
            for w, vw, g, (name, _) in zip(model.parameters(), v_net.parameters(), gradients, list(model.named_parameters())[8:]):
                m = optimizer.state[w].get('momentum_buffer', 0.) * 0.9
                try:
                    vw.copy_(w - 0.025 * (m + g + w))
                except TypeError:
                    print(name, m , g)
                    bad_count += 1
                    if bad_count > 50:
                        exit()
        train_one_epoch_ssd(model, optimizer, train_loader, device, i, print_freq=10)
        model.eval()
        evaluate(model, val_loader, device=device, epoch=i)
        # os.makedirs(f"./tempSave/validate_obj/activations_{visualization_stem}/{i}/", exist_ok=True)
        # for q, key in enumerate(activation.keys()):
        #     act = activation[key][0] # take first of batch arbitrarily
        #     q_mult = min(q*4, 8)
        #     q_mult = max(q_mult, 2) # simplify axarr situation, enforcing always >2 x n fig.
        #     fig, axarr = plt.subplots(q_mult, 4)
        #     row_count = -1
        #     for idx in range(q_mult*4):
        #         if idx % 4 == 0:
        #             row_count += 1
        #         axarr[row_count, idx%4].imshow(act[idx].cpu().numpy())
        #         axarr[row_count, idx%4].set_axis_off()
        #     fig.savefig(f"./tempSave/validate_obj/activations_{visualization_stem}/{i}/{key}.png")
        #     plt.close(fig)


if __name__ == "__main__":
    main()
