import torchvision
import torchvision.transforms as tf
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torch
import sys
import utils
import preproc
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/hdd/PhD/hem/perceptual")
from det_dataset import Imagenet_Det as Pure_Det
from search_obj import collate_fn
from detectionengine import train_one_epoch, evaluate

# from torchvision._internally_replaced_utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

is_pretrained = eval(sys.argv[sys.argv.index('--obj_pretrained')+1])
is_retrained = eval(sys.argv[sys.argv.index('--obj_retrained')+1])
is_fixed = eval(sys.argv[sys.argv.index('--obj_fixed')+1])
dataset = sys.argv[sys.argv.index('--dataset')+1]


def main():
    num_classes = 200
    if dataset == "pure_det":
        train_transforms, _ = preproc.data_transforms("pure_det", cutout_length=0)
        # full_set = Pure_Det(train_path, train_transforms)
        _, _, _, train_data, _ = utils.get_data(
            "pure_det", "", cutout_length=0, validation=True, search=True,
            bede=False, is_concat=False)
    elif dataset == "coco_det":
        train_transforms, _ = preproc.data_transforms("coco_det", cutout_length=0)
        _, _, _, train_data, _ = utils.get_data(
            "coco_det", "", cutout_length=0, validation=True, search=True,
            bede=False, is_concat=False)
        num_classes = 91
    else:
        raise AttributeError("bad dataset")
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=1,
                                               # sampler=train_sampler,
                                               num_workers=0,
                                               pin_memory=True,
                                               collate_fn=collate_fn
                                               )
    if is_pretrained:
        backbone = resnet_fpn_backbone('resnet50', False, trainable_layers=0)
        num_classes = 91
    else:
        # load a pre-trained model for classification and return
        # only the features
        backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        # FasterRCNN needs to know the number of
        # output channels in a backbone. For mobilenet_v2, it's 1280
        # so we need to add it here
        backbone.out_channels = 1280

    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    # anchor_sizes = ((32, 64, 128, 256, 512, ), ) * 3
    # aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    # anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    device = torch.device("cuda")
    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       # rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler,
                       box_score_thresh=0.001)

    if is_pretrained:
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth', progress=True)
        model.load_state_dict(state_dict)
        if is_retrained:
            if is_fixed:
                # fix non backbone weights
                for param in model.parameters():
                    param.requires_grad = False

            # load back in new (untrained) backbone
            backbone = torchvision.models.mobilenet_v2(pretrained=True).features
            backbone[-1][0] = torch.nn.Conv2d(320, 256, kernel_size=(3,3), stride=(1,1), bias=False)
            backbone[-1][1] = torch.nn.BatchNorm2d(256)
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

    if is_pretrained and not is_retrained:
        model.backbone.body.conv1.register_forward_hook(get_activation(f'conv1'))
        for i, module in enumerate(model.backbone.body):
            if i <= 3:  # non layer
                print(module)
            else:
                for j, bottleneck in enumerate(model.backbone.body[module]):  # necessarily bottleneck module
                    bottleneck.conv1.register_forward_hook(get_activation(f'conv{i}-{j}'))
        model.backbone.fpn.inner_blocks[0].register_forward_hook(get_activation(f'fpn1'))
        model.backbone.fpn.layer_blocks[0].register_forward_hook(get_activation(f'fpn2'))
    else:
        model.backbone[0][0].register_forward_hook(get_activation(f'cell{0}'))  # cell preproc1 is necessarily ops.stdconv
        for i, module in enumerate(model.backbone):
            if isinstance(module, torchvision.models.mobilenet.InvertedResidual):
                module.conv[0].register_forward_hook(get_activation(f'cell{i}'))
        model.rpn.head.conv.register_forward_hook(get_activation(f'cellhead'))


    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
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
        train_one_epoch(model, optimizer, train_loader, device, i, print_freq=10)
        model.eval()
        evaluate(model, train_loader, device=device, epoch=i)
        os.makedirs(f"./tempSave/validate_obj/activations_{visualization_stem}/{i}/", exist_ok=True)
        for q, key in enumerate(activation.keys()):
            act = activation[key].squeeze()
            q_mult = min(q*4, 8)
            fig, axarr = plt.subplots(q_mult, 4)
            row_count = -1
            for idx in range(q_mult*4):
                if idx % 4 == 0:
                    row_count += 1
                axarr[row_count, idx%4].imshow(act[idx].cpu().numpy())
                axarr[row_count, idx%4].set_axis_off()
            fig.savefig(f"./tempSave/validate_obj/activations_{visualization_stem}/{i}/{key}.png")
            plt.close(fig)


if __name__ == "__main__":
    main()
