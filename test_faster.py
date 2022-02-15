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

sys.path.insert(0, "/hdd/PhD/hem/perceptual")
from det_dataset import Imagenet_Det as Pure_Det
from search_obj import collate_fn
from detectionengine import train_one_epoch, evaluate

def main():
    train_transforms, _ = preproc.data_transforms("pure_det", cutout_length=0)
    # full_set = Pure_Det(train_path, train_transforms)
    _, _, _, train_data, _ = utils.get_data(
        "pure_det", "", cutout_length=0, validation=True, search=True,
        bede=False, is_concat=False)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=2,
                                               # sampler=train_sampler,
                                               num_workers=0,
                                               pin_memory=True,
                                               collate_fn=collate_fn
                                               )
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
                       num_classes=200,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler,
                       box_score_thresh=0.001)\
        .to(device)

    # Visualize feature maps
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    model.backbone[0][0].register_forward_hook(get_activation(f'cell{0}'))  # cell preproc1 is necessarily ops.stdconv

    for i, module in enumerate(model.backbone):
        if isinstance(module, torchvision.models.mobilenet.InvertedResidual):
            module.conv[0].register_forward_hook(get_activation(f'cell{i}'))
    model.rpn.head.conv.register_forward_hook(get_activation(f'cellhead'))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    for i in range(10):
        # for step, (image, targets) in enumerate(train_loader):
        #     targets = [{k: v.cuda() for k,v in label.items() if not isinstance(v, str)} for label in targets]
        #     output = model(image.to(device), targets)
        #     output = model(image, targets)
        train_one_epoch(model, optimizer, train_loader, device, i, print_freq=10)
        model.eval()
        evaluate(model, train_loader, device=device, epoch=i)
        os.makedirs(f"./tempSave/validate_obj/activations_mobile/{i}/", exist_ok=True)
        for key in activation.keys():
            act = activation[key].squeeze()
            raise AttributeError([activation[acts].shape for acts in activation])
            fig, axarr = plt.subplots(int(act.size(0)/4), 4)
            row_count = -1
            for idx in range(act.size(0)):
                if idx % 4 == 0:
                    row_count += 1
                axarr[row_count, idx%4].imshow(act[idx].cpu().numpy())
                axarr[row_count, idx%4].set_axis_off()
            fig.savefig(f"./tempSave/validate_obj/activations_mobile/{i}/{key}.png")
            plt.close(fig)

if __name__ == "__main__":
    main()
