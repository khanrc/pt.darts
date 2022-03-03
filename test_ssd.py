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
from coco_obj import COCODetLoader as Coco_Det
from search_obj import collate_fn
from detectionengine import train_one_epoch, train_one_epoch_ssd, evaluate

from ssd import ssd
from ssd.layers.modules import MultiBoxLoss

def main():
    # train_transforms, _ = preproc.data_transforms("coco_det", cutout_length=0)
    # _, _, _, train_data, _ = utils.get_data(
    #     "coco_det", "", cutout_length=0, validation=True, search=True,
    #     bede=False, is_concat=False)
    isize = 300
    assert isize == 300, "fixed input size, and only size 300 is supported"
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    normalize = [
        tf.ToTensor(),
        tf.Normalize(MEAN, STD)
    ]
    transf = [
        tf.Resize((isize, isize)),
        # transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1)
    ]
    train_transforms = tf.Compose(transf + normalize)
    train_path = "/hdd/PhD/data/coco"
    train_data = Coco_Det(train_path=train_path, transforms=train_transforms, max_size=1000)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=32, # cheat for now. ensures same number of objects
                                               # per image in batch because only one image.
                                               # sampler=train_sampler,
                                               num_workers=0,
                                               pin_memory=True,
                                               collate_fn=collate_fn
                                               )

    num_classes = 80


    device = torch.device("cuda")

    # Visualize feature maps
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    model = ssd.build_ssd('train', size=isize, num_classes=num_classes).to(device)
    # model.load_weights("./ssd/weights/ssd300_mAP_77.43_v2.pth")
    model.vgg.load_state_dict(torch.load("./ssd/weights/vgg16_reducedfc.pth"))

    for i, module in enumerate(model.vgg):
        if isinstance(module, torch.nn.modules.conv.Conv2d):
            module.register_forward_hook(get_activation(f'cell{i}'))
    for i, module in enumerate(model.loc):
        if isinstance(module, torch.nn.modules.conv.Conv2d):
            module.register_forward_hook(get_activation(f'loc{i}'))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=4e-3,
                                momentum=0.9, weight_decay=0.0005)
    criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5,
                             False, True)

    for i in range(10):
        # for step, (image, targets) in enumerate(train_loader):
        #     targets = [{k: v.cuda() for k,v in label.items() if not isinstance(v, str)} for label in targets]
        #     output = model(image.to(device), targets)
        #     output = model(image, targets)
        train_one_epoch_ssd(model, optimizer, criterion, train_loader, device, i, print_freq=10)
        model.eval()
        evaluate(model, train_loader, device=device, epoch=i)
        os.makedirs(f"./tempSave/validate_obj/activations_ssd/{i}/", exist_ok=True)
        for q, key in enumerate(activation.keys()):
            if key == 'cellhead':
                raise AttributeError(activation[key].shape)
            act = activation[key].squeeze()
            q_mult = min(q*4, 8)
            fig, axarr = plt.subplots(q_mult, 4)
            row_count = -1
            for idx in range(q_mult*4):
                if idx % 4 == 0:
                    row_count += 1
                axarr[row_count, idx%4].imshow(act[idx].cpu().numpy())
                axarr[row_count, idx%4].set_axis_off()
            fig.savefig(f"./tempSave/validate_obj/activations_ssd/{i}/{key}.png")
            plt.close(fig)


if __name__ == "__main__":
    main()
