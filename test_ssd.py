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
from detectionengine import train_one_epoch, train_one_epoch_ssd, evaluate

from ssd import ssd

def main():
    train_transforms, _ = preproc.data_transforms("pure_det", cutout_length=0)
    _, _, _, train_data, _ = utils.get_data(
        "pure_det", "", cutout_length=0, validation=True, search=True,
        bede=False, is_concat=False)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=1,
                                               # sampler=train_sampler,
                                               num_workers=0,
                                               pin_memory=True,
                                               collate_fn=collate_fn
                                               )


    device = torch.device("cuda")

    # Visualize feature maps
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    model = ssd.build_ssd('train', size=300, num_classes=200)

    for i, module in enumerate(model.vgg):
        if isinstance(module, torch.nn.modules.conv.Conv2d):
            module.register_forward_hook(get_activation(f'cell{i}'))
    for i, module in enumerate(model.loc):
        if isinstance(module, torch.nn.modules.conv.Conv2d):
            module.register_forward_hook(get_activation(f'loc{i}'))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    for i in range(10):
        # for step, (image, targets) in enumerate(train_loader):
        #     targets = [{k: v.cuda() for k,v in label.items() if not isinstance(v, str)} for label in targets]
        #     output = model(image.to(device), targets)
        #     output = model(image, targets)
        train_one_epoch_ssd(model, optimizer, train_loader, device, i, print_freq=10)
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
