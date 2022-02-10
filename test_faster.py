import torchvision
import torchvision.transforms as tf
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torch
import sys
import utils
import preproc

sys.path.insert(0, "/hdd/PhD/hem/perceptual")
from det_dataset import Imagenet_Det as Pure_Det
from search_obj import collate_fn
from detectionengine import evaluate

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
                       num_classes=2,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)\
        # .to(device)

    for i in range(10):
        for step, (image, targets) in enumerate(train_loader):
            # targets = [{k: v.cuda() for k,v in label.items() if not isinstance(v, str)} for label in targets]
            # output = model(image.to(device), targets)
            output = model(image, targets)
            model.eval()
            evaluate(model, train_loader, device=device)

if __name__ == "__main__":
    main()
