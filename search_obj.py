""" Search cell """

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from config import SearchConfig
import utils
from models.search_cnn import SearchCNNController
from models.search_cnn_obj import SearchCNNControllerObj
from architect import Architect
from visualize import plot
import torch.nn.functional as F
import time
import csv
from torchvision import transforms
import random

sys.path.insert(0, "./torchsample")
from torchvision.utils import save_image
import wandb
from detectionengine import evaluate
from mean_ap_mmdet import eval_map


config = SearchConfig()

device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)

sys.path.append('./detr/')

from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion
from detr.util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

import sys
sys.path.insert(0, "/home/matt/Documents/hem/perceptual")
from coco_obj import get_dict
from PIL import ImageDraw
import matplotlib.pyplot as plt

class_dict, rev_class_dict = get_dict()

# torch.multiprocessing.set_start_method('spawn') # https://github.com/pytorch/pytorch/issues/40403
# caused by putting dictionary elements onto gpu
def collate_fn(batch):
    data, labels = zip(*batch)
    stacked_data = torch.stack(data, dim=0)
    # good_indices = [q for q, label in enumerate(labels) if len(label["boxes"]) > 0]
    # stacked_data = stacked_data[good_indices]
    # labels = tuple(label for q, label in enumerate(labels) if q in good_indices)

    return stacked_data, labels


def get_split(dataset):
    n_train = len(dataset)
    split = n_train * 0.8
    remainder = split % 8
    return int(split - remainder)


def main():
    os.environ['WANDB_SILENT'] = "true"
    wandb.init(
        entity="mattpoyser",
        project="darts",
        config=config,
    )
    start_time = time.time()
    logger.info("Logger is set - training start {}".format(start_time))

    assert config.dataset == "coco_det"
    is_multi = False
    if config.dataset == "imageobj" or config.dataset == "cocomask":
        is_multi = True

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    # get data with meta info
    input_size, input_channels, n_classes, train_data, val_data = utils.get_data(
        config.dataset, config.data_path, cutout_length=0, validation=True, search=True,
        bede=config.bede, is_concat=config.is_concat)

    net_crit = nn.CrossEntropyLoss().to(device)

    # additional parameters for detr
    class_loss = None
    weight_dict = None

    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5}
    weight_dict['loss_giou'] = 2
    losses = ['labels', 'boxes', 'cardinality']
    net_crit = SetCriterion(n_classes, matcher=matcher, weight_dict=weight_dict,
                            eos_coef=0.1, losses=losses).to(device)
    class_loss = nn.NLLLoss().to(device)
    model = SearchCNNControllerObj(input_channels, config.init_channels, n_classes, config.layers,
                                   net_crit, device_ids=config.gpus, n_nodes=config.nodes, class_loss=class_loss,
                                   weight_dict=weight_dict)

    model = model.to(device)

    # weights optimizer
    w_optim = torch.optim.SGD(model.weights(), config.w_lr, momentum=config.w_momentum,
                              weight_decay=config.w_weight_decay)
    # alphas optimizer
    alpha_optim = torch.optim.Adam(model.alphas(), config.alpha_lr, betas=(0.5, 0.999),
                                   weight_decay=config.alpha_weight_decay)

    # split data to train/validation
    # split = get_split(train_data)
    # indices = list(range(len(train_data)))
    # random.seed(1337) # note must use same random seed as dataloader (and thus process same images)
    # random.shuffle(indices)
    # train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    # valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    if config.dynamic:
        train_sampler = None # do not sample as DynamicDataset does this automatically.
        # needs to be this way else dynamicdataset will process validation images + incorporate them
        # into the tree

    collate_func = collate_fn

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               # sampler=train_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True,
                                               collate_fn=collate_func
                                               )
    valid_loader = torch.utils.data.DataLoader(val_data,
                                               batch_size=config.batch_size,
                                               # sampler=valid_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True,
                                               collate_fn=collate_func
                                               )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, config.epochs, eta_min=config.w_lr_min)
    # print ("grep", config.workers, config.batch_size, config.name)
    # lr_scheduler = torch.optim.lr_scheduler.CyclicLR(w_optim, 0.001, 0.01, step_size_up=10,
    #                                                  step_size_down=None)  # step_size_down=None means same as _up

    architect = Architect(model, config.w_momentum, config.w_weight_decay)

    # training loop
    best_top1 = 0.
    hardness = None
    just_updated = True
    old_loss = 0
    print_mode = False
    non_update_epochs = 0
    top1 = None
    if config.dynamic and not config.nosave:
        save_indices(train_loader.dataset.get_printable(), 0)
    start_epoch = 0
    just_loaded = False

    mastery_threshold_count = 5
    mastery_epochs = [(i + 1) * (config.epochs // mastery_threshold_count) for i in range(mastery_threshold_count)]
    masteries = [config.mastery + (i * 0.35 * config.mastery) for i in range(mastery_threshold_count)]

    if config.resume is not None:
        if os.path.isfile(config.resume):
            print("==> loading checkpoint '{}'".format(config.resume))
            checkpoint = torch.load(config.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            w_optim.load_state_dict(checkpoint['w_optimizer'])
            alpha_optim.load_state_dict(checkpoint['a_optimizer'])
            best_top1 = checkpoint['best_top1']
            just_loaded = True
            print(f"loading at epoch {start_epoch}")

            # load up last used dataset. hardness values will not be stored, but the dataset used will be at least.
            # n.b. accuracies.out & visualized dataset.png should be accurate since these are calculated from the
            # raw csv files
            if config.ncc:
                indices_dir = f"/home2/lgfm95/nas/darts/tempSave/curriculums/{config.name}/"
            else:
                indices_dir = f"/hdd/PhD/nas/pt.darts/tempSave/curriculums/{config.name}/"
            indices_files = os.listdir(indices_dir)
            highest = 0
            for file in indices_files:
                if config.dataset in file and file.endswith('.csv'):
                    epoch_num = int(file[file.rindex("_") + 1:-4])
                    if epoch_num > highest:
                        highest = epoch_num
            print(f"loading indices from {f'{indices_dir}indices_{config.name}_{highest}.csv'}")
            with open(os.path.join(indices_dir, f"indices_{config.name}_{highest}.csv"), 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=' ')
                temp = list(csv_reader)
                temp = np.array(temp).flatten()
                assert len(
                    temp) == config.subset_size, f"number of images in dynamic dataset in checkpoint different to now: {len(temp)} and {config.subset_size} respectively"
                if train_loader.dataset.convert_to_paths:
                    temp = list(map(int, temp))
                    train_loader.dataset.idx = [train_loader.dataset.train_indices.index(idx) for idx in
                                                temp]  # assumes train_indices remains constant over different experiment iterations (should do as seeding w/in dataloader)
                else:
                    train_loader.dataset.idx = list(map(int, temp))
        else:
            print("resume pth file not found")

    # TODO: seperate counter for training epochs as opposed to training / dataset update combined
    for epoch in range(start_epoch, config.epochs):
        if config.curriculum:
            if epoch in mastery_epochs:
                config.mastery = masteries[mastery_epochs.index(epoch)]
                print("mastery threshold change reached")
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]

        model.print_alphas(logger)

        epoch_type = 1  # epoch type is a train epoch rather than dataset update by default
        # (ie for case of loading from resume pth file)
        if not just_loaded:
            epoch_type = get_epoch_type(epoch, hardness, top1)

        if epoch_type or just_updated or not config.dynamic:  # 1 is train, as normal (0 is dataset update)
            just_updated = False
            just_loaded = False
            # training
            hardness, _ = train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch,
                                      is_multi)
            # if config.dynamic:
            #     train_loader.dataset.update_correct(correct)
            #     if config.ncc and config.visualize:
            #         train_loader.dataset.visualize()

            # validation
            cur_step = (epoch + 1) * len(train_loader)
            top1, new_loss = validate(valid_loader, model, epoch, cur_step, print_mode, is_multi, config)
            wandb.log({"acc": top1, "loss": new_loss})

            if print_mode:
                print_mode = False
            # log
            # genotype
            genotype = model.genotype()
            logger.info("genotype = {}".format(genotype))

            # genotype as a image
            plot_path = os.path.join(config.plot_path, "EP{:02d}".format(epoch + 1))
            caption = "Epoch {}".format(epoch + 1)
            # plot(genotype.normal, plot_path + "-normal", caption)
            # plot(genotype.reduce, plot_path + "-reduce", caption)

            # save
            if best_top1 < top1:
                best_top1 = top1
                best_genotype = genotype
                is_best = True
            else:
                is_best = False
            # utils.save_checkpoint(model, config.path, is_best)  # TODO redundant checkpoint save
            print("")
            if config.early_stopping:
                if abs(old_loss - new_loss) < 0.0005:
                    print("stopping early")
                    break
            old_loss = new_loss
            non_update_epochs += 1
        else:
            print("updating subset")
            if config.mining:
                train_loader.dataset.update_subset(hardness, epoch, mining=True)
            else:
                try:
                    train_loader.dataset.update_subset(hardness, epoch)
                except IndexError:
                    raise AttributeError(hardness)
            save_indices(train_loader.dataset.get_printable(), epoch, [item for item in train_loader.dataset.cur_set])

            # set lr_scheduler to same as when started.
            # TODO configure such that does not necessarily start at "first epoch" -
            # do we even want this? starting at 'first epoch' means back to coarse tune, which is exactly what we want
            # if it were to start at a 'later epoch' then we have fine tuning, which we don't necessarily want.
            # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #     w_optim, config.epochs, eta_min=config.w_lr_min)
            for i in range((non_update_epochs + 10) % 20):
                lr_scheduler.step()  # keep stepping until reach peak of cycle, where lr is highest
                # step_size determines how many iterations between full half of a cycle.
            # %20 would represent distance to troughs, so add 10 to find distance to peaks. therefore, highest lr
            # just after update.
            just_updated = True
            print_mode = True

        # print ("grep {}".format(top1))
        if config.is_csv and top1 > 0.95:
            # if len(train_loader.dataset.idx) == 1:
            #     save_indices(train_loader.dataset.idx[0])
            # else:
            save_indices(train_loader.dataset.get_printable(), epoch)

        if config.resume is not None:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'w_optimizer': w_optim.state_dict(),
                'a_optimizer': alpha_optim.state_dict(),
                'best_top1': best_top1
            }, config.resume)

        if config.best_resume is not None:
            if is_best:
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'w_optimizer': w_optim.state_dict(),
                    'a_optimizer': alpha_optim.state_dict(),
                    'best_top1': best_top1
                }, config.best_resume)

        logger.info("Time after epoch {}: {} @ accuracy {}".format(epoch, time.time() - start_time, best_top1))

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("Best Genotype = {}".format(best_genotype))
    logger.info("Training end {}".format(time.time() - start_time))
    wandb.finish()


def save_indices(data, epoch, images=None):
    if not config.badpath and not config.nosave:
        if config.ncc:
            # with open(f'/home2/lgfm95/nas/darts/tempSave/curriculums/{config.name}/indices_{config.name}_{epoch}.csv', 'w') as csv_file:
            #     csv_writer = csv.writer(csv_file, delimiter=' ')
            #     csv_writer.writerow(data)
            if images is not None:
                print(f"about to save curriculum; {len(images)}")
                image_dir = f'/home2/lgfm95/nas/darts/tempSave/curriculums/{config.name}/{config.minedimagefname}_{epoch}'
                print(f"saving curriculum in {image_dir}")
                os.makedirs(image_dir)
                for q, image in enumerate(images):
                    image.save(image_dir + f"{q}.png")
            else:
                print("images is none")

        else:
            with open(f'/hdd/PhD/nas/pt.darts/tempSave/curriculums/{config.name}/indices_{config.name}_{epoch}.csv',
                      'w') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=' ')
                csv_writer.writerow(data)


def train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch, is_multi):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch * len(train_loader)
    writer.add_scalar('train/lr', lr, cur_step)

    model.train()

    hardness = [None for i in range(len(train_loader))]
    correct = [None for i in range(len(train_loader))]

    batch_size = config.batch_size
    print ("summed weight: ", sum([torch.sum(weight) for weight in model.weights()]))
    print ("total epoch duration", len(train_loader), len(valid_loader))
    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):

        for q, image in enumerate(trn_X): # working
            target = trn_y[q]
            image = transforms.ToPILImage()(image)
            draw = ImageDraw.Draw(image)
            for k in range(len(target["boxes"])):
                draw.rectangle(np.array(target["boxes"][k]))
                draw.text((target['boxes'][k][0].item() + 2, target['boxes'][k][1].item() + 2), str(rev_class_dict[target['labels'][k].item()]))
            image.save(f"tempSave/validate_obj/coco/{target['image_id'].item()}.png")

        trn_X = trn_X.to(device, non_blocking=True)
        val_X = val_X.to(device, non_blocking=True)

        N = trn_X.size(0)
        # print("grep label shape", trn_y)

        # phase 2. architect step (alpha)
        alpha_optim.zero_grad()
        architect.unrolled_backward(trn_X, trn_y, val_X, val_y, lr, w_optim, is_multi)
        alpha_optim.step()

        # phase 1. child network step (w)
        w_optim.zero_grad()
        logits, detections, new_hardness = model(trn_X, trn_y, full_ret=True)

        # delete when ready:
        # for i in range(len(detections)): # iterate over batch
        #     try:
        #         obs_detections = torch.cat((detections[i]['boxes'], detections[i]['labels'].unsqueeze(1)), dim=1)# ndarray(m, 5)
        #     except RuntimeError:
        #         raise AttributeError(detections[i]['boxes'], detections[i]['labels'], detections[i]['boxes'].shape, detections[i]['labels'].shape)
        #     gt_detections = trn_y[i]['boxes'] # ndarray(n, 4)
        #     eval_map(obs_detections, gt_detections)

        # we need judge of predictions vs labels.
        # using just classes is not any simpler, since we need associations between given
        # multilabel, multiclass prediction and ground truth, which will have differing size.
        # therefore, we need associations in built into our hardness calculator, i.e. we need
        # to use use location of the prediction.

        # modified to return detections even if not in eval mode
        # 0. per image (rather than per batch as evaluate does): TODO
            # 1. compute res from detections as per detectionengine evaluate
            # 2. update cocoevaluator
            # 3. accumulate evaluator -> recall, precision etc.
            # 4. use recall, precision and scores to formulate hardness.

        loss = sum(_loss for _loss in logits.values())
        losses.update(loss.item(), N)

        new_hardness = [1-new_hardness[i].item() for i in range(len(new_hardness))]
        hardness[(step * batch_size):(step * batch_size) + batch_size] = new_hardness  # assumes batch 1 takes idx 0-8, batch 2 takes 9-16, etc.

        # correct[(step * batch_size):(step * batch_size) + batch_size] = new_correct
        print(step, batch_size, step * batch_size, (step * batch_size) + batch_size,# len(new_correct),
              len(train_loader), len(valid_loader))
        print(hardness, len(hardness))

        loss.backward()

        # gradient clipping
        nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()

        if step % config.print_freq == 0 or step == len(train_loader) - 1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.4f} "
                    .format(
                    epoch + 1, config.epochs, step, len(train_loader) - 1, losses=losses))
        cur_step += 1

        if config.dynamic and config.visualize:
            os.makedirs(f"./tempSave/validate_obj/activations/{epoch}/", exist_ok=True)
            for key in model.net.activation.keys():
                act = model.net.activation[key].squeeze()
                qmult = int(act.size(0)/4)
                idx_range = act.size(0)
                if key == 'cellhead':
                    qmult = 8
                    idx_range = 32
                fig, axarr = plt.subplots(qmult, 4)
                row_count = -1
                for idx in range(idx_range):
                    if idx % 4 == 0:
                        row_count += 1
                    axarr[row_count, idx%4].imshow(act[idx].cpu().numpy())
                    axarr[row_count, idx%4].set_axis_off()
                fig.savefig(f"./tempSave/validate_obj/activations/{epoch}/{key}.png")
                plt.close(fig)
        print(losses.avg)

    logger.info("Train: [{:2d}/{}] Final Loss {:.4%}".format(epoch + 1, config.epochs, losses.avg))

    return hardness, correct


def validate(valid_loader, model, epoch, cur_step, print_mode, is_multi, config):
    model.eval()
    evaluate(model, valid_loader, device=device, epoch=epoch)
    return 0, 0


def train_hardness(train_loader, model):
    hardness = [None for i in range(len(train_loader.dataset))]
    len_hard = len(hardness)

    for step, (trn_X, trn_y) in enumerate(train_loader.dataset):
        # print("step", step)
        # print("len trnX", len(trn_X), trn_X[0])
        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
        N = trn_X.size(0)

        logits = model(trn_X)
        new_hardness = get_hardness(logits.cpu(), trn_y.cpu())
        hardness[(step * N):(step * N) + N] = new_hardness  # assumes batch 1 takes idx 0-N, batch 2 takes N+1-2N, etc.

    # raise AttributeError(len(hardness), len_hard, hardness, "hardness post process")
    return hardness


# low value for hardness means harder.
def get_hardness(output, target, is_multi):
    if not is_multi:
        # currently a binary association between correct classication => 0.8
        # we want it to be a softmax representation. if we instead take crossentropy loss of each individual cf target
        _, predicted = torch.max(output.data, 1)
        confidence = F.softmax(output, dim=1)
        hardness_scaler = np.where((predicted == target), 1,
                                   0.1)  # if correct, simply use confidence as measure of hardness
        # therefore if model can easily say yep this is object X, then confidence will be high. if it only just manages to identify
        # object X, confidence if lower
        # if object X is misclassified, hardness needs to be lower still.
        # assumes that it does not confidently misclassify.
        hardness = [(confidence[i][predicted[i]] * hardness_scaler[i]).item() for i in range(output.size(0))]
    else:
        output = torch.sigmoid(output.float()).detach()
        output[output > 0.5] = 1
        output[output <= 0.5] = 0
        confidence = F.softmax(output, dim=1)

        hardness_scaler = []
        hardness = []
        assert len(output) == len(target)  # should both be equal to batch size
        for q in range(len(output)):
            assert len(output[q]) == len(target[q])  # should both be equal to num_classes eg 184
            correct_avg = (np.array(output[q]) == np.array(target[q])).sum() / len(output[q])
            if correct_avg > 0.5:  # this could be another threshold we change, or have it == hardness threshold
                hardness_scaler.append(1)
            else:
                hardness_scaler.append(0.1)

            correct = np.where(np.array(output[q]) == np.array(target[q]))[0]
            hardness_value = [confidence[q][i] * 1 if i in correct else confidence[q][i] * 0.1 for i in
                              range(len(output[q]))]
            hardness.append(sum(hardness_value) / len(output[q]))

        hardness_scaler = np.asarray(hardness_scaler)
        hardness = np.array(hardness)
        # raise AttributeError(output, target, hardness_scaler, hardness)

    print("hardness scaler", len(hardness_scaler))
    return hardness, hardness_scaler


def get_epoch_type(epoch, hardness, top1):
    # naive alternate, starting with normal training
    if not config.dynamic or epoch < config.init_train_epochs:
        return 1
    is_mastered = get_mastered(hardness, top1)
    if is_mastered:
        print("mastered, therefore epoch type 0")
        return 0
    print("not mastered, therefore epoch type 1")
    return 1


def get_mastered(hardness, top1):
    # if fraction of times where image is unconfidently/mis-classified is less than mastery threshold
    # TODO use hardness across history eg mean hardness over last 5
    # print("ahard", "\n")
    # for aHard in hardness:
    # print("ahard", aHard)
    # print("len hardness", len(hardness))
    # print("len hard ones", np.where(np.array(hardness) > 0.5))
    # print("len hard ones", len(np.where(np.array(hardness) > 0.5)[0]))
    # print("hardness calculations: ", (len(np.where(np.array(hardness) > config.hardness)[0]) / len(hardness)), config.mastery)

    # if percentage of items considered hard exceeds a mastery threshold, update the subset.
    if top1 is None:
        if (len(np.where(np.array(hardness) > config.hardness)[0]) / len(hardness)) < config.mastery:
            print("therefore not mastered")
            return 0
    else:
        # print("grep working", top1)
        if top1 < config.mastery:
            return 0
    # if len(np.where(np.array(hardness) < config.mastery)) > len(hardness)-2:
    # a lot of images still being misclassified
    # return 0
    print("therefore mastered")
    return 1


if __name__ == "__main__":
    main()
