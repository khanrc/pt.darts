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
from architect import Architect
from visualize import plot
import torch.nn.functional as F
import time
import csv
from torchvision import transforms
sys.path.insert(0, "./torchsample")

config = SearchConfig()

device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)


def main():
    start_time = time.time()
    logger.info("Logger is set - training start {}".format(start_time))

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    # get data with meta info
    input_size, input_channels, n_classes, train_data, val_data = utils.get_data(
        config.dataset, config.data_path, cutout_length=0, validation=True, search=True)

    net_crit = nn.CrossEntropyLoss().to(device)
    if config.dataset == "imageobj":
        net_crit = nn.BCEWithLogitsLoss().to(device)
    model = SearchCNNController(input_channels, config.init_channels, n_classes, config.layers,
                                net_crit, device_ids=config.gpus)
    model = model.to(device)

    # weights optimizer
    w_optim = torch.optim.SGD(model.weights(), config.w_lr, momentum=config.w_momentum,
                              weight_decay=config.w_weight_decay)
    # alphas optimizer
    alpha_optim = torch.optim.Adam(model.alphas(), config.alpha_lr, betas=(0.5, 0.999),
                                   weight_decay=config.alpha_weight_decay)

    # split data to train/validation
    # n_train = len(train_data)
    # split = n_train // 2
    # indices = list(range(n_train))
    # train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    # valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               # sampler=train_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(val_data,
                                               batch_size=config.batch_size,
                                               # sampler=valid_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     w_optim, config.epochs, eta_min=config.w_lr_min)
    print ("grep", config.workers, config.batch_size, config.name)
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(w_optim, 0.001, 0.01, step_size_up=10, step_size_down=None) # step_size_down=None means same as _up

    architect = Architect(model, config.w_momentum, config.w_weight_decay)

    # training loop
    best_top1 = 0.
    hardness = None
    just_updated = True
    old_loss = 0
    print_mode = False
    non_update_epochs = 0
    top1 = None
    is_multi = False
    if config.dataset == "imageobj" or config.dataset == "cocomask":
        is_multi = True
    save_indices(train_loader.dataset.get_printable(), 0)

    # TODO: seperate counter for training epochs as opposed to training / dataset update combined
    for epoch in range(config.epochs):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]

        model.print_alphas(logger)

        epoch_type = get_epoch_type(epoch, hardness, top1)

        if epoch_type or just_updated or not config.dynamic: # 1 is train, as normal (0 is dataset update)
            just_updated = False
            # training
            hardness, correct = train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch, is_multi)
            if config.dynamic:
                train_loader.dataset.update_correct(correct)
                if config.ncc and config.visualize:
                    train_loader.dataset.visualize()

            # validation
            cur_step = (epoch+1) * len(train_loader)
            top1, new_loss = validate(valid_loader, model, epoch, cur_step, print_mode)

            if print_mode:
                print_mode = False
            # log
            # genotype
            genotype = model.genotype()
            logger.info("genotype = {}".format(genotype))

            # genotype as a image
            plot_path = os.path.join(config.plot_path, "EP{:02d}".format(epoch+1))
            caption = "Epoch {}".format(epoch+1)
            # plot(genotype.normal, plot_path + "-normal", caption)
            # plot(genotype.reduce, plot_path + "-reduce", caption)

            # save
            if best_top1 < top1:
                best_top1 = top1
                best_genotype = genotype
                is_best = True
            else:
                is_best = False
            utils.save_checkpoint(model, config.path, is_best)
            print("")
            if config.early_stopping:
                if abs(old_loss - new_loss) < 0.0005:
                    print("stopping early")
                    break
            old_loss = new_loss
            non_update_epochs += 1
        else:
            print("updating subset")
            train_loader.dataset.update_subset(hardness, epoch)
            save_indices(train_loader.dataset.get_printable(), epoch)

            # set lr_scheduler to same as when started.
            # TODO configure such that does not necessarily start at "first epoch" -
            # do we even want this? starting at 'first epoch' means back to coarse tune, which is exactly what we want
            # if it were to start at a 'later epoch' then we have fine tuning, which we don't necessarily want.
            # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #     w_optim, config.epochs, eta_min=config.w_lr_min)
            for i in range((non_update_epochs+10) % 20):
                lr_scheduler.step() # keep stepping until reach peak of cycle, where lr is highest
                # step_size determines how many iterations between full half of a cycle.
            # %20 would represent distance to troughs, so add 10 to find distance to peaks. therefore, highest lr
            # just after update.
            just_updated = True
            print_mode = True

        print ("grep {}".format(top1))
        if config.is_csv and top1 > 0.95:
            # if len(train_loader.dataset.idx) == 1:
            #     save_indices(train_loader.dataset.idx[0])
            # else:
            save_indices(train_loader.dataset.get_printable())

        logger.info("Time after epoch {}: {} @ accuracy {}".format(epoch, time.time()-start_time, best_top1))

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("Best Genotype = {}".format(best_genotype))
    logger.info("Training end {}".format(time.time()-start_time))

def save_indices(data, epoch):
    with open(f'/home2/lgfm95/nas/darts/tempSave/curriculums/indices_{config.dataset}_{epoch}.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=' ')
        csv_writer.writerow(data)



def train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch, is_multi):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch*len(train_loader)
    writer.add_scalar('train/lr', lr, cur_step)

    model.train()

    hardness = [None for i in range(len(train_loader))]
    correct = [None for i in range(len(train_loader))]

    batch_size = config.batch_size
    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):
        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
        val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(device, non_blocking=True)
        N = trn_X.size(0)
        print("grep label shape", trn_y)
        if step < 200 and not is_multi:
            for q, im in enumerate(trn_X):
                toSave = transforms.ToPILImage()(im.cpu())
                savePath = "./tempSave/gt/{}-{}.png".format((step * batch_size) + q, trn_y[q])
                toSave.save(savePath)

        # phase 2. architect step (alpha)
        alpha_optim.zero_grad()
        architect.unrolled_backward(trn_X, trn_y, val_X, val_y, lr, w_optim, is_multi)
        alpha_optim.step()

        # phase 1. child network step (w)
        w_optim.zero_grad()
        logits = model(trn_X)
        if is_multi:
            loss = model.criterion(logits, trn_y.float())
        else:
            loss = model.criterion(logits, trn_y)
        new_hardness, new_correct = get_hardness(logits.cpu(), trn_y.cpu(), is_multi)
        loss.backward()
        hardness[(step*batch_size):(step*batch_size)+batch_size] = new_hardness # assumes batch 1 takes idx 0-8, batch 2 takes 9-16, etc.
        correct[(step*batch_size):(step*batch_size)+batch_size] = new_correct
        # gradient clipping
        # nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()

        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.4f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1, top5=top5))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))

    return hardness, correct

def validate(valid_loader, model, epoch, cur_step, print_mode):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits = model(X)
            loss = model.criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if print_mode:
                if step < 5:
                    for q, im in enumerate(X):
                        toSave = transforms.ToPILImage()(im.cpu())
                        savePath = "./tempSave/postupdate/{}-{}.png".format((step * 8) + q, y[q])
                        toSave.save(savePath)

            if step % config.print_freq == 0 or step == len(valid_loader)-1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.4f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, config.epochs, step, len(valid_loader)-1, losses=losses,
                        top1=top1, top5=top5))
            if step > 100:
                break

    writer.add_scalar('val/loss', losses.avg, cur_step)
    writer.add_scalar('val/top1', top1.avg, cur_step)
    writer.add_scalar('val/top5', top5.avg, cur_step)

    logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))

    return top1.avg, losses.avg


def train_hardness(train_loader, model):
    hardness = [None for i in range(len(train_loader.dataset))]
    len_hard = len(hardness)

    for step, (trn_X, trn_y) in enumerate(train_loader.dataset):
        print("step", step)
        print("len trnX", len(trn_X), trn_X[0])
        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
        N = trn_X.size(0)

        logits = model(trn_X)
        new_hardness = get_hardness(logits.cpu(), trn_y.cpu())
        hardness[(step*N):(step*N)+N] = new_hardness # assumes batch 1 takes idx 0-N, batch 2 takes N+1-2N, etc.

    # raise AttributeError(len(hardness), len_hard, hardness, "hardness post process")
    return hardness


# low value for hardness means harder.
def get_hardness(output, target, is_multi):
    if not is_multi:
        # currently a binary association between correct classication => 0.8
        # we want it to be a softmax representation. if we instead take crossentropy loss of each individual cf target
        _, predicted = torch.max(output.data, 1)
        confidence = F.softmax(output, dim=1)
        hardness_scaler = np.where((predicted == target), 1, 0.1) # if correct, simply use confidence as measure of hardness
        # therefore if model can easily say yep this is object X, then confidence will be high. if it only just manages to identify
        # object X, confidence if lower
        # if object X is misclassified, hardness needs to be lower still.
        # assumes that it does not confidently misclassify.
        hardness = [(confidence[i][predicted[i]] * hardness_scaler[i]).item() for i in range(output.size(0))]
    else:
        output = torch.sigmoid(output.float())
        output[output>0.5] = 1
        output[output<=0.5] = 0
        confidence = F.softmax(output, dim=1)

        hardness_scaler = []
        hardness = []
        assert len(output) == len(target) # should both be equal to batch size
        for q in range(len(output)):
            assert len(output[q]) == len(target[q]) # should both be equal to num_classes eg 184
            correct_avg = (np.array(output[q]) == np.array(target[q])).sum() / len(output[q])
            if correct_avg > 0.5: # this could be another threshold we change, or have it == hardness threshold
                hardness_scaler.append(1)
            else:
                hardness_scaler.append(0.1)

            confidence_avg = confidence[q] / len(output[q])
            hardness.append(confidence_avg*hardness_scaler[q])
            # TODO this is average confidence multiplied
            # by average scaler across all labels in an image
            # is this equivalent to confidence * scaler across all labels, then averaged?
        hardness_scaler = np.array(hardness_scaler)
        hardness = np.array(hardness)
        raise AttributeError(output, target, hardness_scaler, hardness)

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
    print("ahard", "\n")
    for aHard in hardness:
        print("ahard", aHard)
    print("len hardness", len(hardness))
    print("len hard ones", np.where(np.array(hardness) > 0.5))
    print("len hard ones", len(np.where(np.array(hardness) > 0.5)[0]))
    print("hardness calculations: ", (len(np.where(np.array(hardness) > config.hardness)[0]) / len(hardness)), config.mastery)
    if top1 is None:
        if (len(np.where(np.array(hardness) > config.hardness)[0]) / len(hardness)) < config.mastery:
            print("therefore not mastered")
            return 0
    else:
        print("grep working", top1)
        if top1 < config.mastery:
            return 0
    # if len(np.where(np.array(hardness) < config.mastery)) > len(hardness)-2:
        # a lot of images still being misclassified
        # return 0
    print("therefore mastered")
    return 1


if __name__ == "__main__":
    main()
