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
        config.dataset, config.data_path, cutout_length=0, validation=True)

    net_crit = nn.CrossEntropyLoss().to(device)
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
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, config.epochs, eta_min=config.w_lr_min)
    architect = Architect(model, config.w_momentum, config.w_weight_decay)

    # training loop
    best_top1 = 0.
    hardness = None
    old_loss = 0
    for epoch in range(config.epochs):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]

        model.print_alphas(logger)

        epoch_type = 1
        if config.dynamic:
            epoch_type = get_epoch_type(epoch, hardness)

        if epoch_type: # 1 is train, as normal (0 is dataset update)
            # training
            hardness = train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch)

            # validation
            cur_step = (epoch+1) * len(train_loader)
            top1, new_loss = validate(valid_loader, model, epoch, cur_step)

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
            if abs(old_loss - new_loss) < 0.0001:
                print("stopping early")
                # break
            old_loss = new_loss
        else:
            try:
                train_loader.dataset.update_subset(hardness, epoch)
            except IndexError:
                raise AttributeError(len(hardness), len(train_loader.dataset))

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("Best Genotype = {}".format(best_genotype))
    logger.info("Training end {}".format(time.time()-start_time))


def train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch*len(train_loader)
    writer.add_scalar('train/lr', lr, cur_step)

    model.train()

    hardness = [None for i in range(len(train_loader))]
    batch_size = config.batch_size
    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):
        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
        val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(device, non_blocking=True)
        N = trn_X.size(0)

        # phase 2. architect step (alpha)
        alpha_optim.zero_grad()
        architect.unrolled_backward(trn_X, trn_y, val_X, val_y, lr, w_optim)
        alpha_optim.step()

        # phase 1. child network step (w)
        w_optim.zero_grad()
        logits = model(trn_X)
        loss = model.criterion(logits, trn_y)
        new_hardness = get_hardness(logits.cpu(), trn_y.cpu(), loss)
        loss.backward()
        hardness[(step*batch_size):(step*batch_size)+batch_size] = new_hardness # assumes batch 1 takes idx 0-8, batch 2 takes 9-16, etc.

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

    return hardness

def validate(valid_loader, model, epoch, cur_step):
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

            if step % config.print_freq == 0 or step == len(valid_loader)-1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.4f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, config.epochs, step, len(valid_loader)-1, losses=losses,
                        top1=top1, top5=top5))

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


def get_hardness(output, target, loss):
    # currently a binary association between correct classication => 0.8
    # we want it to be a softmax representation. if we instead take crossentropy loss of each individual cf target
    _, predicted = torch.max(output.data, 1)
    confidence = F.softmax(output, dim=1)
    hardness_scaler = np.where((predicted == target), 1, 3) # if correct, simply use confidence as measure of hardness
    # therefore if model can easily say yep this is object X, then hardness will be low. if it only just manages to identify
    # object X, hardness if higher
    # if object X is misclassified, hardness needs to be higher still.
    # assumes that it does not confidently misclassify.
    hardness = [confidence[i][predicted[i]] * hardness_scaler[i] for i in range(output.size(0))]
    raise AttributeError(hardness, confidence, predicted, hardness_scaler)
    return hardness


def get_epoch_type(epoch, hardness):
    # naive alternate, starting with normal training
    if not config.dynamic or epoch < config.init_train_epochs:
        return 1
    is_mastered = get_mastered(hardness)
    if is_mastered:
        return 1
    return 0


def get_mastered(hardness):
    if np.sum(hardness==0.8) > len(hardness)-2:
        # a lot of images still being misclassified
        return 0
    return 1


if __name__ == "__main__":
    main()
