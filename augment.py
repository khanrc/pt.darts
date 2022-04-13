""" Training augmented model """
import os
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from config import AugmentConfig
import utils
from models.augment_cnn import AugmentCNN
from curriculum import Curriculum_loader
from torchvision import transforms

config = AugmentConfig()

device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)


def main():
    logger.info("Logger is set - training start")

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    # get data with meta info
    input_size, input_channels, n_classes, train_data, valid_data = utils.get_data(
        config.dataset, config.data_path, config.cutout_length, validation=True,
        search=False, bede=config.bede, is_concat=config.is_concat)

    criterion = nn.CrossEntropyLoss().to(device)
    if config.dataset == "imageobj":
        criterion = nn.BCEWithLogitsLoss().to(device)
    use_aux = config.aux_weight > 0.
    model = AugmentCNN(input_size, input_channels, config.init_channels, n_classes, config.layers,
                       use_aux, config.genotype)
    model = nn.DataParallel(model, device_ids=config.gpus).to(device)

    # model size
    mb_params = utils.param_size(model)
    logger.info("Model size = {:.3f} MB".format(mb_params))

    # weights optimizer
    optimizer = torch.optim.SGD(model.parameters(), config.lr, momentum=config.momentum,
                                weight_decay=config.weight_decay)

    if config.use_curriculum:
        train_loader = Curriculum_loader(config.dataset, val=False)
    else:
        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=config.batch_size,
                                                   shuffle=True,
                                                   num_workers=config.workers,
                                                   pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=config.batch_size,
                                               shuffle=False,
                                               num_workers=config.workers,
                                               pin_memory=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)

    best_top1 = 0.
    is_multi = False
    if config.dataset == "imageobj" or config.dataset == "cocomask":
        is_multi = True

    if config.use_curriculum:
        update_epochs = train_loader.update_epochs

    # training loop
    for epoch in range(config.epochs):
        if config.use_curriculum:
            if epoch in update_epochs:
                if config.final_mined:
                    train_loader.generate_cur_set(max(update_epochs))
                else:
                    train_loader.generate_cur_set(epoch)

        lr_scheduler.step()
        drop_prob = config.drop_path_prob * epoch / config.epochs
        model.module.drop_path_prob(drop_prob)

        # training
        train(train_loader, model, optimizer, criterion, epoch, is_multi)

        # validation
        cur_step = (epoch+1) * len(train_loader)
        top1 = validate(valid_loader, model, criterion, epoch, cur_step, is_multi)

        # save
        if best_top1 < top1:
            best_top1 = top1
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model, config.path, is_best)

        print("")

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))


def train(train_loader, model, optimizer, criterion, epoch, is_multi):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch*len(train_loader)
    cur_lr = optimizer.param_groups[0]['lr']
    logger.info("Epoch {} LR {}".format(epoch, cur_lr))
    writer.add_scalar('train/lr', cur_lr, cur_step)

    model.train()

    for step, (X, y) in enumerate(train_loader):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        N = X.size(0)

        optimizer.zero_grad()
        if step == 0:
            # summ = summary(model, input_size=list(X.size()))
            # print("grep summ", summ)
            total_params = 0
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad: continue
                param = parameter.numel()
                total_params+=param
            print("grep params", total_params)
        logits, aux_logits = model(X)
        if is_multi:
            loss = criterion(logits, y.float())
        else:
            loss = criterion(logits, y)
        if config.aux_weight > 0.:
            loss += config.aux_weight * criterion(aux_logits, y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        if is_multi:
            prec1, prec5 = utils.accuracy_multilabel(logits, y) # top5 doesnt apply
        else:
            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1, top5=top5))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))


def validate(valid_loader, model, criterion, epoch, cur_step, is_multi):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits, _ = model(X)
            if is_multi:
                loss = criterion(logits, y.float())
            else:
                loss = criterion(logits, y)
            for q, im in enumerate(X):
                toSave = transforms.ToPILImage()(im.cpu())
                savePath = "./tempSave/gt/{}.png".format((step * N) + q)
                toSave.save(savePath)
                with open(f"./tempSave/gt/{(step*N)+q}.txt", "w") as txtfile:
                    sigmoid = torch.sigmoid(logits)
                    sigmoid[logits>0.5] = 1
                    sigmoid[logits<=0.5] = 0
                    for sigm in sigmoid[q]:
                        txtfile.write(str(sigm.item()) + ", ")

            if is_multi:
                prec1, prec5 = utils.accuracy_multilabel(logits, y)  # top5 doesnt apply
            else:
                prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % config.print_freq == 0 or step == len(valid_loader)-1:
                logger.info(
                    "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, config.epochs, step, len(valid_loader)-1, losses=losses,
                        top1=top1, top5=top5))

    writer.add_scalar('val/loss', losses.avg, cur_step)
    writer.add_scalar('val/top1', top1.avg, cur_step)
    writer.add_scalar('val/top5', top5.avg, cur_step)

    logger.info("Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))

    return top1.avg


if __name__ == "__main__":
    main()
