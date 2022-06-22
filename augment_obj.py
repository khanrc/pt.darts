""" Training augmented model """
import os
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from config import AugmentConfig
import utils
from models.augment_cnn_obj import AugmentCNN
from curriculum import Curriculum_loader
from torchvision import transforms
from detectionengine import evaluate

config = AugmentConfig()

device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)

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
    logger.info("Logger is set - training start")

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    # get data with meta info
    input_size, input_channels, n_classes, train_data, val_data = utils.get_data(
        config.dataset, config.data_path, config.cutout_length, validation=True,
        search=False, bede=config.bede, is_concat=config.is_concat)

    criterion = nn.CrossEntropyLoss().to(device)
    if config.dataset == "imageobj":
        criterion = nn.BCEWithLogitsLoss().to(device)
    # use_aux = config.aux_weight > 0.
    use_aux = False
    model = AugmentCNN(input_size, input_channels, config.init_channels, n_classes, config.layers,
                       use_aux, config.genotype).to(device)
    # model = nn.DataParallel(model, device_ids=config.gpus).to(device)

    # model size
    mb_params = utils.param_size(model)
    logger.info("Model size = {:.3f} MB".format(mb_params))

    # weights optimizer
    optimizer = torch.optim.SGD(model.parameters(), config.lr, momentum=config.momentum,
                                weight_decay=config.weight_decay)

    collate_func = collate_fn
    # split = get_split(train_data)
    # indices = list(range(len(train_data)))
    # # random.seed(1337) # note must use same random seed as dataloader (and thus process same images)
    # # random.shuffle(indices)
    # train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    # valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    if config.use_curriculum:
        train_loader = Curriculum_loader(config.dataset, val=False)
    else:
        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=config.batch_size,
                                                   # sampler=train_sampler,
                                                   num_workers=config.workers,
                                                   pin_memory=True,
                                                   collate_fn=collate_func,
                                                   drop_last=True)
    valid_loader = torch.utils.data.DataLoader(val_data,
                                               batch_size=config.batch_size,
                                               # sampler=valid_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True,
                                               collate_fn=collate_func,
                                               drop_last=True)
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
        # model.module.drop_path_prob(drop_prob) # TODO

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
        # X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        X = X.to(device, non_blocking=True)
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
        logits = model(X, y)
        loss = sum(_loss for _loss in logits.values())
        losses.update(loss.item(), N)

        # if config.aux_weight > 0.:
        #     loss += config.aux_weight * criterion(aux_logits, y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()


        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1, top5=top5))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        # writer.add_scalar('train/top1', prec1.item(), cur_step)
        # writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1

    logger.info("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))


def validate(valid_loader, model, criterion, epoch, cur_step, is_multi):
    model.eval()
    evaluate(model, valid_loader, device=device, epoch=epoch, augment=True)
    print (f"epoch: {epoch}")
    return 0


if __name__ == "__main__":
    main()
