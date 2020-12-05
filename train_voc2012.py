from tqdm import tqdm
import time
import os

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from ignite.metrics import IoU, Precision, Recall
import numpy as np


from PrecipitationForecastRadar.dataset.voc2012 import VOCSegmentation
from PrecipitationForecastRadar.models.SmaAt_UNet import SmaAt_UNet
from PrecipitationForecastRadar.helper import prepare_pt_context, get_lr
from PrecipitationForecastRadar.metric import iou

def train_epoch(train_dl, model, criterion, optimizer, device, show_progress=True):
    """ Training network in single epoch

    Args:
        train_dl (DataLoader): DataLoader of training set
        model (nn.Module): model in PyTorch
        criterion (loss): PyTorch loss
        optimizer (optimizer): PyTorch optimizer
        epoch (int): epoch number
        device (torch.device): torch.device
        writer (SummaryWriter): instance of SummaryWriter for TensorBoard
        show_progress (bool): if True, tqdm will be shown

    Returns:

    """
    model.train()
    train_loss = 0.0

    if show_progress:
        train_dl = tqdm(train_dl, "Train", unit="batch")
    for i, (images, targets) in enumerate(train_dl):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss /= len(train_dl)

    return train_loss

def evaluate_epoch(eval_dl, model, criterion):
    """ evaluation in a epoch

    Args:
        eval_dl (DataLoader): DataLoader of validation set
        model (nn.Module): model in PyTorch
        criterion (loss): PyTorch loss
        epoch (int): epoch number
        writer (SummaryWriter): instance of SummaryWriter for TensorBoard

    Returns:

    """
    device = next(model.parameters()).device

    model.eval()
    recall = Recall(lambda x: (x[0], x[1]))
    precision = Precision(lambda x: (x[0], x[1]))
    mean_recall = []
    mean_precision = []
    mean_loss = []

    val_loss = 0.0
    iou_metric = iou.IoU(21, normalized=False)

    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(eval_dl):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            # preds = outputs.argmax(1)
            pred_class = torch.argmax(nn.functional.softmax(outputs, dim=1), dim=1)
            iou_metric.add(pred_class, target=targets)

            precision.update((outputs, targets))
            recall.update((outputs, targets))
            mean_loss.append(loss.item())
            # mean_recall.append(recall.compute().item())
            # mean_precision.append(precision.compute().item())

        iou_class, mean_iou = iou_metric.value()
        val_loss /= len(eval_dl)

    # mean_precision, mean_recall = np.array(mean_precision).mean(), np.array(mean_recall).mean()
    # f1 = mean_precision * mean_recall * 2 / (mean_precision + mean_recall + 1e-20)

    return val_loss, iou_class, mean_iou


def train_classification(batch_size=10, num_gpus=1, learning_rate=0.001, epochs=5, resume=None, tensorboard=True,
                         early_stopping=None, log_dir="./logs", dir_ckp='./ckp'):
    # torch.backends.cudnn.benchmark = True
    root = 'data'
    use_cuda, batch_size, device = prepare_pt_context(
        num_gpus=num_gpus,
        batch_size=batch_size)

    # Prepare dataset
    # kwargs for selected dataset leading function
    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])
    dataset_train = VOCSegmentation(root, mode='train',
                                    transform=transformations,
                                    augmentations=True)
    dataset_val = VOCSegmentation(root, mode='val',
                                  transform=transformations,
                                  augmentations=False)
    train_dl = DataLoader(dataset_train,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0,
                          pin_memory=True)
    valid_dl = DataLoader(dataset_val,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=0,
                          pin_memory=True)

    # Load SmaAt-UNet
    model = SmaAt_UNet(n_channels=3, n_classes=21)
    # Move model to device
    model.to(device)
    if resume is not None:
        model.load_state_dict(torch.load(resume, map_location=device))

    # loss
    criterion = nn.CrossEntropyLoss().to(device)

    # optim and lr scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=4)

    # print(model)
    # print(summary(model, input_size=(3, size, size), device=device))
    # mean, std = calc_normalization(train_loader)
    # normalization = {}
    # normalization = {'mean': mean, 'std': std}

    # writer = SummaryWriter(log_dir="./logs")
    # # display some examples in tensorboard
    # images, labels = next(iter(train_loader))
    # originals = images * std.view(3, 1, 1) + mean.view(3, 1, 1)
    # writer.add_images('images/original', originals, 0)
    # writer.add_images('images/normalized', images, 0)
    # writer.add_graph(model, images.to(device))
    #

    # todo: from here, add train_epoch
    if tensorboard:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        writer = SummaryWriter(log_dir=log_dir)

    start_time = time.time()
    best_mIoU = -1.0
    early_stopping_counter = 0
    for epoch in tqdm(range(epochs), desc="Epochs", leave=True):
        train_loss = train_epoch(train_dl, model, criterion, optimizer, device)
        val_loss, iou_class, mean_iou = evaluate_epoch(valid_dl, model, criterion)
        lr_scheduler.step(mean_iou)

        dict_save = {
            'model': model,
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'train_loss': train_loss,
            'mIOU': mean_iou,
        }

        torch.save(dict_save, os.path.join(dir_ckp, f"checkpoints/model_{model.__class__.__name__}_epoch_{epoch}.pt"))

        if tensorboard:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Metric/mIOU', mean_iou, epoch)
            # writer.add_scalar('Metric/f1', f1, epoch)
            # writer.add_scalar('Metric/precision', mean_precision, epoch)
            # writer.add_scalar('Metric/recall', mean_recall, epoch)
            writer.add_scalar('Parameters/learning_rate', get_lr(optimizer), epoch)

        # Save the model with the best mean IoU
        if mean_iou > best_mIoU:
            os.makedirs("checkpoints", exist_ok=True)
            if not os.path.exists(dir_ckp):
                os.makedirs(dir_ckp)

            torch.save(dict_save,
                       os.path.join(dir_ckp, f"checkpoints/best_mIoU_model_{model.__class__.__name__}.pt"))
            best_mIoU = mean_iou
            earlystopping_counter = 0
        else:
            if early_stopping is not None:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping:
                    print(f"Stopping early --> mean IoU has not decreased over {early_stopping} epochs")
                    break

if __name__ == '__main__':
    train_classification()
