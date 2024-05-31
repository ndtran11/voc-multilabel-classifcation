import torch
import numpy as np
from tqdm import tqdm

from model import MultiLabelClassification
from dataset import Dataset

import pandas as pd
import clip
import traceback
import random
import os
import sys
import shutil

def train_multilabel_classification(
    model, 
    train_dataset, 
    val_dataset, 
    loss_fn, 
    optimizer, 
    lr_scheduler=None,
    n_epoch=10, 
    train_batch_size=64,
    val_batch_size=None,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    wd=os.path.join(os.getcwd(), '.tmp')
) -> MultiLabelClassification:
    os.makedirs(wd, exist_ok=True)
    best_model = os.path.join(wd, 'best_model.pth')

    val_batch_size = val_batch_size or train_batch_size
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=train_batch_size, 
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=val_batch_size, 
        shuffle=False
    )

    best_val_loss = np.inf
    for epoch in range(n_epoch):
        model.train()
        train_loss, val_loss, train_acc, val_acc = 0, 0, 0, 0

        for i, (images, labels) in enumerate(tqdm(train_loader, total=len(train_loader), desc='Training')):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images) 

            logits = torch.nn.functional.sigmoid(logits)
            loss = loss_fn(logits, labels)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            train_acc += ((logits > 0.5).int() == (labels > 0.5).int()).sum(dim=1).float().mean().item()

        if lr_scheduler is not None:
            lr_scheduler.step()

        model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(val_loader, total=len(val_loader), desc='Validation')):
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                logits = torch.nn.functional.sigmoid(logits)

                loss = loss_fn(logits, labels)
                val_loss += loss.item()
                val_acc += ((logits > 0.5).int() == (labels > 0.5).int()).sum(dim=1).float().mean().item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_acc /= len(train_loader)
        val_acc /= len(val_loader)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Best model found at epoch {epoch} with val loss: {val_loss}")
            torch.save(model.state_dict(), best_model)

    for file in os.listdir(wd):
        if os.path.isfile(os.path.join(wd, file)):
            os.remove(os.path.join(wd, file))
        else:
            shutil.rmtree(os.path.join(wd, file))

    return model

def sanity_check(
    model, 
    train_ds, 
    val_ds, 
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)

    model.eval()

    try:
        for i, (images, labels) in enumerate(train_loader):
            model(images.to(device))
            if i == 10:
                break

        for i, (images, labels) in enumerate(val_loader):
            model(images.to(device))
            if i == 10:
                break

    except Exception as err:
        traceback.print_exc()
        return False

    return True

def parse_options():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_epoch', help='Number of epochs to train the model', type=int, default=10)
    parser.add_argument('--batch_size', help='Training batch size', type=int, default=64)
    parser.add_argument('--lr', help='Initial learning rate', type=float, default=1e-4)
    parser.add_argument('--lr_decay', help='Learning rate decay factor', type=float, default=0.7)
    parser.add_argument('--lr_sched_step', help='Number of epochs before learning rate decay', type=int, default=3)

    parser.add_argument('--train_ratio', help='Ratio of training data', type=float, default=0.7)
    parser.add_argument('--dataset_root', help='Root directory of the dataset', type=str, default='data')
    parser.add_argument('--output_model', help='Output model file', type=str, default='best_model.pth')

    parser.add_argument('--clip_model', help='CLIP model to use', type=str, default='ViT-B/32')
    parser.add_argument('--save_only_classifier', help='Just save the classifier', type=bool, default=True)

    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_options()

    device = torch.device('cpu') # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, clip_preprocess = clip.load(opt.clip_model, device=device)

    dataset_root = opt.dataset_root
    assert os.path.isdir(dataset_root), f"Dataset root directory {dataset_root} not found"

    labels_file = os.path.join(dataset_root, 'labels.json')
    assert os.path.isfile(labels_file), f"Labels file {labels_file} not found"


    labels = pd.read_json(labels_file)

    train_ratio = opt.train_ratio
    train_size = int(train_ratio * len(labels))

    train_labels = labels[:train_size]
    val_labels = labels[train_size:]

    num_classes = 20

    train_dataset = Dataset(
        os.path.join(dataset_root, 'images'),
        train_labels['filename'].to_list(),
        train_labels['label'].to_list(),
        num_classes,
        clip_preprocess  
    )

    val_dataset = Dataset(
        os.path.join(dataset_root, 'images'),
        val_labels['filename'].to_list(),
        val_labels['label'].to_list(),
        num_classes,
        clip_preprocess  
    )

    model = MultiLabelClassification(clip_model.visual, num_classes).to(device)
    loss_fn = torch.nn.functional.binary_cross_entropy

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_sched_step, opt.lr_decay)

    if not sanity_check(model, train_dataset, val_dataset, device=device):
        print("Model is not working properly")
        sys.exit(1)

    res = train_multilabel_classification(
        model, 
        train_dataset, 
        val_dataset, 
        loss_fn, 
        optimizer, 
        lr_scheduler,
        n_epoch=opt.n_epoch,
        train_batch_size=opt.batch_size,
        device=device
    )

    torch.save(
        res.state_dict() if not opt.save_only_classifier else res.classifier.state_dict(), 
        opt.output_model
    )