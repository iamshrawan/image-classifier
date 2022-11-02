import argparse
from cProfile import label
from statistics import mode
from torch.utils.data import DataLoader
import torch
import os
import wandb
import time
import datetime
import sys

from train_utils import select_device, calculate_metrics
from model import Classifier
from dataset import ClassifierDataset


parser = argparse.ArgumentParser()
parser.add_argument('--split_folder', '-s', type=str, default='dataset_splits', help='Folder containing json files of dataset splits')
parser.add_argument('--exp', type=str, default='test', help='Experiment name for logging')
parser.add_argument('--n_classes', type=int, default=10, help="Number of categories")
parser.add_argument('--n_epochs', type=int, default=100, help="Number of epochs to train")
parser.add_argument('--epoch', type=int, default=0, help="Starting Epoch")
parser.add_argument('--batch_sz', '-b', type=int, default=16, help="Batch size")
parser.add_argument('--lr', type=float, default=1e-4, help="Learning Rate")
parser.add_argument('--pretrained', action='store_true', help='use pretrained weights')
parser.add_argument("--device" , default='', help="Gpu id for training: 0 or 0,1 or cpu")
parser.add_argument("--checkpoint_interval", default=50, type=int, help="Epoch interval to save model checkpoints")
opt = parser.parse_args()
print(opt)

os.makedirs('./checkpoints', exist_ok=True)
os.makedirs('./checkpoints/%s' % (opt.exp), exist_ok=True)

opt.cuda = torch.cuda.is_available()
if opt.cuda:
    device = select_device(opt.device, opt.batch_size)
    device_id = device[0]
else:
    device_id = 0

torch.manual_seed(2022)

train_dataset = ClassifierDataset(os.path.join(opt.split_folder, 'train.json'))
dev_dataset = ClassifierDataset(os.path.join(opt.split_folder, 'dev.json'))
train_loader = DataLoader(train_dataset, batch_size=opt.batch_sz, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=opt.batch_sz, shuffle=False)

model = Classifier(pre_trained=opt.pretrained, num_classes=opt.n_classes)


if opt.epoch > 0:
    # Load pretrained models
    print(f"Loading model from epoch {opt.epoch} ")
    checkpoint = torch.load("./checkpoints/%s/model_checkpoint_%d.pth" % (opt.exp, opt.epoch), map_location=torch.device(f"cuda:{device_id}"))
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['best_acc']

elif opt.epoch == -1:
    checkpoint = torch.load("./checkpoitns/%s/best_generator.pth" % (opt.exp), map_location=torch.device(f"cuda:{device_id}"))
    print(f"Continuing from best performing model checkpoint i.e epoch:{checkpoint['epoch']}")
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['best_acc']
    opt.epoch = checkpoint['epoch']
    

else:
    # Initialize weights
    best_acc = 0.0

ce_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=opt.lr)
if opt.cuda:
    model = model.cuda(device=device_id)
    ce_loss = ce_loss.cuda(device=device_id)


prev_time = time.time()

runs = wandb.init(project="Object-Classification", entity = "shrawan", name=f"{opt.exp}", reinit=True)
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(train_loader):
        img = batch['image']
        labels = batch['label']

        if opt.cuda:
            img = img.cuda(device_id)
            labels = img.cuda(device_id)

        model.train()
        optimizer.zero_grad()
        out = model(img)
        loss = ce_loss(out, labels)
        loss.backward()
        optimizer.step()
        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(train_loader) + i
        batches_left = opt.n_epochs * len(train_loader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        
        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [CE Loss: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(train_loader),
                loss.item(),
                time_left,
            )
        )

        wandb.log({
                'CE_loss': loss.item()
            })

        break
    
    #Calculating Metrics
    prec, rec, f1, acc = calculate_metrics(model, dev_loader, opt.cuda, device_id)

    wandb.log({
        'Precision': prec,
        'recall': rec,
        'f1_score': f1,
        'accuracy': acc
    })

    if acc > best_acc:
        best_acc = acc
        print("Best Accuracy: ", best_acc)
        filepath = './checkpoints/%s/best_model.pth' % (opt.exp)
        torch.save({
            'epoch': epoch,
            'best_acc': best_acc,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, filepath)

    if (epoch+1) % opt.checkpoint_interval == 0:
        torch.save({
            'best_acc': best_acc,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, "./checkpoints/%s/model_checkpoint_%d.pth" % (opt.exp, epoch))





