import torch
from tqdm import tqdm
from torchmetrics.functional import accuracy, precision_recall

def select_device(device, batch_size):
    device = str(device).strip().lower()
    gpus = [int(gpu) for gpu in device.split(',')]
    n = len(gpus)
    if n > 1:
        assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
    return gpus

def calculate_metrics(model, dataloader, cuda, device_id=0):
    model.eval()

    all_preds = torch.Tensor([])
    all_labels = torch.Tensor([])
    if cuda:
        all_labels = all_labels.cuda(device_id)
        all_preds = all_preds.cuda(device_id)
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            imgs = batch['image']
            labels = batch['label']

            if cuda:
                imgs = imgs.cuda(device_id)
                labels = labels.cuda(device_id)

            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)
            all_labels = torch.cat([all_labels, labels], dim=-1)
            all_preds = torch.cat([all_preds, preds], dim=-1)

    all_labels = all_labels.type(torch.int64)
    all_preds = all_preds.type(torch.int64)
    acc = accuracy(all_preds, all_labels)
    prec, rec = precision_recall(all_preds, all_labels)
    f1 = 2*prec*rec / (prec + rec)
    return prec, rec, f1, acc
