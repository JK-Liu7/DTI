import math
import random
import timeit

import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.utils.data import DataLoader

from utils import *

from metrics import *
from dataset import *

from models.net import DTDTInet


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]

def train(model, dataloader, criterion):
    model.eval()
    running_loss = AverageMeter()

    pred_list = []
    pred_cls_list = []
    label_list = []

    for data in dataloader:

        label = data.y
        label = label.to(device)
        data = data.to(device)

        with torch.no_grad():

            pred = model(data)
            loss = criterion(pred, label)
            pred_cls = torch.argmax(pred, dim=-1)

            pred_prob = F.softmax(pred, dim=-1)
            pred_prob, indices = torch.max(pred_prob, dim=-1)
            pred_prob[indices == 0] = 1. - pred_prob[indices == 0]

            pred_list.append(pred_prob.view(-1).detach().cpu().numpy())
            pred_cls_list.append(pred_cls.view(-1).detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            running_loss.update(loss.item(), label.size(0))

    pred = np.concatenate(pred_list, axis=0)
    pred_cls = np.concatenate(pred_cls_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    acc = accuracy(label, pred_cls)
    pre = precision(label, pred_cls)
    rec = recall(label, pred_cls)
    auc = auc_score(label, pred)

    epoch_loss = running_loss.get_average()
    running_loss.reset()

    model.train()

    return epoch_loss, acc, pre, rec, auc


if __name__ == '__main__':

    model_dir = ('save/')

    DATASET = "human"
    dir_input = ('dataset/' + DATASET + '/processed/')

    os.makedirs(dir_input, exist_ok=True)

    # """Create a dataset and split it into train/dev/test."""

    dropout = 0.2
    batch = 8
    lr = 1e-4
    weight_decay = 1e-4
    decay_interval = 5
    lr_decay = 0.5

    dataset_train = TestbedDataset(root='dataset/' + DATASET + '/processed/', dataset=DATASET+'_train')
    dataset_val = TestbedDataset(root='dataset/' + DATASET + '/processed/', dataset=DATASET + '_val')
    dataset_test = TestbedDataset(root='dataset/' + DATASET + '/processed/', dataset=DATASET+'_test')

    train_loader = DataLoader(dataset_train, batch_size=batch, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=batch, shuffle=False)

    model = DTDTInet(max_length=1000, compound_feature=41, compound_graph_dim=64, compound_smiles_dim=4, protein_dim=64, out_dim=2, dropout=0.2)
    model.to(device)

    """Output files."""

    start = timeit.default_timer()
    max_AUC_test = 0

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    running_loss = AverageMeter()

    model.train()

    epochs = 1000
    steps_per_epoch = 10
    num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))
    print(num_iter)

    AUCs = ('Epoch\t\tTime(sec)\t\tLoss_train\t\tAUC_test\t\tPrecision_test\t\tRecall_test')
    # with open(file_AUCs, 'w') as f:
    #     f.write(AUCs + '\n')

    """Start training."""
    print('Training on ' + DATASET)
    print(AUCs)

    global_step = 0
    global_epoch = 0

    running_loss = AverageMeter()

    model.train()

    for i in range(num_iter):
        for data in train_loader:

            global_step += 1

            label = data.y
            label = label.to(device)
            data = data.to(device)
            pred = model(data)

            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.update(loss.item(), label.size(0))

            if global_step % steps_per_epoch == 0:
                global_epoch += 1

                epoch_loss = running_loss.get_average()
                running_loss.reset()

                val_loss, val_acc, val_pre, val_rec, val_auc = train(model, val_loader, criterion)
                test_loss, test_acc, test_pre, test_rec, test_auc = train(model, test_loader, criterion)


                end = timeit.default_timer()
                time = end - start

                AUCs = [global_epoch, time, loss.item(), test_auc, test_pre, test_rec]
                # AUCs = [epoch, time, loss_train, AUC_test]

                if test_auc > max_AUC_test:
                    # tester.save_model(model, file_model)
                    max_AUC_test = test_auc
                    epoch_label = global_epoch
                print('\t'.join(map(str, AUCs)))

    print("The best model is epoch", epoch_label)

