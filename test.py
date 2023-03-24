from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable

from dataloader import HDF5Dataset
from slideModel import Attention

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from tqdm import tqdm
import h5py
import os

# Inference settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--file_path', help='where to search for dataset .h5')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA')
parser.add_argument('--model', type=str, help='model location')
parser.add_argument('--wd', type=str, default='./', help='write results where')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

loader_kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

test_loader = data_utils.DataLoader(HDF5Dataset(file_path=args.file_path),
                                     batch_size=1,
                                     shuffle=False,
                                     **loader_kwargs)
print('# of bags: '+str(len(test_loader)))

if args.model is None:
    print('specify model location')
    exit(0);
    
model = torch.load(args.model)
    
if args.cuda:
    model.cuda()
model.eval()

with torch.no_grad():
    prs=[]
    gts=[]
    probs=[]
    for batch_idx, (data, label, n) in enumerate(tqdm(test_loader)):
        bag_label = label[0]
        data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        _, _, Y_hat, Y_prob, A = model.calculate_objective(data, bag_label)
        
        label=label[0].data.cpu().numpy()[0][0]
        Y_hat=Y_hat.data.cpu().numpy()[0][0]
        Y_prob=Y_prob.data.cpu().numpy()[0][0]
        gts.append(label)
        prs.append(Y_hat)
        probs.append(Y_prob)

        n=n[0]
        if not os.path.exists(n.split('/')[0]):
            os.mkdir(n.split('/')[0])
        f = h5py.File(args.wd+'/'+n+'.h5', 'w')
        dset = f.create_dataset("A", data=A.cpu().numpy())
        dset = f.create_dataset("Y_hat", data=Y_hat)
        dset = f.create_dataset("Y_prob", data=Y_prob)
        dset = f.create_dataset("Y", data=label)
        f.close()
        
    auc=roc_auc_score(gts,probs)
    cm=confusion_matrix(gts,prs)
    print('AUC: {:.4f} Pr: {:.4f} Re: {:.4f}'.format(auc, cm[1,1]/cm[:,1].sum(), cm[1,1]/cm[1,:].sum()))
