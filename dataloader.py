import h5py
import numpy as np
from pathlib import Path
import torch
from torch.utils import data
from datetime import datetime
import random
import openslide as opsl
import glob

class HDF5Dataset(data.Dataset):
    def __init__(self, file_path):
        super(HDF5Dataset,self).__init__()
        self.data_info = []

        # Search for all folders
        slides = sorted(glob.glob(file_path+'/TB*'))
        
        for s in slides:
            bags=glob.glob(s+'/*.h5')
            for b in bags:
                f=h5py.File(b, 'r')
                label=f[('label')]
                slide=s.split('/')[-1].split('.')[0]
                bid=b.split('/')[-1].split('.')[0]
                bag=f[('bag')]
                self.data_info.append({'file_path': file_path, 'bag': bag, 'label': label, 'id': slide+'/'+bid})
        
    def __getitem__(self, index):
        i=index
        x=self.data_info[i]['bag']
        x = np.float32(x)
        if len(x.shape)==3:
            x=x[np.newaxis,:,:,:]
        x = torch.from_numpy(x/127.5 - 1.)
        y = self.data_info[i]['label']
        y = torch.from_numpy(np.array(y))
        n = self.data_info[i]['id']
        return (x, y, n)

    def __len__(self):
        return len(self.data_info)