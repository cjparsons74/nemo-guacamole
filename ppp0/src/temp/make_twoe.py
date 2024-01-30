import os
import sys
import pdb
import scipy
import math
import time
import pickle
import random
import utils as u
import numpy as np
import helpers as h
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.gridspec as Gridspec


print("franctically getting all file paths...")
data_root = "/Users/duuta/ppp/data/stringer/dpickle/"
nroot = "/Users/duuta/ppp/data/stringer/live_data/"
saveROOT="/Users/duuta/ppp/plots/makeplots/"
data_files = list(glob(f"{nroot}natimg2800_M*.mat"))
data_names = [fname.split('/')[-1].strip('.mat').strip("natimg2800_") for fname in data_files]
picklepaths = list(glob(f"{data_root}data_*.pickle"))
print('here are all pickle paths', picklepaths)


print("frantically reading all files ...")
# read files all 
N = len(picklepaths)

ivals = [] # arr for alpha vals


print(f"there are {N} files")
for i, fpath in enumerate(picklepaths):
    print(f"reading data {i}................")

    ofile = open(fpath, 'rb')
    data = pickle.load(ofile)
    ofile.close()


    print(f'working hard on {fpath}')
    resp, spon, istim = h.unbox(data)
    resp = h.denoise_resp(resp, spon)
    
    resp_ = h.dupSignal(resp, istim)

    print("computing...cross validated PCA...")
    ss0 = u.shuff_cvPCA(resp_)
    ss0 = ss0.mean(axis=0)
    
    ss  = ss0/ss0.sum()


    print("computing power laws...")
    a0, ypred = u.get_powerlaw(ss, np.arange(10, 5e2).astype('int'))	
    alf = round(a0, 2)
    
    ivals.append(alf)


bins = np.arange(0.75, 1.1, 0.05)


print("Here are the vals of alpha", ivals)
print("making plot 2E")
plt.hist(ivals, bins=bins, color='gray', edgecolor='black', linewidth=1.8)
plt.xlim([0.85, 1.15])
plt.ylim([0, 4.20])
plt.xticks([0.9, 1.0, 1.1])
plt.yticks([0, 2, 4])
plt.xlabel("values of alpha", fontsize='x-large')
plt.ylabel("No of recordings", fontsize='x-large')
plt.tick_params(top='off', right='off')
plt.title("Plot 2E")
plt.savefig(f'{saveROOT}plot_twoe.png')
plt.close()
