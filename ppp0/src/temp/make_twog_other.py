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


maxiter = 2
fracycle = 3
print("franctically getting all file paths...")
data_root = "/Users/duuta/ppp/data/stringer/dpickle/"
nroot = "/Users/duuta/ppp/data/stringer/live_data/"
oroot = "/Users/duuta/ppp/src/temp/PLOTS/test/"
data_files = list(glob(f"{nroot}natimg2800_M*.mat"))
data_names = [fname.split('/')[-1].strip('.mat').strip("natimg2800_") for fname in data_files]
picklepaths = list(glob(f"{data_root}data_*.pickle"))[:maxiter]
print('here are all pickle paths', picklepaths)
N = len(picklepaths)


# nos of samples averaged over 
fracs = [1.000, 0.500, 0.250, 0.125, 0.062, 0.031, 0.016][:fracycle]
maxneurons = 9000 # set for uniformity across filesl; not all files have 2800 images
nl = len(fracs)
arr_list = []
maxrows = 2366 


# init data storage array 
resarr = np.zeros((nl, maxrows, fracycle))


while picklepaths: 
    i = 0
    print(f"reading data.........")
    fpath=picklepaths.pop(-1)
    for j, frac in enumerate(fracs): 

        ofile = open(fpath, 'rb')
        data = pickle.load(ofile)
        ofile.close()


        print(f'reading files and unpacking struct...')
        resp, spon, istim = h.unbox(data)
        resp_ = h.denoise_resp(resp, spon)
        resp_ = h.dupSignal(resp_, istim)
        

        print('setting maxneurons for array....')
        resp_ = resp_[:, :, :maxneurons]
        print("shape of arr[:, :maxneurons]", resp_.shape)
        

        print("getting portion of neurons...")
        portion = math.ceil(frac*maxneurons)
        sub_resp = resp_[:, :, :portion]
        print('shape of sub resp', sub_resp.shape)


        print("computing...cross validated PCA...")
        ss_resp_ = u.shuff_cvPCA(resp_, nshuff=10)
        ss_resp_ = ss_resp_.mean(axis=0)
        ss_ = ss_resp_/ss_resp_.sum()
        ss_ = ss_[:maxrows]
            
        print(f'shape of resulting arr {ss_.shape}')


        # store results for given frac and dataset 
        resarr[j,:,i] += ss_ # could add results here...
        arr_list.append(resarr)
        i +=1
        print('results arr', resarr)



print("making plot 2G")
p=0
print('mean vector...', np.mean(resarr[p,:,:], axis=1).shape)
print('mean vector...', np.mean(resarr, axis=1).shape)
print('sample ..', resarr[:, :,:].shape)
for i in range(nl):
    plt.loglog(resarr[i, :, i], label=f'frac={fracs[0]}')
plt.ylabel("Variance")
plt.xlabel("Dimension")
plt.title("Plot 2G")
plt.legend()
plt.savefig(f"{oroot}plot_twog.png")
plt.close()
