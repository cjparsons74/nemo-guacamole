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
data_files = list(glob(f"{nroot}natimg2800_M*.mat"))
data_names = [fname.split('/')[-1].strip('.mat').strip("natimg2800_") for fname in data_files]
picklepaths = list(glob(f"{data_root}data_*.pickle"))
print('here are all pickle paths', picklepaths)



title_dict = dict({'spont': 'Spontaneous', 'resp': 'Response'})
tag = 'resp'
#dasheader = f'Dashboard for {title_dict[tag]} Activity'

print("frantically reading all files ...")
# read files all 
N = len(picklepaths)


ivals = [] # arr for alpha vals
ss_arr = [] # arr for  
ssp_arr = [] # arr for partial set of neurons
fracs = [1.000, 0.500, 0.250, 0.125, 0.062, 0.031, 0.016]
dims = []

tot_ss = 0

print(f"there are {N} files")
for i, fpath in enumerate(picklepaths):
    print(f"reading data {i}................")
    set_filename = f"Dashboard of {title_dict[tag]} Activity: " + data_names[i]

    ofile = open(fpath, 'rb')
    data = pickle.load(ofile)
    ofile.close()


    print(f'working hard on {fpath}')
    resp, spon, istim = h.unbox(data)
    resp = h.denoise_resp(resp, spon)
    
    resp_ = h.dupSignal(resp, istim)


    print("preping data for structures...")
    #resp = np.array(random.sample(list(resp), max_rows))

    tcluster = time.time()

    print("computing...cross validated PCA...")
    ss_resp = u.shuff_cvPCA(resp_)
    ss_resp_ = ss_resp.mean(axis=0)
    
    ss_  = ss_resp_/ss_resp_.sum()
    tot_ss += ss_

    ss_arr.append(ss_)
    dims.append(ss_.shape[0])

    print("computing power laws...")
    #alpha_fit, ypred = u.get_powerlaw(ss_, np.arange(10, 5e2).astype('int'))	
    #alf = round(alpha_fit, 2)
    
    #ivals.append(alf)


bins = np.arange(0.75, 1.1, 0.05)

ntotss = tot_ss.shape[0]

# 2C
av_fit, a_ypred = u.get_powerlaw(tot_ss, np.arange(10, 5e2).astype('int'))
fav = round(av_fit, 2)
print("av_fit", fav)
plt.loglog(np.arange(0, ntotss)+1, tot_ss, label='observed')
plt.loglog(np.arange(0, ntotss)+1, a_ypred, label=f'fav=')
plt.xlabel("Dimension")
plt.ylabel("Variance")
plt.xlim([10**0, 10**3+1e-6])
plt.title("Plot 2C")
plt.legend()
plt.savefig("plot2C.png")


mindim = np.min(dims)
print("mindim", mindim)    
print(f'here are the dims{dims}')
print("Here are the vals of alpha", ivals)
print("making plot 2E")
plt.hist(ivals, bins=bins)
plt.xlim([0.85, 1.15])
plt.xlabel("values of alpha")
plt.ylabel("No of recordings")
plt.title("Plot 2E")
plt.savefig('plot2E.png')
plt.close()


print("making plot 2D")
plt.loglog(np.transpose(ss_arr), label='all recordings')
plt.xlim([10**0, 10**4])
plt.ylabel("Variance")
plt.xlabel("PC Dimension")
plt.title("Plot 2D")
plt.savefig("plot2D.png")
plt.close()

nn = ss_resp_.shape[0]
print("making plot 2G")
for frac in fracs:
    rsxy_ = ss_[:math.ceil(frac*nn)]
    nrsxy = len(rsxy_)
    plt.loglog(np.arange(0, nrsxy) + 1, rsxy_, label=f'{frac=}')

plt.ylabel("Variance")
plt.xlabel("Dimension")
plt.title("Plot 2G")
plt.legend()
plt.savefig("plot2G.png")
