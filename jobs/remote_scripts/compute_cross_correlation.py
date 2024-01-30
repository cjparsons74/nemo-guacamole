import os
import time
import h5py
import numpy as np
from glob import glob
from scipy import signal
from scipy.io import loadmat



def get_tag_paths(zebfolderpath):
    tag  = zebfolderpath.split('/')[8]

    files = os.listdir(zebfolderpath)
    files.sort()
    files.remove('README_LICENSE.rtf')

    path0, path1 = [zebfolderpath + '/' + fname for fname in files]

    return tag, path0, path1


def compute_cross_corr(sig0, sig1, **kwargs):
    xcorr = signal.correlate(sig0, sig1, **kwargs)
    lags = signal.correlation_lags(len(sig0), len(sig1), **kwargs)

    return xcorr, lags


def read_data(path0, path1, num_rois, scells=True):
    dholder = h5py.File(path0, "r")
    d0 = dholder["CellResp"][:]  # responses
    d1 = loadmat(path1, simplify_cells=scells)

    eliminated_rois = d1["data"]["IX_inval_anat"]
    all_rois = d1["data"]["CellXYZ"]

    used_rois_coor = np.array(
        [row for j, row in enumerate(all_rois) if j not in list(eliminated_rois)]
    )

    x, y, z = used_rois_coor[:num_rois, :].T

    return x, y, z, d0



def main():
    # set vars and path
    num_rois = 100
    idx0 = 1408
    PPATH = '/camp/home/duuta/working/duuta/ppp0/data/zeb*'

    # get zeb folder path(s) and path for specfic files
    zpath = glob(PPATH)[0]
    _, path0, path1 = get_tag_paths(zpath)
    _, _, _, d0 = read_data(path0, path1, num_rois, scells=True)

    # set neuron of interest 
    sig0 = d0[:, idx0]

    # dict to store sigidx associate lags and xcorrs
    lags_dict = {}
    xcorr_dict = {}

    # get any for asdf
    for idx1 in range(num_rois):

        sig1 = d0[:, idx1]
        xcorr, lags = compute_cross_corr(sig0, sig1)

        lags_dict[idx1] = lags
        xcorr_dict[idx1] = xcorr

    np.save('lags_dict0.npy', lags_dict)
    np.save('xcorr_dict0.npy', xcorr_dict)

 

if __name__ == "__main__":
    st = time.time()
    main()
    print(f'{time.time() - st} secs ----------- all done')
