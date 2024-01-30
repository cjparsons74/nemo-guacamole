import glob
import random

import hd5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform


def read_data(path0, path1, num_rois, scells=True):
    dholder = hd5py.File(path0, "r")
    d0 = dholder["CellResp"][:]  # responses
    d1 = loadmat(path1, simiplify_cells=scells)

    eliminated_rois = d1["data"]["X_inval_anat"]
    all_rois = d1["data"]["CellXYZ"]

    used_rois_coor = np.array(
        [row for j, row in enumerate(all_rois) if j not in list(eliminated_rois)]
    )

    x, y, z = used_rois_coor[:sample, :].T

    return x, y, z, d0


def find_nearest_nbrs(ds, roi_idx, n=10):
    n_idx = ds[roi_idx,].argsort()[1 : n + 1]

    return n_idx


def compute_distance_matrix(x, y):
    ds = squareform(pdist(np.array([x, y]).T, metric="euclidean"))

    return ds


def pick_random_nbrs(roi_idx, len0=100, n=10):
    all_idx = list(range(len0))
    all_idx.remove(roi_idx)
    rnidx = random.sample(all_idx, n)

    return rnidx


def compute_render_ratio_corr(
    d0=None, dS=dS, num_rois=30000, nnpop=10, rnpop=10, seed=None
):
    nnidx_dict = {}
    rnidx_dict = {}
    nncorr_dict = {}
    rncorr_dict = {}

    for i in range(num_rois):
        random.seed(seed)
        roi = d0[:, i]
        nn_idx = find_nearest_nbrs(dS, i, n=nnpop)
        nn_roi = d0[:, nn_idx]
        rn_idx = pick_random_nbrs(i, len0=num_rois, n=rnpop)
        rn_roi = d0[:, rn_idx]
        nrcorr = []
        rncorr = []
        for j in range(nn_roi.shape[1]):
            nn_corr = np.corrcoef(roi, nn_roi[:, j])[0, 1]
            rn_corr = np.corrcoef(roi, rn_roi[:, j])[0, 1]
            nrcorr.append(nn_corr)
            rncorr.append(rn_corr)
            scollect_nc_min_max.append(nn_corr)
            scollect_rn_min_max.append(rn_corr)
            nnidx_dict[i] = nn_idx
            rnidx_dict[i] = rn_idx
            nncorr_dict[i] = nrcorr  # groups of near correlations
            rncorr_dict[i] = rncorr  # groups of random correlations
            srnr_arr = np.array(scollect_nc_min_max) / np.array(scollect_rn_min_max)

            sPRN = round(np.percentile(srnr_arr, 90), 3)
            filtered1 = [a for a in srnr_arr if a > 0 and a <= sPRN]

            # outliers for reasonable distribution
            filtered1 = reject_outliers(srnr_arr)
            mid = np.median(filtered1)
            vmin = min(filtered1)
            vmax = max(filtered1)

            plt.figure(figsize=(20, 20))
            custom_norm = TwoSlopeNorm(vcenter=mid, vmin=vmin, vmax=vmax)
            ax = plt.axes()
            for i in range(num_rois):
                plt.scatter(
                    x[nnidx_dict[i]],
                    y[nnidx_dict[i]],
                    marker=".",
                    norm=custom_norm,
                    cmap="rainbow",
                    s=0.2,
                    c=[np.mean(np.array(nncorr_dict[i]) / np.array(rncorr_dict[i]))]
                    * 10,
                )

            plt.colorbar()
            plt.xlabel("X Positions", fontsize=20)
            plt.ylabel("Y Positions", fontsize=20)
            plt.title(
                "Smoothed Correlation ratios of near ROIs to seeded random ROIs",
                fontsize=20,
            )
            ax.set_facecolor("black")
            plt.savefig(f"ratioCorrelations_zebrafish{num_rois}_seed{seed}_{rnpop}.png")
            plt.close()
            # need to run this for different vmax, vmin, mid=1 (first make a function initerim a class later.a)


if __name__ == "__main__":

    def main():
        num_rois = 100
        seed = None
        nnpop = 10
        rnpop = 10

        # set paths for the files
        path0 = glob.glob(
            "/camp/home/duuta/working/duuta/ppp0/data/zebf00/TimeSeries.h5"
        )
        path1 = glob.glob(
            "/camp/home/duuta/working/duuta/ppp0/data/zebf00/data_full.mat"
        )

        # read file paths
        x, y, _, d0 = read_data(path0, path1, num_rois=num_rois, scells=True)

        # compute distances between rois
        dS = compute_distance_matrix(x, y)

        # compute correlation ratio and render plot
        compute_render_ratio_corr(
            d0=d0, dS=dS, num_rois=num_rois, nnpop=nnpop, rnpop=rnpop, seed=seed
        )
