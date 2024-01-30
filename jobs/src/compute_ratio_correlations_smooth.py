import os
import random
import time
from glob import glob

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform


def get_tag_paths(zebfolderpath):
    tag = zebfolderpath.split("/")[8]

    files = os.listdir(zebfolderpath)
    files.sort()
    files.remove("README_LICENSE.rtf")

    path0, path1 = [zebfolderpath + "/" + fname for fname in files]

    return tag, path0, path1


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


def find_nearest_nbrs(ds, roi_idx, n=10):
    nn_idx = ds[roi_idx,].argsort()[1 : n + 1]

    return nn_idx


def compute_distance_matrix(x, y):
    ds = squareform(pdist(np.array([x, y]).T, metric="euclidean"))

    return ds


def pick_random_nbrs(roi_idx, len0=100, n=10):
    all_idx = list(range(len0))
    all_idx.remove(roi_idx)
    rn_idx = random.sample(all_idx, n)

    return rn_idx


def reject_outliers(data, m=2):
    X = data[abs(data - np.mean(data)) < m * np.std(data)]

    return X


def smooth_corr(foobar, nn_list):
    smooth_ver = 0 * foobar
    for idx, _ in enumerate(foobar):
        x0 = np.mean(foobar[nn_list[idx]])
        smooth_ver[idx] = x0
    return smooth_ver


def compute_render_ratio_corr(
    x, y, d0, dS, num_rois=30000, nnpop=10, rnpop=10, seed=None, tag=None, sdir=None
):
    nnidx_dict = {}
    rnidx_dict = {}
    nncorr_dict = {}
    rncorr_dict = {}
    collect_nn_min_max = []
    collect_rn_min_max = []

    for roi_idx in range(
        num_rois
    ):  # need to account for cases where the pass a list of roi indices
        random.seed(seed)
        roi = d0[:, roi_idx]
        nn_idx = find_nearest_nbrs(dS, roi_idx, n=nnpop)
        nn_roi = d0[:, nn_idx]
        rn_idx = pick_random_nbrs(roi_idx, len0=num_rois, n=rnpop)
        rn_roi = d0[:, rn_idx]
        nrcorr = []
        rncorr = []

        for j in range(nn_roi.shape[1]):
            nn_corr = np.corrcoef(roi, nn_roi[:, j])[0, 1]
            rn_corr = np.corrcoef(roi, rn_roi[:, j])[0, 1]
            nrcorr.append(nn_corr)
            rncorr.append(rn_corr)
            collect_nn_min_max.append(nn_corr)
            collect_rn_min_max.append(rn_corr)

        nnidx_dict[roi_idx] = nn_idx
        rnidx_dict[roi_idx] = rn_idx
        nncorr_dict[roi_idx] = nrcorr  # groups of near correlations
        rncorr_dict[roi_idx] = rncorr  # groups of random correlations

    # srnr_arr = np.array(collect_nn_min_max) / np.array(collect_rn_min_max)
    # sPRN = round(np.percentile(srnr_arr, 90), 3)
    # filtered0 = [a for a in srnr_arr if a > 0 and a <= sPRN]

    # outliers for reasonable distribution
    # filtered1 = reject_outliers(srnr_arr)
    # mid = np.median(filtered0)
    # print('mid', mid)
    # vmin = min(filtered0)
    # print('vmin', vmin)
    # vmax = max(filtered0)
    # print('vmax', vmax)

    plt.figure(figsize=(20, 20))
    custom_norm = TwoSlopeNorm(vcenter=1, vmin=0.001, vmax=9)
    ax = plt.axes()

    for roi_idx in range(
        num_rois
    ):  # need to take care of cases where num_rois is of indexes
        plt.scatter(
            x[nnidx_dict[roi_idx]],
            y[nnidx_dict[roi_idx]],
            marker=".",
            norm=custom_norm,
            cmap="rainbow",
            s=0.5,
            c=[
                np.array(nncorr_dict[roi_idx]).mean()
            ]
            * len(nncorr_dict[roi_idx]),
        )

    plt.colorbar(shrink=0.5)
    plt.xlabel("ROI X Positions", fontsize=20)
    plt.ylabel("ROI Y Positions", fontsize=20)
    plt.margins(x=0, y=0)
    plt.title(
        f"{tag}: Smooth correlation ratios of near ROIs:{nnpop} to random ROIs:{rnpop} seed:{seed}",
        fontsize=20,
    )
    ax.set_facecolor("black")
    plt.tight_layout()

    plt.savefig(
        f"{sdir}smoothratiocorrelations_{tag}_ROIs:{num_rois}_NN:{nnpop}_seed:{seed}_RN:{rnpop}.png",
    )
    plt.close()

    # need to run this for different vmax, vmin, mid=1 (first make a function initerim a class later)


def main():
    num_rois = 5000
    seed = None
    nnpop = 10
    sdir = "/camp/home/duuta/working/duuta/jobs/plots/ratioCorr/"

    for rnpop in [10, 100, 1000]:
        for fpath in glob("/camp/home/duuta/working/duuta/ppp0/data/zebf*"):
            print(f"reading {fpath}..........")

            # get tag and paths
            tag, path0, path1 = get_tag_paths(fpath)

            # read file paths
            x, y, _, d0 = read_data(path0, path1, num_rois=num_rois, scells=True)
            print("can read the file path... yes.... frantically reading files.....")

            # compute distances between rois
            dS = compute_distance_matrix(x, y)

            print("franticall computing distances......")

            # compute correlation ratio and render plot
            compute_render_ratio_corr(
                x,
                y,
                d0=d0,
                dS=dS,
                num_rois=num_rois,
                nnpop=nnpop,
                rnpop=rnpop,
                seed=seed,
                tag=tag,
                sdir=sdir,
            )


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"{time.time() - start_time} ----all done")
