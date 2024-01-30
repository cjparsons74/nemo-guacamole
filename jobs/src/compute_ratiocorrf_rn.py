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


def compute_render_ratio_corr(
    x,
    y,
    d0,
    dS,
    num_rois=30000,
    nnpop=10,
    rnpop=10,
    seed=None,
    tag=None,
    sdir=None,
    outliers=False,
):
    nnidx_dict = {}
    rnidx_dict = {}
    nncorr_dict = {}
    rncorr_dict = {}
    ratioCorr = {}
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

        ratioCorr[roi_idx] = sum(nrcorr) / sum(rncorr)
        nnidx_dict[roi_idx] = nn_idx
        rnidx_dict[roi_idx] = rn_idx
        nncorr_dict[roi_idx] = nrcorr  # groups of near correlations
        rncorr_dict[roi_idx] = rncorr  # groups of random correlations

    rnr_arr = np.array(collect_nn_min_max) / np.array(collect_rn_min_max)

    np.save("ratioCorr.npy", ratioCorr)

    if not outliers:
        filtered0 = reject_outliers(rnr_arr, m=2)
    else:
        PRN = round(np.percentile(rnr_arr, 90), 3)
        filtered0 = [a for a in rnr_arr if a > 0 and a <= PRN]

    # remove outliers for all behavioured distribution
    mid = np.median(filtered0)
    print("mid", mid)
    vmin = min(filtered0)
    print("vmin", vmin)
    vmax = max(filtered0)
    print("vmax", vmax)

    # plotting figure
    plt.figure(figsize=(20, 20))
    custom_norm = TwoSlopeNorm(vcenter=1, vmin=-50, vmax=50)
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
                np.array(nncorr_dict[roi_idx]).sum()
                / np.array(rncorr_dict[roi_idx]).sum()
            ]
            * 10,
        )

    plt.colorbar(aspect=10)
    plt.xlabel("ROI X Positions", fontsize=20)
    plt.ylabel("ROI Y Positions", fontsize=20)
    plt.title(
        f"{tag}:Correlation ratios of near ROIs to random ROIs: {rnpop} seed:{seed}",
        fontsize=20,
    )
    ax.set_facecolor("black")

    plt.savefig(
        f"{sdir}fo_{tag}_ROIs:{num_rois}_seed:{seed}_{rnpop}.png", bbox_inches="tight"
    )
    plt.close()

    # need to run this for different vmax, vmin, mid=1 (first make a function initerim a class later)
    # filtered outliers (fo) computing the number ratio of sum(nn)/sum(rc) instead of nn_1/rn_1


def main():
    num_rois = 100
    seed = None
    nnpop = 10
    sdir = "/camp/home/duuta/job_templates/plots/ratioCorr/"

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

            print("frantically computing distances......")

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
                outliers=False,
            )


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"{time.time() - start_time} ----all done")
