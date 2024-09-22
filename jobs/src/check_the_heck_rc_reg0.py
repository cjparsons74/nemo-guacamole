import os
import random
import time
from glob import glob

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

# Rather than have a copy of the file with one or two lines different, it would be better to add a parameter when calling
# the code to say which reg to use "python check_the_heck_rc.py reg0" and use sys.argv[1] to get the reg to use
# Alternatively use the argparse module for command line procesing (I think that is the modern one)


def get_tag_paths(zebfolderpath):
    tag = zebfolderpath.split("/")[8]

    files = os.listdir(zebfolderpath)
    files.sort()
    files.remove("README_LICENSE.rtf")

    path0, path1 = [zebfolderpath + "/" + fname for fname in files]

    return tag, path0, path1


def read_data(path0, path1, num_rois, scells=True):
    dholder = h5py.File(path0, "r")
    d0 = dholder["CellResp"][:][:, :num_rois]  # responses
    d1 = loadmat(path1, simplify_cells=scells)

    eliminated_rois = d1["data"]["IX_inval_anat"]
    all_rois = d1["data"]["CellXYZ"]

    used_rois_coor = np.array(
        [row for j, row in enumerate(all_rois) if j not in list(eliminated_rois)]
    )

    x, y, z = used_rois_coor[:num_rois, :].T  # for computation of distance matrix

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


def corr_sanity_checks(
    d0,
    roi_list,
    nncorr_dict,
    nnidx_dict,
    rnidx_dict,
    rncorr_dict,
    xlims=[80, 1880],
    lw=0.5,
    sdir=None,
    num_rois=None,
    tag=None,
):
    # correctness plots
    mpl.rcParams["lines.linewidth"] = lw
    # plot segment of signal. [80:1880], run to test for first two cases
    # frames for ROIs
    # find max correlations
    llim, rlim = xlims[0], xlims[1]

    for j in tqdm(roi_list):
        avncorr = round(np.mean(nncorr_dict[j]), 3)  # index goes here
        avrcorr = round(np.mean(rncorr_dict[j]), 3)  # index goes here

        plt.figure(figsize=(20, 20))
        fig0, axs = plt.subplots(
            nrows=11, ncols=2, figsize=(15, 15), layout="tight", frameon=False
        )

        axs[0, 0].plot(
            d0[:, j][llim:rlim],
            label=f"ROI {j} : avg correlation with near nbrs={avncorr}",
        )
        axs[0, 0].tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        axs[0, 0].legend()

        axs[0, 1].plot(
            d0[:, j][llim:rlim],
            label=f"ROI {j}: avg correlation with rand nbrs={avrcorr}",
        )
        axs[0, 1].tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        axs[0, 1].legend()

        # switch variables of ease
        nnrois_ids = nnidx_dict[j]
        nnrois_corr = nncorr_dict[j]
        # frames of rendering nnROIs

        axs[1, 0].plot(
            d0[:, nnrois_ids[0]][llim:rlim],
            label=f" Neuron {nnrois_ids[0]} : corr \w ROI={round(nnrois_corr[0], 3)}",
        )
        axs[1, 0].tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        axs[1, 0].legend()

        axs[2, 0].plot(
            d0[:, nnrois_ids[1]][llim:rlim],
            label=f"Neuron {nnrois_ids[1]} : corr \w ROI={round(nnrois_corr[1], 3)}",
        )
        axs[2, 0].tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        axs[2, 0].legend()

        axs[3, 0].plot(
            d0[:, nnrois_ids[2]][llim:rlim],
            label=f"Neuron {nnrois_ids[2]} : corr \w ROI={round(nnrois_corr[2], 3)}",
        )
        axs[3, 0].tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        axs[3, 0].legend()

        axs[4, 0].plot(
            d0[:, nnrois_ids[3]][llim:rlim],
            label=f"Neuron {nnrois_ids[3]} : corr \w ROI={round(nnrois_corr[3], 3)}",
        )
        axs[4, 0].tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        axs[4, 0].legend()

        axs[5, 0].plot(
            d0[:, nnrois_ids[4]][llim:rlim],
            label=f"Neuron {nnrois_ids[4]} : corr \w ROI={round(nnrois_corr[4], 3)}",
        )
        axs[5, 0].tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        axs[5, 0].legend()

        axs[6, 0].plot(
            d0[:, nnrois_ids[5]][llim:rlim],
            label=f"Neuron {nnrois_ids[5]}: corr \w ROI={round(nnrois_corr[5], 3)}",
        )
        axs[6, 0].tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        axs[6, 0].legend()

        axs[7, 0].plot(
            d0[:, nnrois_ids[6]][llim:rlim],
            label=f"Neuron {nnrois_ids[6]} : corr \w ROI={round(nnrois_corr[6], 3)}",
        )
        axs[7, 0].tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        axs[7, 0].legend()

        axs[8, 0].plot(
            d0[:, nnrois_ids[7]][llim:rlim],
            label=f"Neuron {nnrois_ids[7]} : corr \w ROI={round(nnrois_corr[7], 3)}",
        )
        axs[8, 0].tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        axs[8, 0].legend()

        axs[9, 0].plot(
            d0[:, nnrois_ids[8]][llim:rlim],
            label=f"Neuron {nnrois_ids[8]} : corr \w ROI={round(nnrois_corr[8], 3)}",
        )
        axs[9, 0].tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        axs[9, 0].legend()

        axs[10, 0].plot(
            d0[:, nnrois_ids[9]][llim:rlim],
            label=f"Neuron {nnrois_ids[9]} : corr \w ROI={round(nnrois_corr[9], 3)}",
        )
        axs[10, 0].legend()

        # switching vars for ease
        rnrois = rnidx_dict[j]  # index goes here
        rnrois_corr = rncorr_dict[j]  # index goes here
        # frames rendering rnROIs

        axs[1, 1].plot(
            d0[:, rnrois[0]][llim:rlim],
            label=f"Neuron {rnrois[0]} : corr \w ROI={round(rnrois_corr[0], 3)}",
        )
        axs[1, 1].tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        axs[1, 1].legend()

        axs[2, 1].plot(
            d0[:, rnrois[1]][llim:rlim],
            label=f"Neuron {rnrois[1]} : corr \w ROI={round(rnrois_corr[1], 3)}",
        )
        axs[2, 1].tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        axs[2, 1].legend()

        axs[3, 1].plot(
            d0[:, rnrois[2]][llim:rlim],
            label=f"Neuron {rnrois[2]} : corr \w ROI={round(rnrois_corr[2], 3)}",
        )
        axs[3, 1].tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        axs[3, 1].legend()

        axs[4, 1].plot(
            d0[:, rnrois[3]][llim:rlim],
            label=f"Neuron {rnrois[3]} : corr \w ROI={round(rnrois_corr[3], 3)}",
        )
        axs[4, 1].tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        axs[4, 1].legend()

        axs[5, 1].plot(
            d0[:, rnrois[4]][llim:rlim],
            label=f"Neuron {rnrois[4]} : corr \w ROI={round(rnrois_corr[4], 3)}",
        )
        axs[5, 1].tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        axs[5, 1].legend()

        axs[6, 1].plot(
            d0[:, rnrois[5]][llim:rlim],
            label=f"Neuron {rnrois[5]}: corr \w ROI={round(rnrois_corr[5], 3)}",
        )
        axs[6, 1].tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        axs[6, 1].legend()

        axs[7, 1].plot(
            d0[:, rnrois[6]][llim:rlim],
            label=f"Neuron {rnrois[6]} : corr \w ROI={round(rnrois_corr[6], 3)}",
        )
        axs[7, 1].tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        axs[7, 1].legend()

        axs[8, 1].plot(
            d0[:, rnrois[7]][llim:rlim],
            label=f"Neuron {rnrois[7]} : corr \w ROI={round(rnrois_corr[7], 3)}",
        )
        axs[8, 1].tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        axs[8, 1].legend()

        axs[9, 1].plot(
            d0[:, rnrois[8]][llim:rlim],
            label=f"Neuron {rnrois[8]}: corr \w ROI={round(rnrois_corr[8], 3)}",
        )
        axs[9, 1].tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        axs[9, 1].legend()

        axs[10, 1].plot(
            d0[:, rnrois[9]][llim:rlim],
            label=f"Neuron {rnrois[9]}: corr \w ROI={round(rnrois_corr[9], 3)}",
        )
        axs[10, 1].legend()

        fig0.suptitle("Cross correlation sanity checks")
        plt.savefig(f"{sdir}reg0_check_heck_{tag}_{num_rois}_corr_ratio_for_roi{j}.png")
        plt.close()


def compute_ratio_corr(
    d0,
    dS,
    roisIDX=None,
    nnpop=10,
    rnpop=10,
    num_rois=None,
    seed=None,
):
    nnidx_dict = {}
    rnidx_dict = {}
    nncorr_dict = {}
    rncorr_dict = {}
    ratioCorr = {}

    for (
        roi_idx
    ) in roisIDX:  # need to account for cases where the pass a list of roi indices
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

        ratioCorr[roi_idx] = [sum(nrcorr) / sum(rncorr)] * 10
        nnidx_dict[roi_idx] = nn_idx
        rnidx_dict[roi_idx] = rn_idx
        nncorr_dict[roi_idx] = nrcorr  # groups of near correlations
        rncorr_dict[roi_idx] = rncorr  # groups of random correlations

    return ratioCorr, nnidx_dict, rnidx_dict 

    # need to run this for different vmax, vmin, mid=1 (first make a function initerim a class later)


def main():
    num_rois = 20000
    seed = None
    nnpop = 10
    rnpop = 10
    sdir = "/camp/home/duuta/working/duuta/jobs/plots/checkRCregs/"

    for fpath in glob("/camp/home/duuta/working/duuta/ppp0/data/zebf*"):
        print("reading {fpath}..........")

        # get tag and paths
        tag, path0, path1 = get_tag_paths(fpath)

        # read file paths
        x, y, _, d0 = read_data(path0, path1, num_rois=num_rois, scells=True)
        print("can read the file path... yes.... frantically reading files.....")

        # reg2 = (x > 1000) & (x < 1250) & (y > 500) & (y < 650)
        # reg1 = (x > 400)  & (x < 600) & (y > 500) & (y < 650)
        reg0 = (x > 0)  & (x < 250) & (y > 550) & (y < 700)

        # compute distances between rois
        dS = compute_distance_matrix(x, y)

        print("frantically computing distances......")

        # getting idx of rois in region of interest
        roisIDX = reg0.nonzero()[0]
        print(roisIDX)

        # compute correlation ratio and render plot
        ratioCorr, nnidx_dict, rnidx_dict = compute_ratio_corr(
            d0=d0,
            dS=dS,
            roisIDX=roisIDX,
            num_rois=num_rois,
            nnpop=nnpop,
            rnpop=rnpop,
            seed=seed,
        )

        corr_sanity_checks(
            d0=d0,
            roi_list=roisIDX,
            nncorr_dict=ratioCorr,
            rncorr_dict=ratioCorr,
            nnidx_dict=nnidx_dict,
            rnidx_dict=rnidx_dict,
            sdir=sdir,
            tag=tag,
            num_rois=num_rois,
        )


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"{time.time() - start_time} ----all done")
