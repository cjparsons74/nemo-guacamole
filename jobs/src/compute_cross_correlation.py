import os
import time
from glob import glob

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import loadmat


def get_tag_paths(zebfolderpath):
    tag = zebfolderpath.split("/")[8]

    files = os.listdir(zebfolderpath)
    files.sort()
    files.remove("README_LICENSE.rtf")

    path0, path1 = [zebfolderpath + "/" + fname for fname in files]

    return tag, path0, path1


def compute_cross_corr(sig0, sig1):
    xcorr = signal.correlate(sig0, sig1)
    lags = signal.correlation_lags(len(sig0), len(sig1))

    return lags, xcorr


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


def figure_sanity_checks(
    d0=None,
    autolag0=None,
    autocorr0=None,
    dlags0=None,
    dxcorrs=None,
    idx0=None,
    nnrois_ids=None,
    nnrois_corr=None,
    iter=None,
    sdir=None,
):
    mpl.rcParams["lines.linewidth"] = 0.5

    # headers
    plt.figure(figsize=(20, 20))
    fig0, axs = plt.subplots(
        nrows=11, ncols=2, figsize=(15, 15), layout="tight", frameon=False
    )

    axs[0, 0].plot(
        d0[:, idx0][80:1880], label=f"ROI {idx0} " 
    )
    axs[0, 0].tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    axs[0, 0].legend()

    axs[0, 1].plot(
        autolag0, autocorr0, label=f"Lag at max correlation {autolag0[np.argmax(autocorr0)]}"
    )
    axs[0, 1].tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    axs[0, 1].legend()

    # Subplots for signals

    axs[1, 0].plot(
        d0[:, nnrois_ids[0]][80:1880],
        label=f" Neuron {nnrois_ids[0]} : xcorr \w ROI av={np.mean(nnrois_corr[0])}",
    )
    axs[1, 0].tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    axs[1, 0].legend()

    axs[2, 0].plot(
        d0[:, nnrois_ids[1]][80:1880],
        label=f"Neuron {nnrois_ids[1]} : xcorr \w ROI av={np.mean(nnrois_corr[1])}",
    )
    axs[2, 0].tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    axs[2, 0].legend()

    axs[3, 0].plot(
        d0[:, nnrois_ids[2]][80:1880],
        label=f"Neuron {nnrois_ids[2]} : xcorr \w ROI av={np.mean(nnrois_corr[2])}",
    )
    axs[3, 0].tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    axs[3, 0].legend()

    axs[4, 0].plot(
        d0[:, nnrois_ids[3]][80:1880],
        label=f"Neuron {nnrois_ids[3]} : xcorr \w ROI av={np.mean(nnrois_corr[3])}",
    )
    axs[4, 0].tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    axs[4, 0].legend()

    axs[5, 0].plot(
        d0[:, nnrois_ids[4]][80:1880],
        label=f"Neuron {nnrois_ids[4]} : xcorr \w ROI av={np.mean(nnrois_corr[4])}",
    )
    axs[5, 0].tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    axs[5, 0].legend()

    axs[6, 0].plot(
        d0[:, nnrois_ids[5]][80:1880],
        label=f"Neuron {nnrois_ids[5]}: xcorr \w ROI av={np.mean(nnrois_corr[5])}",
    )
    axs[6, 0].tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    axs[6, 0].legend()

    axs[7, 0].plot(
        d0[:, nnrois_ids[6]][80:1880],
        label=f"Neuron {nnrois_ids[6]} : xcorr \w ROI av={np.mean(nnrois_corr[6])}",
    )
    axs[7, 0].tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    axs[7, 0].legend()

    axs[8, 0].plot(
        d0[:, nnrois_ids[7]][80:1880],
        label=f"Neuron {nnrois_ids[7]} : xcorr \w ROI av={np.mean(nnrois_corr[7])}",
    )
    axs[8, 0].tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    axs[8, 0].legend()

    axs[9, 0].plot(
        d0[:, nnrois_ids[8]][80:1880],
        label=f"Neuron {nnrois_ids[8]} : xcorr \w ROI av={np.mean(nnrois_corr[8])}",
    )
    axs[9, 0].tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    axs[9, 0].legend()

    axs[10, 0].plot(
        d0[:, nnrois_ids[9]][80:1880],
        label=f"Neuron {nnrois_ids[9]} : xcorr \w ROI av={np.mean(nnrois_corr[9])}",
    )
    axs[10, 0].set_xlabel("Signal")
    axs[10, 0].set_ylabel("AU")
    axs[10, 0].legend()

    # subplots for lags and xcorrs
    axs[1, 1].plot(
        dlags0[0],
        dxcorrs[0],
        label=f"Lag at max correlation: {dlags0[0][np.argmax(dxcorrs[0])]}",
    )
    axs[1, 1].tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    axs[1, 1].legend()

    axs[2, 1].plot(
        dlags0[1],
        dxcorrs[1],
        label=f"Lag at max correlation: {dlags0[1][np.argmax(dxcorrs[1])]}",
    )
    axs[2, 1].tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    axs[2, 1].legend()

    axs[3, 1].plot(
        dlags0[2],
        dxcorrs[2],
        label=f"Lag at max correlation: {dlags0[2][np.argmax(dxcorrs[2])]}",
    )
    axs[3, 1].tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    axs[3, 1].legend()

    axs[4, 1].plot(
        dlags0[3],
        dxcorrs[3],
        label=f"Lag at max correlation: {dlags0[3][np.argmax(dxcorrs[3])]}",
    )
    axs[4, 1].tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    axs[4, 1].legend()

    axs[5, 1].plot(
        dlags0[4],
        dxcorrs[4],
        label=f"Lag at max correlation: {dlags0[4][np.argmax(dxcorrs[4])]}",
    )
    axs[5, 1].tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    axs[5, 1].legend()

    axs[6, 1].plot(
        dlags0[5],
        dxcorrs[5],
        label=f"Lag at max correlation: {dlags0[5][np.argmax(dxcorrs[5])]}",
    )
    axs[6, 1].tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    axs[6, 1].legend()

    axs[7, 1].plot(
        dlags0[6],
        dxcorrs[6],
        label=f"Lag at max correlation: {dlags0[6][np.argmax(dxcorrs[6])]}",
    )
    axs[7, 1].tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    axs[7, 1].legend()

    axs[8, 1].plot(
        dlags0[7],
        dxcorrs[7],
        label=f"Lag at max correlation: {dlags0[7][np.argmax(dxcorrs[7])]}",
    )
    axs[8, 1].tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    axs[8, 1].legend()

    axs[9, 1].plot(
        dlags0[8],
        dxcorrs[8],
        label=f"Lag at max correlation: {dlags0[8][np.argmax(dxcorrs[8])]}",
    )
    axs[9, 1].tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    axs[9, 1].legend()

    axs[10, 1].plot(
        dlags0[9],
        dxcorrs[9],
        label=f"Lag at max correlation: {dlags0[9][np.argmax(dxcorrs[9])]}",
    )
    axs[10, 1].set_xlabel("Lags")
    axs[10, 1].set_ylabel("X - Correlation")
    axs[10, 1].legend()

    fig0.suptitle("Signal                                        Correlations and Lags")
    plt.savefig(
        f"{sdir}cross_correlation_sanity_checks_plots_{idx0}_{iter}.png",
        bbox_inches="tight",
    )
    plt.close()

    return 0


def main():
    # set vars and path
    num_rois = 300
    num_rois_list = list(range(num_rois))
    Idx0 = 1408
    PPATH = "/camp/home/duuta/working/duuta/ppp0/data/zeb*"
    sdir = "/camp/home/duuta/working/duuta/jobs/plots/xCorr/"

    # get zeb folder path(s) and path for specfic files
    zpath = glob(PPATH)[0]
    _, path0, path1 = get_tag_paths(zpath)
    _, _, _, d0 = read_data(path0, path1, num_rois, scells=True)

    # set neuron of interest
    sig0 = d0[:, Idx0]

    # compute autocorrelations
    autolags, autocorr = compute_cross_corr(sig0, sig0)

    for iter in range(int(num_rois / 10)):
        # dict to store sigidx associate lags and xcorrs
        _lags = {}
        _ccorr = {}

        # get a signal
        for idx in num_rois_list[iter * 10 : iter * 10 + 10]:
            idx0 = idx % 10

            sig1 = d0[:, idx]
            lags, xcorr = compute_cross_corr(sig0, sig1)

            _lags[idx0] = lags
            _ccorr[idx0] = xcorr

        figure_sanity_checks(
            d0=d0,
            dlags0=_lags,
            dxcorrs=_ccorr,
            autolag0=autolags,
            autocorr0=autocorr,
            idx0=Idx0,
            nnrois_ids=list(_ccorr.keys()),
            nnrois_corr=_ccorr,
            iter=iter,
            sdir=sdir,
        )
        plt.close()


if __name__ == "__main__":
    st = time.time()
    main()
    print(f"{time.time() - st} secs ----------- all done")
