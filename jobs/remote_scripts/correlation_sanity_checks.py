import os
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt




def load_xcross_lags(f0, f1):
    xcorr = np.load(f0)
    lags =np.load(f1)

    return xcorr, lags


def corr_sanity_checks(d0, roi_list, nncorr_dict, nnidx_dict, rnidx_dict, rncorr_dict, xlims=[80, 1880],  lw= 0.5):
    # correctness plots
    mpl.rcParams['lines.linewidth'] = lw
    # plot segment of signal. [80:1880], run to test for first two cases
    # frames for ROIs
    # find max correlations 
    llim, rlim = xlims[0], xlims[1]

    for j in tqdm(roi_list[:50]):
        avncorr = round(np.mean(nncorr_dict[j]), 3) # index goes here
        avrcorr = round(np.mean(rncorr_dict[j]), 3)  # index goes here
        
        plt.figure(figsize=(20, 20))
        fig0, axs = plt.subplots(nrows=11, ncols=2, figsize=(15, 15), layout='tight', frameon=False)

        axs[0, 0].plot(d0[:, j][llim:rlim], label=f'ROI {j} : avg correlation with near nbrs={avncorr}')
        axs[0, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axs[0, 0].legend()
        
        
        axs[0, 1].plot(d0[:, j][llim:rlim], label=f'ROI {j}: avg correlation with rand nbrs={avrcorr}')
        axs[0, 1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axs[0, 1].legend()
        

        # switch variables of ease
        nnrois_ids = nnidx_dict[j]
        nnrois_corr = nncorr_dict[j]
        # frames of rendering nnROIs
        
        axs[1, 0].plot(d0[:, nnrois_ids[0]][llim:rlim], label=f' Neuron {nnrois_ids[0]} : corr \w ROI={round(nnrois_corr[0], 3)}')
        axs[1, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axs[1, 0].legend()

        
        axs[2, 0].plot(d0[:, nnrois_ids[1]][llim:rlim], label= f"Neuron {nnrois_ids[1]} : corr \w ROI={round(nnrois_corr[1], 3)}")
        axs[2, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axs[2, 0].legend()

        
        axs[3, 0].plot(d0[:, nnrois_ids[2]][llim:rlim], label=f"Neuron {nnrois_ids[2]} : corr \w ROI={round(nnrois_corr[2], 3)}")
        axs[3, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axs[3, 0].legend()

        
        axs[4, 0].plot(d0[:, nnrois_ids[3]][llim:rlim], label=f"Neuron {nnrois_ids[3]} : corr \w ROI={round(nnrois_corr[3], 3)}")
        axs[4, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axs[4, 0].legend()

        
        axs[5, 0].plot(d0[:, nnrois_ids[4]][llim:rlim], label=f"Neuron {nnrois_ids[4]} : corr \w ROI={round(nnrois_corr[4], 3)}")
        axs[5, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axs[5, 0].legend()

        
        axs[6, 0].plot(d0[:, nnrois_ids[5]][llim:rlim], label=f"Neuron {nnrois_ids[5]}: corr \w ROI={round(nnrois_corr[5], 3)}")
        axs[6, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axs[6, 0].legend()

       
        axs[7, 0].plot(d0[:, nnrois_ids[6]][llim:rlim], label=f"Neuron {nnrois_ids[6]} : corr \w ROI={round(nnrois_corr[6], 3)}")
        axs[7, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axs[7, 0].legend()

        
        axs[8, 0].plot(d0[:, nnrois_ids[7]][llim:rlim], label=f"Neuron {nnrois_ids[7]} : corr \w ROI={round(nnrois_corr[7], 3)}")
        axs[8, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axs[8, 0].legend()

        
        axs[9, 0].plot(d0[:, nnrois_ids[8]][llim:rlim], label=f"Neuron {nnrois_ids[8]} : corr \w ROI={round(nnrois_corr[8], 3)}")
        axs[9, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axs[9, 0].legend()

        
        axs[10, 0].plot(d0[:, nnrois_ids[9]][llim:rlim], label=f"Neuron {nnrois_ids[9]} : corr \w ROI={round(nnrois_corr[9], 3)}")
        axs[10, 0].legend()

        # switching vars for ease
        rnrois=  rnidx_dict[j]  # index goes here
        rnrois_corr = rncorr_dict[j]  # index goes here
        # frames rendering rnROIs
       
        axs[1, 1].plot(d0[:, r
        axs[1, 1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axs[1, 1].legend()

        
        axs[2, 1].plot(d0[:, rnrois[1]][llim:rlim], label=f"Neuron {rnrois[1]} : corr \w ROI={round(rnrois_corr[1], 3)}")
        axs[2, 1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axs[2, 1].legend()

        
        axs[3, 1].plot(d0[:, rnrois[2]][llim:rlim], label=f"Neuron {rnrois[2]} : corr \w ROI={round(rnrois_corr[2], 3)}")
        axs[3, 1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axs[3, 1].legend()

        
        axs[4, 1].plot(d0[:, rnrois[3]][llim:rlim], label=f"Neuron {rnrois[3]} : corr \w ROI={round(rnrois_corr[3], 3)}")
        axs[4, 1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axs[4, 1].legend()

        
        axs[5, 1].plot(d0[:, rnrois[4]][llim:rlim], label=f"Neuron {rnrois[4]} : corr \w ROI={round(rnrois_corr[4], 3)}")
        axs[5, 1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axs[5, 1].legend()

        
        axs[6, 1].plot(d0[:, rnrois[5]][llim:rlim], label=f"Neuron {rnrois[5]}: corr \w ROI={round(rnrois_corr[5], 3)}")
        axs[6, 1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axs[6, 1].legend()

        
        axs[7, 1].plot(d0[:, rnrois[6]][llim:rlim], label=f"Neuron {rnrois[6]} : corr \w ROI={round(rnrois_corr[6], 3)}")
        axs[7, 1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axs[7, 1].legend()

        
        axs[8, 1].plot(d0[:, rnrois[7]][llim:rlim], label=f"Neuron {rnrois[7]} : corr \w ROI={round(rnrois_corr[7], 3)}")
        axs[8, 1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axs[8, 1].legend()

        
        axs[9, 1].plot(d0[:, rnrois[8]][llim:rlim], label=f"Neuron {rnrois[8]}: corr \w ROI={round(rnrois_corr[8], 3)}")
        axs[9, 1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axs[9, 1].legend()

        
        axs[10, 1].plot(d0[:, rnrois[9]][llim:rlim], label=f"Neuron {rnrois[9]}: corr \w ROI={round(rnrois_corr[9], 3)}")
        axs[10, 1].legend()

        fig0.suptitle('Cross correlation sanity checks')
        plt.savefig(f"sanity_checks_cross_correlations_{j}.png")
        plt.close()



def main():
    # paths
    p0 = '/camp/home/duuta/job_templates/xcorr_dict.npy' 
    p1 = '/camp/home/duuta/job_templates/lags_dict.npy' 

    # load data files
    f0, f1 = load_xcross_lags(p0, p1)

    # read 


if __name__ == "__main__":
   main()
