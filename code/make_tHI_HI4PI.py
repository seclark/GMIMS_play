import numpy as np
import healpy as hp
import h5py
import copy

import sys 
sys.path.insert(0, '../../ForegroundModels/code')
from make_data_table import get_RHT_data, get_list_fns, get_start_stop_hpx
sys.path.insert(0, '../../RHT')
import RHT_tools

def masked_smoothing(U, rad=5.0):    
    # from https://stackoverflow.com/questions/50009141/smoothing-without-filling-missing-values-with-zeros 
    V=U.copy()
    V[U!=U]=0
    VV=hp.smoothing(V, fwhm=np.radians(rad))    
    W=0*U.copy()+1
    W[U!=U]=0
    WW=hp.smoothing(W, fwhm=np.radians(rad))    
    return VV/WW
    
def smooth_overnans(map, fwhm = 15):
    """
    Takes map with nans, etc
    """
    mask = np.ones(map.shape, np.float_)
    mask[np.isnan(map)] = 0
    
    map_zeroed = copy.copy(map)
    map_zeroed[mask == 0] = 0
    
    blurred_map = hp.smoothing(map_zeroed, fwhm=fwhm, pol=False)
    blurred_mask = hp.smoothing(mask, fwhm=fwhm, pol=False)
    
    map = blurred_map / blurred_mask
  
    return map

if __name__ == "__main__":
    LOCAL = False
        
    # where the RHT of HI4PI data is stored
    if LOCAL:
        data_root = "../data/HI4PI_RHT/"
        hi4pi_root = "/Users/Dropbox/HI4PI/"
    else:
        data_root = "../../HI4PI/slice_data/"
        hi4pi_root = "/data/seclark/HI4PI/full_data/"

        
    # dimensions of the data
    nside = 1024
    npix = 12*nside**2
    ntheta = 165
    
    # smooth QRHT and URHT
    smoothQU = True
    smooth_fwhm = 180
    if smoothQU:
        smoothQUstr = "_smooth{}".format(smooth_fwhm)
    else:
        smoothQUstr = ""
    
    # Values of theta for RHT output
    wlen = 75
    thets = RHT_tools.get_thets(wlen, save = False)
    
    all_vels = np.arange(81, 102)
    nvels = len(all_vels)
    startvel = all_vels[0]
    stopvel = all_vels[-1]
    
    #IRHT = np.zeros((npix, nvels), np.float_)
    #QRHT = np.zeros((npix, nvels), np.float_)
    #URHT = np.zeros((npix, nvels), np.float_)
    HI_n_v = np.zeros((npix, nvels), np.float_)
    #theta_RHT_n_v = np.zeros((npix, nvels), np.float_)
    #backproj_n_v = np.zeros((npix, nvels), np.float_)
    IRHT_tot = np.zeros(npix)
    QRHT_tot = np.zeros(npix)
    URHT_tot = np.zeros(npix)
    I_HI_tot = np.zeros(npix)
    
    # step through all velocities
    for v_i, _vel in enumerate(all_vels):
        
        IRHT = np.zeros(npix)
        QRHT = np.zeros(npix)
        URHT = np.zeros(npix)
        
        # load HI intensity data 
        hi4pi_fn = hi4pi_root + 'HI4PI_120kms.h5'
        with h5py.File(hi4pi_fn, 'r') as f:
            HI_n_v[:, v_i] = f['survey'][:, _vel]
        
        # get all filenames for given velocity
        fns = get_list_fns(_vel, gal=True, data_root=data_root)
        
        for fn in fns:
            #hpix, hthets, backproj = get_RHT_data(data_root+fn, returnbp=True)
            hpix, hthets = get_RHT_data(data_root+fn, returnbp=False)
            #print(hthets[0], len(hthets), hthets.shape)
            #IRHT[hpix, v_i] = np.nansum(np.array(hthets), axis=1)
            #QRHT[hpix, v_i] = np.nansum(np.cos(2*thets)*hthets, axis=1)
            #URHT[hpix, v_i] = np.nansum(np.sin(2*thets)*hthets, axis=1)
            IRHT[hpix] = np.nansum(np.array(hthets), axis=1)
            QRHT[hpix] = np.nansum(np.cos(2*thets)*hthets, axis=1)
            URHT[hpix] = np.nansum(np.sin(2*thets)*hthets, axis=1)
            
            #backproj_n_v[hpix, v_i] = backproj[hpix]
            #print("number of hpix: {}, nonzero IRHT: {}, backproj: {}".format(len(hpix), len(np.nonzero(IRHT[:, v_i])[0]), len(np.nonzero(backproj_n_v[:, v_i])[0])))
            
            # note: np.nansum(hthets) != np.nansum(backproj) because the backprojection is normalized!
        #IRHTslice = IRHT[:, v_i]
        #print("for vel {}, setting {} to zero".format(_vel, len(np.where(IRHTslice <= 0)[0])))
        #QRHT[np.where(IRHTslice <= 0), v_i] = None # set to nan
        #URHT[np.where(IRHTslice <= 0), v_i] = None
        #print("nonnan Q = {}".format(len(np.nonzero(QRHT[~np.isnan(QRHT)])[0])))
        
        # remove RHT amplitude by normalizing to 1
        thetaRHT = np.mod(0.5*np.arctan2(URHT, QRHT), np.pi)
        QRHT = np.cos(2*thetaRHT)
        URHT = np.sin(2*thetaRHT)
        
        #QRHT[np.where(IRHT <= 0), v_i] = None # set to nan
        #URHT[np.where(IRHT <= 0), v_i] = None
        QRHT[np.where(IRHT <= 0)] = None # set to nan
        URHT[np.where(IRHT <= 0)] = None
        
        # smooth each map
        #IRHT[:, v_i] = smooth_overnans(IRHT[:, v_i], fwhm=np.radians(smooth_fwhm/60.))
        #QRHT[:, v_i] = smooth_overnans(QRHT[:, v_i], fwhm=np.radians(smooth_fwhm/60.))
        #URHT[:, v_i] = smooth_overnans(URHT[:, v_i], fwhm=np.radians(smooth_fwhm/60.))
        #HI_n_v[:, v_i] = smooth_overnans(HI_n_v[:, v_i], fwhm=np.radians(smooth_fwhm/60.))
        IRHTsmooth = smooth_overnans(IRHT, fwhm=np.radians(smooth_fwhm/60.))
        QRHTsmooth = smooth_overnans(QRHT, fwhm=np.radians(smooth_fwhm/60.))
        URHTsmooth = smooth_overnans(URHT, fwhm=np.radians(smooth_fwhm/60.))
        I_HIsmooth = smooth_overnans(HI_n_v[:, v_i], fwhm=np.radians(smooth_fwhm/60.))
        
        
        #(IRHT[:, v_i], QRHT[:, v_i], URHT[:, v_i]) = hp.sphtfunc.smoothing([IRHT[:, v_i], QRHT[:, v_i], URHT[:, v_i]], fwhm=np.radians(smooth_fwhm/60.), pol=True)
        #HI_n_v[:, v_i] = hp.sphtfunc.smoothing(HI_n_v[:, v_i], fwhm=np.radians(smooth_fwhm/60.), pol=False)
        
        #print("IRHT shape {} IRHTslice shape {}".format(IRHT.shape, IRHTslice.shape))
        #theta_RHT_n_v[:, v_i] = np.mod(0.5*np.arctan2(URHT[:, v_i], QRHT[:, v_i]), np.pi)
        #theta_RHT_n_v[np.where(IRHTslice <= 0), v_i] = None
        
        I_HI_tot += I_HIsmooth
        QRHT_tot += I_HIsmooth*QRHTsmooth
        URHT_tot += I_HIsmooth*URHTsmooth
        #IRHT_tot += IRHTsmooth
        
    #IHI = np.nansum(HI_n_v, axis=-1)
    #QHI = np.nansum(HI_n_v*np.cos(2*theta_RHT_n_v), axis=-1)
    #UHI = np.nansum(HI_n_v*np.sin(2*theta_RHT_n_v), axis=-1)

    #hp.fitsfunc.write_map("../data/IHI_HI4PI_vels{}_to_{}_IRHTcut_presmooth5{}.fits".format(startvel, stopvel, smoothQUstr), IHI)
    #hp.fitsfunc.write_map("../data/QHI_HI4PI_vels{}_to_{}_IRHTcut_presmooth5{}.fits".format(startvel, stopvel, smoothQUstr), QHI)
    #hp.fitsfunc.write_map("../data/UHI_HI4PI_vels{}_to_{}_IRHTcut_presmooth5{}.fits".format(startvel, stopvel, smoothQUstr), UHI)
    hp.fitsfunc.write_map("../data/IHI_HI4PI_vels{}_to_{}_IRHTcut{}.fits".format(startvel, stopvel, smoothQUstr), I_HI_tot)
    hp.fitsfunc.write_map("../data/QHI_HI4PI_vels{}_to_{}_IRHTcut{}.fits".format(startvel, stopvel, smoothQUstr), QRHT_tot)
    hp.fitsfunc.write_map("../data/UHI_HI4PI_vels{}_to_{}_IRHTcut{}.fits".format(startvel, stopvel, smoothQUstr), URHT_tot)

        
        