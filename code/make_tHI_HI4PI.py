import numpy as np
import healpy as hp

import sys 
sys.path.insert(0, '../../ForegroundModels/code')
from make_data_table import get_RHT_data, get_list_fns, get_start_stop_hpx
sys.path.insert(0, '../../RHT')
import RHT_tools

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
    
    # Values of theta for RHT output
    thets = get_thets(wlen, save = False)
    
    all_vels = np.arange(81, 102)
    nvels = len(all_vels)
    startvel = all_vels[0]
    stopvel = all_vels[-1]
    
    QRHT = np.zeros((npix, nvels), np.float_)
    URHT = np.zeros((npix, nvels), np.float_)
    HI_n_v = np.zeros((npix, nvels), np.float_)
    
    
    # step through all velocities
    for v_i, _vel in enumerate(all_vels):
        
        # load HI intensity data 
        hi4pi_fn = hi4pi_root + 'HI4PI_120kms.h5'
        with h5py.File(hi4pi_fn, 'r') as f:
            HI_n_v[:, v_i] = f['survey'][:, _vel]
        
        # get all filenames for given velocity
        fns = get_list_fns(_vel, gal=True, data_root=data_root)
        
        for fn in fns:
            hpix, hthets = get_RHT_data(data_root+fn, returnbp=False)
        
            QRHT[hpix, _vel] = np.nansum(np.cos(2*thets)*hthets)
            URHT[hpix, _vel] = np.nansum(np.sin(2*thets)*hthets)
    
    theta_RHT_n_v = np.mod(0.5*np.arctan2(URHT, QRHT), np.pi)
    
    IHI = np.nansum(HI_n_v, axis=-1)
    QHI = np.nansum(HI_n_v*np.cos(2*theta_RHT_n_v), axis=-1)
    UHI = np.nansum(HI_n_v*np.sin(2*theta_RHT_n_v), axis=-1)

    hp.fitsfunc.write_map("../data/IHI_HI4PI_vels{}_to_{}.fits".format(startvel, stopvel), IHI)
    hp.fitsfunc.write_map("../data/QHI_HI4PI_vels{}_to_{}.fits".format(startvel, stopvel), QHI)
    hp.fitsfunc.write_map("../data/UHI_HI4PI_vels{}_to_{}.fits".format(startvel, stopvel), UHI)

        
        