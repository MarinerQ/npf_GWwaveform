import logging
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from npf.utils.helpers import rescale_range
import h5py
from npf.neuralproc.base import LatentNeuralProcessFamily
from npf.architectures import CNN, MLP, ResConvBlock, SetConv, discard_ith_arg
from npf import CNPFLoss
from utils.ntbks_helpers import add_y_dim
from utils.train import train_models
from npf import ConvCNP
import time
from functools import partial
import scipy
import bilby
import lal
import lalsimulation

logging.disable(logging.ERROR)
#import warnings
#warnings.filterwarnings("ignore")
#warnings.simplefilter("ignore")


def get_LAL_TDfmin(injection_parameters, fmin_wavegen):
    extra_cycles = 3
    extra_time_fraction = 0.1
    mc = injection_parameters['chirp_mass']
    q = injection_parameters['mass_ratio']
    mtot = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_total_mass(mc,q)
    m1 = mtot/(1+q)*lal.MSUN_SI
    m2 = m1*q
    spin1z = injection_parameters['a_1']*np.cos(injection_parameters['tilt_1'])
    spin2z = injection_parameters['a_2']*np.cos(injection_parameters['tilt_2'])

    tchirp = lalsimulation.SimInspiralChirpTimeBound(fmin_wavegen, m1, m2, spin1z, spin2z)
    s = lalsimulation.SimInspiralFinalBlackHoleSpinBound(spin1z, spin2z)
    tmerge = lalsimulation.SimInspiralMergeTimeBound(m1, m2) +\
            lalsimulation.SimInspiralRingdownTimeBound(m1 + m2, s)
    textra = extra_cycles / fmin_wavegen

    tchirp = (1.0 + extra_time_fraction) * tchirp + tmerge + textra
    fstart = lalsimulation.SimInspiralChirpStartFrequencyBound(tchirp, m1, m2)

    return fstart



def safe_fmin_NRSur7dq4(injection_parameters):
    mc = injection_parameters['chirp_mass']
    q = injection_parameters['mass_ratio']
    mtot = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_total_mass(mc,q)

    theo_fmin = 20 * (66/mtot)
    LAL_fmin = get_LAL_TDfmin(injection_parameters, theo_fmin)

    f_diff = theo_fmin - LAL_fmin
    #print(f"theo_fmin: {theo_fmin}, LAL_fmin: {LAL_fmin}, return {theo_fmin + f_diff}")
    return theo_fmin + f_diff

def my_inner_product(hf1,hf2,det,flag):
    inner_prod_complex = bilby.gw.utils.noise_weighted_inner_product(
                            aa=hf1[det.strain_data.frequency_mask],
                            bb=hf2[det.strain_data.frequency_mask],
                            power_spectral_density=det.power_spectral_density_array[det.strain_data.frequency_mask],
                            duration=det.strain_data.duration)

    #inner_prod_complex = det.inner_product_2(hf1, hf2)
    if flag == "c":
        return inner_prod_complex
    elif flag == "r":
        return np.real(inner_prod_complex)
    else:
        raise("Wrong flag!")


def get_dtdphi_withift_zeropad(h1,h2,det,freq_scale_factor=1,zero_pad_factor=15):

    if det is None:
        duration=8
        f_lower=20
        sampling_frequency=4096
        ifos = bilby.gw.detector.InterferometerList(['L1'])
        det = ifos[0]
        det.duration = duration
        det.sampling_frequency=sampling_frequency
        det.frequency_mask = det.frequency_array>=f_lower
    psd = det.power_spectral_density_array
    f_array = det.frequency_array / freq_scale_factor
    
    X_of_f = h1*h2.conjugate()/psd
    #add_zero = np.zeros(int(63*len(X_of_f)))
    add_zero = np.zeros(int(zero_pad_factor*len(X_of_f)))
    X_of_f = np.append(X_of_f,add_zero)
    X_of_t = np.fft.ifft(X_of_f)
    
    timelength = 1/(f_array[1]-f_array[0])
    t = np.linspace(-timelength/2,timelength/2,len(X_of_t))
    X_shifted = np.roll(X_of_t,len(X_of_t)//2)

    jmax = np.argmax( abs(X_shifted) )
    deltat = t[jmax]
    phase1 = 2*np.pi*f_array*deltat
    
    freq_mask = det.strain_data.frequency_mask
    inner_product = my_inner_product(h1.conjugate(), h2.conjugate()*np.exp(1j*phase1), det, 'c')
    
    '''
    gwutils.noise_weighted_inner_product(
                    aa=h1.conjugate()[freq_mask],
                    bb=(h2.conjugate()*np.exp(1j*phase1))[freq_mask],
                    power_spectral_density=det.power_spectral_density_array[freq_mask],
                    duration=det.strain_data.duration)'''
    
    deltaphi = -np.angle(inner_product)
    #phase2 = deltaphi
    
    return deltat,deltaphi


def get_shifted_h2_zeropad(h1,h2,det,freq_scale_factor=1,zero_pad_factor=15):
    '''
    Return the h2*exp(-i*phase_shift), i.e. h2* exp -i*(2\pi f \Delta t + \Delta \phi)
    '''
    deltat,deltaphi = get_dtdphi_withift_zeropad(h1,h2,det,freq_scale_factor,zero_pad_factor)
    f_array = det.frequency_array
    exp_phase = np.exp(-1j*(2*np.pi*f_array*deltat + deltaphi) )
    return h2*exp_phase


# The following 4 hdf saving/reading functions are from
# https://codereview.stackexchange.com/questions/120802/recursively-save-python-dictionaries-to-hdf5-files-using-h5py
def save_dict_to_hdf5(dic, filename):
    """
    
    """
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)

def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    
    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes, list)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s type'%type(item))

def load_dict_from_hdf5(filename):
    """
    
    """
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

def recursively_load_dict_contents_from_group(h5file, path):
    """
    
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans


def amp_scaling(f, chirp_mass):
    '''
    Eq.4.34 of GW, for 1Mpc. Mc in unit of Msun
    '''
    c=299792458
    G=6.67e-11
    A=(np.pi)**(-2/3)*np.sqrt(5/24)

    amp = A*c*(G*chirp_mass/c**3)**(5/6) * f**(-7/6)
    
    return amp

def freq_scaling(f, chirp_mass):
    '''
    Eq.4.37 of GW
    '''
    #G=6.67e-11
    #c=299792458
    #A = 3/4 * (8*np.pi*G*chirp_mass/c**3)**(-5/3)
    return f**(-5/3) * 1e3
    
    
def scale_aligned_fdwaveforms(farray, h_list, chirp_mass):
    amp = amp_scaling(farray, chirp_mass)
    
    amp_scaled_h_list = []
    for h in h_list:
        amp_scaled_h_list.append(h / amp)
    
    scaled_f = freq_scaling(farray, chirp_mass)
    return scaled_f, amp_scaled_h_list

def unscale_scaled_fdwaveforms(farray_scaled, h_list_scaled, chirp_mass):
    '''
    Transform back to original waveforms.
    '''
    farray = (farray_scaled/1e3) ** (-3/5)
    
    amp = amp_scaling(farray, chirp_mass)
    
    h_list_unscaled = []
    for h in h_list_scaled:
        h_list_unscaled.append(h*amp)
    
    return farray, h_list_unscaled

def resample_scaled_fdwaveforms(farray_scaled, h_list_scaled):
    farray_scaled = farray_scaled[::-1]
    for i,h in enumerate(h_list_scaled):
        h_list_scaled[i] = h[::-1]
    
    fs_min = min(farray_scaled)
    fs_max = max(farray_scaled)
    new_fs = np.linspace(fs_min, fs_max, len(farray_scaled))
    
    h_list_scaled_resampled = []
    for h in h_list_scaled:
        interpolator = scipy.interpolate.CubicSpline(farray_scaled, h)
        new_h = interpolator(new_fs)
        h_list_scaled_resampled.append(new_h[::-1])
    
    return new_fs[::-1], h_list_scaled_resampled



def get_gwh5_nsample(gwh5file):
    with h5py.File(gwh5file, "r") as f:
        try:
            l = len( list(f['time_stretched']) )
        except:
            l = len( list(f['time']) )
    return l

def get_gwfdh5_nsample(gwh5file):
    with h5py.File(gwh5file, "r") as f:
        approx = list(f['waveform_fd'])[0]
        l = len(list(f['waveform_fd'][approx]['plus']['scaled_resampled']))
    return l