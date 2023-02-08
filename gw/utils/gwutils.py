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

from functools import partial
import scipy
import bilby

logging.disable(logging.ERROR)
#import warnings
#warnings.filterwarnings("ignore")
#warnings.simplefilter("ignore")


def my_inner_product(hf1,hf2,det,flag):
    inner_prod_complex = gwutils.noise_weighted_inner_product(
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


def get_dtdphi_withift_zeropad(h1,h2,det):

    psd = det.power_spectral_density_array
    f_array = det.frequency_array
    
    X_of_f = h1*h2.conjugate()/psd
    add_zero = np.zeros(int(63*len(X_of_f)))
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


def get_shifted_h2_zeropad(h1,h2,det):
    '''
    Return the h2*exp(-i*phase_shift), i.e. h2* exp -i*(2\pi f \Delta t + \Delta \phi)
    '''
    deltat,deltaphi = get_dtdphi_withift_zeropad(h1,h2,det)
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

def get_predictions(model, x_context, y_context, x_target, nsample, return_samples = False):
    ''' 
    from visualize_1d.gen_p_y_pred and visualize_1d._plot_posterior_predefined_cntxt

    possible mistake here
    '''
    model.eval()
    if isinstance(model, LatentNeuralProcessFamily):
        old_n_z_samples_test = model.n_z_samples_test
        model.n_z_samples_test = nsample
        p_yCc, *_ = model.forward(x_context, y_context, x_target)
        model.n_z_samples_test = old_n_z_samples_test

        mean_ys = p_yCc.base_dist.loc.detach().numpy()
        std_ys = p_yCc.base_dist.scale.detach().numpy()

        mean_return = np.mean( mean_ys, axis=0)
        std_return = np.mean( std_ys, axis=0)

        if return_samples:
            return mean_return, std_return, mean_ys
        else:
            return mean_return, std_return
    else:
        p_yCc, *_ = model.forward(x_context, y_context, x_target)
        sampled_y = p_yCc.sample_n(nsample).detach().numpy()
        mean_ys = p_yCc.base_dist.loc.detach().squeeze().numpy()
        std_ys = p_yCc.base_dist.scale.detach().squeeze().numpy()

        mean_return = mean_ys
        std_return = std_ys 

        if return_samples:
            return mean_return, std_return, sampled_y
        else:
            return mean_return, std_return


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

def get_trained_gwmodels(path,device):
    R_DIM = 128
    KWARGS = dict(
        r_dim=R_DIM,
        Decoder=discard_ith_arg(  # disregards the target features to be translation equivariant
            partial(MLP, n_hidden_layers=4, hidden_size=R_DIM), i=0
        ),
    )


    CNN_KWARGS = dict(
        ConvBlock=ResConvBlock,
        is_chan_last=True,  # all computations are done with channel last in our code
        n_conv_layers=2,  # layers per block
    )


    # off the grid
    model_1d = partial(
        ConvCNP,
        x_dim=1,
        y_dim=1,
        Interpolator=SetConv,
        CNN=partial(
            CNN,
            Conv=torch.nn.Conv1d,
            Normalization=torch.nn.BatchNorm1d,
            n_blocks=5,
            kernel_size=19,
            **CNN_KWARGS,
        ),
        density_induced=64,  # density of discretization
        **KWARGS,
    )


    # FULLFD_IMREOB_PHM_q25a8M40_2N10k
    KWARGS = dict(
        is_retrain=False,  # whether to load precomputed model or retrain
        is_continue_train=False,
        criterion=CNPFLoss,
        chckpnt_dirname=path,
        device=device,
        batch_size=1
    )

    # 1D
    trainers_1d = train_models(
        datasets={'plus_real':[],'plus_imag':[],'cross_real':[],'cross_imag':[] },
        models={"ConvCNP": model_1d},
        test_datasets={'plus_real':[],'plus_imag':[],'cross_real':[],'cross_imag':[] },
        **KWARGS
    )

    model_dict = {}
    for key in ['plus_real', 'plus_imag', 'cross_real', 'cross_imag']:
        model_dict[key] = trainers_1d[f'{key}/ConvCNP/run_0'].module_

    return model_dict


class GWDatasetFDMultimodel(Dataset):
    def __init__(self, h5file,
                root_dir=None, indcies=None,
                mode='plus', part='real', wavetype='scaled_resampled',
                models='all'):
        """
        Gravitational wave dataset. Waveforms are in FD.


        Parameters:
        h5file: Path to waveform h5 filename.

        indcies: Indices of waveforms you want to select. E.g. a h5 file contains 1000 waveforms, you
        may want to select 700 of them for training, 2*150 for testing and validating.

        mode: 'plus' (default) or 'cross'.

        part: 'real' (default) or 'imag'.

        wavetype: 'scaled_resampled'

        models: 'all' (default) or list of waveform models you want to use, e.g. ['IMRPhenomPv2'].
        """
        self.h5file = h5file
        self.root_dir = root_dir
        self.n_samples = get_gwfdh5_nsample(h5file)
        if indcies is None:
            indcies = np.arange(self.n_samples)
        else:
            self.n_samples = len(indcies)
        self.mode = mode
        self.part = part
        self.wavetype = wavetype

        self.fdwaveforms = dict()
        with h5py.File(h5file, "r") as f:
            if models=='all':
                self.waveform_models = np.array(list(f['waveform_fd']))
                self.n_models = len(self.waveform_models)
            else:
                self.waveform_models = []
                for m in models:
                    self.waveform_models.append(m)
                self.waveform_models = np.array(self.waveform_models)
                self.n_models = len(self.waveform_models)
            self.frequency_array = np.array(f['frequency'][f'frequency_array_{wavetype}'])
            
            for approx in self.waveform_models:
                if part=='real':
                    self.fdwaveforms[approx] = np.real(np.array(list(f['waveform_fd'][approx][mode][wavetype])))[indcies]
                elif part=='imag':
                    self.fdwaveforms[approx] = np.imag(np.array(list(f['waveform_fd'][approx][mode][wavetype])))[indcies]
                else:
                    raise Exception(f'Wrong part {part}!')

            self.source_parameter_names = np.array(list(f['source_parameters']))
            self.source_parameters = {}
            for name in self.source_parameter_names:
                self.source_parameters[name] = np.array(list(f['source_parameters'][name]))[indcies]
        
        
        
    def __len__(self):
        return self.n_samples
    
    def remove_nan(self, array):
        return array[~np.isnan(array)]
    
    def __getitem__(self, index):
        f = np.array([])
        h = np.array([])
        for approx in self.waveform_models:
            f = np.append(f, self.frequency_array)
            h = np.append(h, self.fdwaveforms[approx][index])

        arg_sorted = np.argsort(f)
        f = f[arg_sorted]
        h = h[arg_sorted] #/ 100

        f = rescale_range(f, (f.min(),f.max()), (-1,1))
        return torch.from_numpy(f).unsqueeze(-1).type(torch.float32), torch.from_numpy(h).unsqueeze(-1).type(torch.float32)

    
    def get_specific_model(self,index, approx, rescalex=True):
        f = self.frequency_array
        h = self.fdwaveforms[approx][index] #/ 100

        if rescalex:
            f = rescale_range(f, (f.min(),f.max()), (-1,1))

        return torch.from_numpy(f).unsqueeze(-1).type(torch.float32), torch.from_numpy(h).unsqueeze(-1).type(torch.float32)


class NPWaveformGenerator():
    def __init__(self, model_path,
                context_waveform_generator,
                device='cpu',
                npmodel_total_mass=40, 
                npmodel_f_low=20,
                npmodel_f_ref = 50, 
                npmodel_duration = 32,
                npmodel_sampling_frequency = 8192
                ):
        npmodel_dict = get_trained_gwmodels(model_path,device=device)
        self.npmodel_dict = npmodel_dict
        self.npmodel_total_mass = npmodel_total_mass
        self.npmodel_f_low = npmodel_f_low
        self.npmodel_f_ref = npmodel_f_ref
        self.npmodel_duration = npmodel_duration
        self.npmodel_sampling_frequency = npmodel_sampling_frequency
        #self.npmodel_frequency_array =\
        #    np.linspace(0, npmodel_sampling_frequency/2, npmodel_sampling_frequency//2 * duration + 1)

        self.context_waveform_generator = context_waveform_generator
        self.context_duration = context_waveform_generator.duration
        self.context_sampling_frequency = context_waveform_generator.sampling_frequency
        self.context_f_low = context_waveform_generator.waveform_arguments['minimum_frequency']
        self.context_f_ref = context_waveform_generator.waveform_arguments['reference_frequency']
        self.context_frequency_array = context_waveform_generator.frequency_array
        self.context_frequency_mask = context_waveform_generator.frequency_array>=self.context_f_low
        
    
    def scale_waveform(self, target_mass, base_f, base_h, base_mass):
        mratio = target_mass/base_mass
        target_f = base_f / mratio
        target_h = base_h * mratio**2

        return target_f, target_h

    def monochromatize_waveform(self, farray, harray, chirp_mass):
        c=299792458
        G=6.67e-11
        A=(np.pi)**(-2/3)*np.sqrt(5/24)
        amp = A*c*(G*chirp_mass/c**3)**(5/6) * farray**(-7/6)
        h_mono = harray / amp
        f_mono = farray**(-5/3) * 1e3

        return f_mono, h_mono
    
    def unmonochromatize_waveform(self, f_mono, h_mono, chirp_mass):
        c=299792458
        G=6.67e-11
        A=(np.pi)**(-2/3)*np.sqrt(5/24)
        farray = (f_mono/1e3)**(-3/5)
        amp = A*c*(G*chirp_mass/c**3)**(5/6) * farray**(-7/6)
        harray = h_mono * amp

        return farray, harray

    def resample_mono_fdwaveforms(self, f_mono, h_mono):
        f_mono_new = f_mono[::-1]
        h_mono_new = h_mono[::-1]
 
        fs_min = min(f_mono_new)
        fs_max = max(f_mono_new)
        
        fs_rd = 0.1 # for M=25, ringfown freq=0.015. Take 0.02 here
        new_fs1 = np.linspace(fs_min, fs_rd, int(len(f_mono_new)*fs_rd/(fs_max-fs_min)) )
        new_fs2 = np.linspace(fs_rd, fs_max, len(f_mono_new)//10)
        new_fs = np.append(new_fs1,new_fs2[1:])
        
        interpolator = scipy.interpolate.CubicSpline(f_mono_new, h_mono_new)
        new_h = interpolator(new_fs)
        
        return new_fs[::-1], new_h[::-1]
    
    def unresample_mono_fdwaveforms(self, f_mono_resampled, h_mono_resampled, f_mono):
        interpolator = scipy.interpolate.CubicSpline(f_mono_resampled[::-1], h_mono_resampled[::-1])
        h_mono = interpolator(f_mono)
        return h_mono

        
    def frequency_domain_strain(self, parameters):
        mtot = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_total_mass(
            parameters['chirp_mass'],
            parameters['mass_ratio']
        )
        mratio = mtot/self.npmodel_total_mass

        # Changing f_ref is effectively changing inclination and rotating spins, however our training includes all inclinations and spins. Therefore I am NOT going to scale f_ref and commented this out.
        #self.context_waveform_generator.waveform_arguments['reference_frequency'] /= mratio

        # However, keep in mind that our model learns weaveforms from 40Msun, f_low=20Hz - these two parameters do not span over the whole space. We have to scale them. 
        self.context_waveform_generator.waveform_arguments['minimum_frequency'] /= mratio
        fmin4inj = self.context_waveform_generator.waveform_arguments['minimum_frequency']
        if fmin4inj>self.context_f_low: 
            logging.warning(f"Mtot={mtot} < Training Mtot {self.npmodel_total_mass}, therefore changing starting frequency to {fmin4inj}Hz, which is higher than given {self.context_f_low}Hz.")

        mask = self.context_frequency_array>=fmin4inj
        d_l = parameters['luminosity_distance']
        parameters['luminosity_distance'] = 100
        h_dict_context = self.context_waveform_generator.frequency_domain_strain(parameters)
        h_dict_context_components={}
        h_dict_context_components['plus_real'] = np.real(h_dict_context['plus'])[mask]
        h_dict_context_components['plus_imag'] = np.imag(h_dict_context['plus'])[mask]
        h_dict_context_components['cross_real'] = np.real(h_dict_context['cross'])[mask]
        h_dict_context_components['cross_imag'] = np.imag(h_dict_context['cross'])[mask]
        farray = self.context_frequency_array[mask]

        mean_dict={}
        std_dict={}
        for label,harray in h_dict_context_components.items():
            scaled_context_f, scaled_context_h = self.scale_waveform(target_mass=self.npmodel_total_mass, base_f=farray, base_h=harray, base_mass=mtot)
            f_mono, h_mono = self.monochromatize_waveform(scaled_context_f, scaled_context_h, parameters['chirp_mass']/mratio)
            f_mono_resampled, h_mono_resampled = self.resample_mono_fdwaveforms(f_mono, h_mono)

            model = self.npmodel_dict[label]
            mean_resampled, std_resampled = get_predictions(model,
                                                torch.from_numpy(f_mono_resampled[::-1]).unsqueeze(-1).unsqueeze(-1).type(torch.float32), 
                                                torch.from_numpy(h_mono_resampled[::-1]).unsqueeze(-1).unsqueeze(-1).type(torch.float32), 
                                                torch.from_numpy(f_mono_resampled[::-1]).unsqueeze(-1).unsqueeze(-1).type(torch.float32), 
                                                1)
            mean_resampled = mean_resampled[::-1]
            std_resampled = std_resampled[::-1]
            mean_mono = self.unresample_mono_fdwaveforms(f_mono_resampled, mean_resampled, f_mono)
            std_mono = self.unresample_mono_fdwaveforms(f_mono_resampled, std_resampled, f_mono)

            farray2, mean_scaled = self.unmonochromatize_waveform(f_mono, mean_mono, parameters['chirp_mass']/mratio)
            farray2, std_scaled = self.unmonochromatize_waveform(f_mono, std_mono, parameters['chirp_mass']/mratio)

            # unscale to injection mass
            # farray3 should == farray (?)
            farray3, mean = self.scale_waveform(target_mass=mtot, base_f=scaled_context_f, base_h=mean_scaled, base_mass=self.npmodel_total_mass)
            farray3, std = self.scale_waveform(target_mass=mtot, base_f=scaled_context_f, base_h=std_scaled, base_mass=self.npmodel_total_mass)

            zerolenth = len(np.where(self.context_frequency_array<min(fmin4inj,self.context_f_low))[0])
            zero_paddings = np.zeros(zerolenth)
            
            mean_dict[label] = np.append(zero_paddings, mean) * 100 / d_l
            std_dict[label] = np.append(zero_paddings, std) * 100 / d_l
        
        h_dict = {}
        error_dict = {}
        for key in ['plus', 'cross']:
            h_dict[key] = mean_dict[f'{key}_real'] + mean_dict[f'{key}_imag']*1j
            error_dict[key] = std_dict[f'{key}_real'] + std_dict[f'{key}_imag']*1j

        #self.context_waveform_generator.waveform_arguments['reference_frequency'] *= mratio
        self.context_waveform_generator.waveform_arguments['minimum_frequency'] *= mratio
        parameters['luminosity_distance']=d_l
        return h_dict, error_dict


class GWDatasetFDMultimodelNDOutput(Dataset):
    def __init__(self, h5file,
                root_dir=None, indcies=None,
                mode=['plus'], part=['real'], wavetype='scaled_resampled',
                models='all'):
        """
        Gravitational wave dataset. Waveforms are in FD.


        Parameters:
        h5file: Path to waveform h5 filename.

        indcies: Indices of waveforms you want to select. E.g. a h5 file contains 1000 waveforms, you
        may want to select 700 of them for training, 2*150 for testing and validating.

        mode: 'plus' (default) or 'cross'.

        part: 'real' (default) or 'imag'.

        wavetype: 'scaled_resampled'

        models: 'all' (default) or list of waveform models you want to use, e.g. ['IMRPhenomPv2'].
        """
        self.h5file = h5file
        self.root_dir = root_dir
        self.n_samples = get_gwfdh5_nsample(h5file)
        if indcies is None:
            indcies = np.arange(self.n_samples)
        else:
            self.n_samples = len(indcies)
        self.mode = mode
        self.part = part
        self.wavetype = wavetype

        #self.fdwaveforms = dict()
        self.fdwaveforms_all = dict()

        with h5py.File(h5file, "r") as f:
            if models=='all':
                self.waveform_models = np.array(list(f['waveform_fd']))
                self.n_models = len(self.waveform_models)
            else:
                self.waveform_models = []
                for m in models:
                    self.waveform_models.append(m)
                    
                self.waveform_models = np.array(self.waveform_models)
                self.n_models = len(self.waveform_models)
            self.frequency_array = np.array(f['frequency'][f'frequency_array_{wavetype}'])
            
            
            for approx in self.waveform_models:
                self.fdwaveforms_all[approx] = dict()
                self.fdwaveforms_all[approx]['plus'] = dict()
                self.fdwaveforms_all[approx]['cross'] = dict()
                self.fdwaveforms_all[approx]['plus']['real'] = np.real(np.array(list(f['waveform_fd'][approx]['plus'][wavetype])))[indcies]
                self.fdwaveforms_all[approx]['plus']['imag'] = np.imag(np.array(list(f['waveform_fd'][approx]['plus'][wavetype])))[indcies]
                self.fdwaveforms_all[approx]['cross']['real'] = np.real(np.array(list(f['waveform_fd'][approx]['cross'][wavetype])))[indcies]
                self.fdwaveforms_all[approx]['cross']['imag'] = np.imag(np.array(list(f['waveform_fd'][approx]['cross'][wavetype])))[indcies]
                
                '''
                if part=='real':
                    self.fdwaveforms[approx] = np.real(np.array(list(f['waveform_fd'][approx][mode][wavetype])))[indcies]
                elif part=='imag':
                    self.fdwaveforms[approx] = np.imag(np.array(list(f['waveform_fd'][approx][mode][wavetype])))[indcies]
                else:
                    raise Exception(f'Wrong part {part}!')
                '''
            self.source_parameter_names = np.array(list(f['source_parameters']))
            self.source_parameters = {}
            for name in self.source_parameter_names:
                self.source_parameters[name] = np.array(list(f['source_parameters'][name]))[indcies]
        
        
        
    def __len__(self):
        return self.n_samples
    
    def remove_nan(self, array):
        return array[~np.isnan(array)]
    
    def __getitem__(self, index):
        f = np.array([])
        h1 = np.array([])
        h2 = np.array([])
        for approx in self.waveform_models:
            f = np.append(f, self.frequency_array)
            h1 = np.append(h1, self.fdwaveforms_all[approx]['plus']['real'][index])
            h2 = np.append(h2, self.fdwaveforms_all[approx]['cross']['imag'][index])

        arg_sorted = np.argsort(f)
        f = f[arg_sorted]
        h1 = h1[arg_sorted] / 100
        h2 = h2[arg_sorted] / 100
        h = np.array([h1, h2]).T
        f = rescale_range(f, (f.min(),f.max()), (-1,1))
        return torch.from_numpy(f).unsqueeze(-1).type(torch.float32), torch.from_numpy(h).type(torch.float32)

    
    def get_specific_model(self,index, approx, rescalex=True):
        f = self.frequency_array
        h1 = self.fdwaveforms_all[approx]['plus']['real'][index] 
        h2 = self.fdwaveforms_all[approx]['cross']['imag'][index]
        h1 = h1 / 100
        h2 = h2 / 100
        h = np.array([h1, h2]).T

        if rescalex:
            f = rescale_range(f, (f.min(),f.max()), (-1,1))

        return torch.from_numpy(f).unsqueeze(-1).type(torch.float32), torch.from_numpy(h).type(torch.float32)

'''
class GWDataset(Dataset):
    def __init__(self, h5file,
                 root_dir=None, indcies=None):
        """
        Gravitational wave dataset.
        """
        self.h5file = h5file
        self.root_dir = root_dir
        self.n_samples = get_gwh5_nsample(h5file)
        if indcies is None:
            indcies = np.arange(self.n_samples)
            
        with h5py.File(h5file, "r") as f:
            self.time = np.array(list(f['time']))[indcies]
            self.strain = np.array(list(f['strain']))[indcies]

            self.source_parameter_names = np.array(list(f['source_parameters']))
            self.source_parameters = {}
            for name in self.source_parameter_names:
                self.source_parameters[name] = np.array(list(f['source_parameters'][name]))[indcies]
        
        self.n_samples = len(self.time)
        
        
    def __len__(self):
        return self.n_samples
    
    def remove_nan(self, array):
        return array[~np.isnan(array)]
    
    def __getitem__(self, index):
        t = self.time[index]
        h = self.strain[index]
        
        t = self.remove_nan(t)
        h = self.remove_nan(h)

        t = rescale_range(t, (t.min(),t.max()), (-1,1))

        return torch.from_numpy(t).unsqueeze(-1).type(torch.float32), torch.from_numpy(h).unsqueeze(-1).type(torch.float32)
    



class GWDatasetCut(Dataset):
    def __init__(self, h5file, cut_fraction,
                 root_dir=None, indcies=None):
        """
        Gravitational wave dataset. 
        """
        self.h5file = h5file
        self.root_dir = root_dir
        self.n_samples = get_gwh5_nsample(h5file)
        if indcies is None:
            indcies = np.arange(self.n_samples)
            
        with h5py.File(h5file, "r") as f:
            self.time = np.array(list(f['time']))[indcies]
            self.strain = np.array(list(f['strain']))[indcies]

            self.source_parameter_names = np.array(list(f['source_parameters']))
            self.source_parameters = {}
            for name in self.source_parameter_names:
                self.source_parameters[name] = np.array(list(f['source_parameters'][name]))[indcies]
        
        self.n_samples = len(self.time)
        self.cut_fraction = cut_fraction
        
    def __len__(self):
        return self.n_samples
    
    def remove_nan(self, array):
        return array[~np.isnan(array)]
    
    def __getitem__(self, index):
        t = self.time[index]
        h = self.strain[index]
        
        t = self.remove_nan(t)
        h = self.remove_nan(h)
        ll=len(t)

        cut1 = int(ll*self.cut_fraction[0])
        cut2 = int(ll*self.cut_fraction[1])
        # cut at both direction
        t=t[cut1:-cut2]
        h=h[cut1:-cut2]

        t = rescale_range(t, (t.min(),t.max()), (-1,1))
        return torch.from_numpy(t).unsqueeze(-1).type(torch.float32), torch.from_numpy(h).unsqueeze(-1).type(torch.float32)


class GWDatasetCutStRe(Dataset):
    def __init__(self, h5file, cut_fraction,
                 root_dir=None, indcies=None):
        """
        Gravitational wave dataset. Waveforms are cut, stretched, and resampled.
        """
        self.h5file = h5file
        self.root_dir = root_dir
        self.n_samples = get_gwh5_nsample(h5file)
        if indcies is None:
            indcies = np.arange(self.n_samples)
            
        with h5py.File(h5file, "r") as f:
            self.time_stretched = np.array(list(f['time_stretched']))[indcies]
            self.strain_stretched = np.array(list(f['strain_stretched']))[indcies]
            self.time_stretched_resampled = np.array(list(f['time_stretched_resampled']))[indcies]
            self.strain_stretched_resampled = np.array(list(f['strain_stretched_resampled']))[indcies]

            self.source_parameter_names = np.array(list(f['source_parameters']))
            self.source_parameters = {}
            for name in self.source_parameter_names:
                self.source_parameters[name] = np.array(list(f['source_parameters'][name]))[indcies]
        
        self.n_samples = len(self.time_stretched)
        self.cut_fraction = cut_fraction
        
    def __len__(self):
        return self.n_samples
    
    def remove_nan(self, array):
        return array[~np.isnan(array)]
    
    def __getitem__(self, index):
        t = self.time_stretched_resampled[index]
        h = self.strain_stretched_resampled[index]
        
        t = self.remove_nan(t)
        h = self.remove_nan(h)
        ll=len(t)

        cut1 = int(ll*self.cut_fraction[0])
        cut2 = int(ll*self.cut_fraction[1])
        # cut at both direction
        t=t[cut1:-cut2]
        h=h[cut1:-cut2]

        t = rescale_range(t, (t.min(),t.max()), (-1,1))
        return torch.from_numpy(t).unsqueeze(-1).type(torch.float32), torch.from_numpy(h).unsqueeze(-1).type(torch.float32)



class GWDatasetCutStReMultimodel(Dataset):
    def __init__(self, h5file, cut_fraction,
                 root_dir=None, indcies=None):
        """
        Gravitational wave dataset. Waveforms are cut, stretched, and resampled.
        """
        self.h5file = h5file
        self.root_dir = root_dir
        self.n_samples = get_gwh5_nsample(h5file)
        if indcies is None:
            indcies = np.arange(self.n_samples)
            
        with h5py.File(h5file, "r") as f:
            self.waveform_models = np.array(list(f['waveform_models']))
            self.time_stretched = np.array(list(f['time_stretched']))[indcies]
            self.n_samples = len(self.time_stretched)
            self.n_models = len(self.waveform_models)
            self.n_physamples = self.n_samples // self.n_models

            self.strain_stretched = np.array(list(f['strain_stretched']))[indcies]
            self.time_stretched_resampled = np.array(list(f['time_stretched_resampled']))[indcies]
            self.strain_stretched_resampled = np.array(list(f['strain_stretched_resampled']))[indcies]

            self.source_parameter_names = np.array(list(f['source_parameters']))
            self.source_parameters = {}
            indcies_for_sourcepara = indcies // self.n_models
            for name in self.source_parameter_names:
                self.source_parameters[name] = np.array(list(f['source_parameters'][name]))[indcies_for_sourcepara]
        
        
        self.cut_fraction = cut_fraction
        
    def __len__(self):
        return self.n_samples
    
    def remove_nan(self, array):
        return array[~np.isnan(array)]
    
    def __getitem__(self, index):
        t = self.time_stretched_resampled[index]
        h = self.strain_stretched_resampled[index]
        
        t = self.remove_nan(t)
        h = self.remove_nan(h)
        ll=len(t)

        cut1 = int(ll*self.cut_fraction[0])
        cut2 = int(ll*self.cut_fraction[1])
        # cut at both direction
        t=t[cut1:-cut2]
        h=h[cut1:-cut2]

        t = rescale_range(t, (t.min(),t.max()), (-1,1))
        return torch.from_numpy(t).unsqueeze(-1).type(torch.float32), torch.from_numpy(h).unsqueeze(-1).type(torch.float32)
    
    def get_unresampled_item(self, index):
        t = self.time_stretched[index]
        h = self.strain_stretched[index]
        
        t = self.remove_nan(t)
        h = self.remove_nan(h)
        ll=len(t)

        cut1 = int(ll*self.cut_fraction[0])
        cut2 = int(ll*self.cut_fraction[1])
        # cut at both direction
        t=t[cut1:-cut2]
        h=h[cut1:-cut2]

        t = rescale_range(t, (t.min(),t.max()), (-1,1))
        return torch.from_numpy(t).unsqueeze(-1).type(torch.float32), torch.from_numpy(h).unsqueeze(-1).type(torch.float32)
    
    def get_specific_model(self, model_index, data_index_in_model):
        data_index_in_dataset = int(self.n_models * data_index_in_model + model_index)
        return self[data_index_in_dataset]

    def get_specific_model_unresampled(self, model_index, data_index_in_model):
        data_index_in_dataset = int(self.n_models * data_index_in_model + model_index)
        return self.get_unresampled_item(data_index_in_dataset)

'''