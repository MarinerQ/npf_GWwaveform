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

from .gwutils import *

logging.disable(logging.ERROR)

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


def get_trained_gwmodels(path,device,R_DIM=128):
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
        self.context_waveform_generator.waveform_arguments['reference_frequency'] /= mratio

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

        self.context_waveform_generator.waveform_arguments['reference_frequency'] *= mratio
        self.context_waveform_generator.waveform_arguments['minimum_frequency'] *= mratio
        parameters['luminosity_distance']=d_l
        return h_dict, error_dict


class NPMixWaveformGenerator():
    def __init__(self, model_path,
                context_waveform_generator1,
                context_waveform_generator2,
                context_fraction = 0.3,
                device='cpu',
                R_DIM=128,
                npmodel_total_mass=40, 
                npmodel_f_low=20,
                npmodel_f_ref = 50, 
                npmodel_duration = 32,
                npmodel_sampling_frequency = 8192,
                example_det = None
                ):

        self.context_fraction = context_fraction
        self.R_DIM = R_DIM
        npmodel_dict = get_trained_gwmodels(model_path,device=device,R_DIM=R_DIM)
        self.npmodel_dict = npmodel_dict
        self.npmodel_total_mass = npmodel_total_mass
        self.npmodel_f_low = npmodel_f_low
        self.npmodel_f_ref = npmodel_f_ref
        self.npmodel_duration = npmodel_duration
        self.npmodel_sampling_frequency = npmodel_sampling_frequency
        #self.npmodel_frequency_array =\
        #    np.linspace(0, npmodel_sampling_frequency/2, npmodel_sampling_frequency//2 * duration + 1)

        self.context_waveform_generator1 = context_waveform_generator1
        self.context_waveform_generator2 = context_waveform_generator2
        self.context_duration = context_waveform_generator1.duration
        self.context_sampling_frequency = context_waveform_generator1.sampling_frequency
        self.context_f_low = context_waveform_generator1.waveform_arguments['minimum_frequency']
        self.context_f_ref = context_waveform_generator1.waveform_arguments['reference_frequency']
        self.context_frequency_array = context_waveform_generator1.frequency_array
        self.context_frequency_mask = context_waveform_generator1.frequency_array>=self.context_f_low

        if example_det is None:
            duration=self.context_duration
            f_lower=self.context_f_low
            sampling_frequency=self.context_sampling_frequency
            ifos = bilby.gw.detector.InterferometerList(['L1'])
            self.example_det = ifos[0]
            self.example_det.duration = duration
            self.example_det.sampling_frequency=sampling_frequency
            self.example_det.frequency_mask = self.example_det.frequency_array>=f_lower
        else:
            self.example_det = example_det
        
    
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
        #return new_fs, new_h
    
    def unresample_mono_fdwaveforms(self, f_mono_resampled, h_mono_resampled, f_mono):
        interpolator = scipy.interpolate.CubicSpline(f_mono_resampled[::-1], h_mono_resampled[::-1])
        h_mono = interpolator(f_mono)
        return h_mono


    def mix_waveforms(self, farray, h1_dict, h2_dict, mode, part, fraction=0.01):
        #length = len(h1_dict['plus'][mask])
        #index1 = np.random.permutation(length)[:int(length*weight)]
        #index2 = np.random.permutation(length)[:length-int(length*weight)]
        #index1 = np.arange(length)[::2]
        #index2 = np.arange(length)[1::2]
        #argsort = np.argsort(np.append(index1, index2))

        l = len(farray)
        
        farray_m1to1 = rescale_range(farray, (farray.min(),farray.max()), (-1,1))
        ctxtindex = np.arange(0,l,int(1/fraction))
        ffarray = np.array(list(zip(farray_m1to1[ctxtindex], farray_m1to1[ctxtindex]))).flatten()
        if part == 'real':
            return ffarray, np.real(np.array(list(zip(h1_dict[mode][ctxtindex], h2_dict[mode][ctxtindex]))).flatten())
        elif part == 'imag':
            return ffarray, np.imag(np.array(list(zip(h1_dict[mode][ctxtindex], h2_dict[mode][ctxtindex]))).flatten())

    def frequency_domain_strain(self, parameters):
        mtot = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_total_mass(
            parameters['chirp_mass'],
            parameters['mass_ratio']
        )
        mratio = mtot/self.npmodel_total_mass
        self.context_waveform_generator1.waveform_arguments['reference_frequency'] *= mratio
        self.context_waveform_generator2.waveform_arguments['reference_frequency'] *= mratio

        # However, keep in mind that our model learns weaveforms from 40Msun, f_low=20Hz - these two parameters do not span over the whole space. We have to scale them. 
        self.context_waveform_generator1.waveform_arguments['minimum_frequency'] = self.npmodel_f_low
        self.context_waveform_generator2.waveform_arguments['minimum_frequency'] = self.npmodel_f_low

        if mratio<1 and self.context_f_low<self.npmodel_f_low/mratio: 
            #logging.warning(f"Mtot={mtot} < Training Mtot {self.npmodel_total_mass}, starting frequency is at least {self.npmodel_f_low/mratio}Hz, which is higher than given {self.context_f_low}Hz. Setting zeros between two frequencies.")
            print(f"Warning: Mtot={mtot} < Training Mtot {self.npmodel_total_mass}, starting frequency should be at least {self.npmodel_f_low/mratio}Hz, which is higher than given {self.context_f_low}Hz. Setting zeros between two frequencies.")

        mask = self.context_frequency_array>=self.npmodel_f_low
        mask_output = self.context_frequency_array>=max(self.context_f_low,self.npmodel_f_low/mratio)
        d_l = parameters['luminosity_distance']
        parameters['luminosity_distance'] = 100
        parameters['chirp_mass'] = parameters['chirp_mass']/mratio

        h_dict_context1 = {}
        h_dict_context2 = {}
        t1=time.time()
        h_dict_context1 = self.context_waveform_generator1.frequency_domain_strain(parameters)
        h_dict_context2 = self.context_waveform_generator2.frequency_domain_strain(parameters)
        t2=time.time()
        print(f'Context waveform generation time cost: {t2-t1}')
        for key in ['plus', 'cross']:
            h_dict_context2[key] = get_shifted_h2_zeropad(h_dict_context1[key], h_dict_context2[key], self.example_det)
        t3=time.time()
        print(f'Context waveform shifting time cost: {t3-t2}')
        farray0 = self.context_frequency_array[mask]

        h_dict_context1_processed = {}
        h_dict_context2_processed = {}
        for mode2scale in ['plus','cross']:
            # mask
            h_dict_context1_processed[mode2scale] = h_dict_context1[mode2scale][mask]
            h_dict_context2_processed[mode2scale] = h_dict_context2[mode2scale][mask]

            # monochromatize
            f_mono, h_dict_context1_processed[mode2scale] = self.monochromatize_waveform(farray0, h_dict_context1_processed[mode2scale], parameters['chirp_mass'])
            f_mono, h_dict_context2_processed[mode2scale] = self.monochromatize_waveform(farray0, h_dict_context2_processed[mode2scale], parameters['chirp_mass'])

            # resample
            f_mono_resampled, h1_mono_resampled_real = self.resample_mono_fdwaveforms(f_mono, np.real(h_dict_context1_processed[mode2scale]))
            f_mono_resampled, h1_mono_resampled_imag = self.resample_mono_fdwaveforms(f_mono, np.imag(h_dict_context1_processed[mode2scale]))
            h_dict_context1_processed[mode2scale] = h1_mono_resampled_real + h1_mono_resampled_imag*1j

            f_mono_resampled, h2_mono_resampled_real = self.resample_mono_fdwaveforms(f_mono, np.real(h_dict_context2_processed[mode2scale]))
            f_mono_resampled, h2_mono_resampled_imag = self.resample_mono_fdwaveforms(f_mono, np.imag(h_dict_context2_processed[mode2scale]))
            h_dict_context2_processed[mode2scale] = h2_mono_resampled_real + h2_mono_resampled_imag*1j
            


        h_dict_context_components={}
        ff_mono_resampled, h_dict_context_components['plus_real'] = self.mix_waveforms(f_mono_resampled, h_dict_context1_processed, h_dict_context2_processed, 'plus', 'real', self.context_fraction)
        ff_mono_resampled, h_dict_context_components['plus_imag'] = self.mix_waveforms(f_mono_resampled, h_dict_context1_processed, h_dict_context2_processed, 'plus', 'imag', self.context_fraction)
        ff_mono_resampled, h_dict_context_components['cross_real'] = self.mix_waveforms(f_mono_resampled, h_dict_context1_processed, h_dict_context2_processed, 'cross', 'real', self.context_fraction)
        ff_mono_resampled, h_dict_context_components['cross_imag'] = self.mix_waveforms(f_mono_resampled, h_dict_context1_processed, h_dict_context2_processed, 'cross', 'imag', self.context_fraction)
        
        #h_dict_context_components['plus_real'] = np.zeros(len(ff_mono_resampled))
        #h_dict_context_components['plus_imag'] = np.zeros(len(ff_mono_resampled))
        #h_dict_context_components['cross_real'] = np.zeros(len(ff_mono_resampled))
        #h_dict_context_components['cross_imag'] = np.zeros(len(ff_mono_resampled))

        mean_dict={}
        std_dict={}
        t4=time.time()
        for label,h_mono_resampled in h_dict_context_components.items():
            model = self.npmodel_dict[label]
            target_f_mono_resampled = rescale_range(f_mono_resampled, (f_mono_resampled.min(),f_mono_resampled.max()), (-1,1))
            mean_resampled, std_resampled = get_predictions(model,
                                        torch.from_numpy(ff_mono_resampled).unsqueeze(-1).unsqueeze(0).type(torch.float32), 
                                        torch.from_numpy(h_mono_resampled).unsqueeze(-1).unsqueeze(0).type(torch.float32), 
                                        torch.from_numpy(target_f_mono_resampled).unsqueeze(-1).unsqueeze(0).type(torch.float32), 
                                        1)

            farray2, mean_scaled = self.unmonochromatize_waveform(f_mono_resampled, mean_resampled, parameters['chirp_mass'])
            farray2, std_scaled = self.unmonochromatize_waveform(f_mono_resampled, std_resampled, parameters['chirp_mass'])

            # unscale to injection mass
            # farray3 should == farray (?)
            farray3, mean = self.scale_waveform(target_mass=mtot, base_f=farray2, base_h=mean_scaled, base_mass=self.npmodel_total_mass)
            farray3, std = self.scale_waveform(target_mass=mtot, base_f=farray2, base_h=std_scaled, base_mass=self.npmodel_total_mass)

            interpolator_mean = scipy.interpolate.CubicSpline(farray3, mean)
            mean_at_freq = interpolator_mean(self.context_frequency_array[mask_output])

            interpolator_std = scipy.interpolate.CubicSpline(farray3, std)
            std_at_freq = interpolator_std(self.context_frequency_array[mask_output])

            #zerolenth = len(np.where(self.context_frequency_array<min(fmin4inj,self.context_f_low))[0])
            zerolenth = len(np.where(self.context_frequency_array<max(self.context_f_low,self.npmodel_f_low/mratio))[0])
            zero_paddings = np.zeros(zerolenth)
            
            mean_dict[label] = np.append(zero_paddings, mean_at_freq) * 100 / d_l
            #mean_dict[label] = mean_resampled
            std_dict[label] = np.append(zero_paddings, std_at_freq) * 100 / d_l
        t5=time.time()
        print(f'NP prediction time cost: {t5-t4}')
        h_dict = {}
        error_dict = {}
        for key in ['plus', 'cross']:
            h_dict[key] = mean_dict[f'{key}_real'] + mean_dict[f'{key}_imag']*1j
            error_dict[key] = std_dict[f'{key}_real'] + std_dict[f'{key}_imag']*1j

        self.context_waveform_generator1.waveform_arguments['reference_frequency'] /= mratio
        self.context_waveform_generator2.waveform_arguments['reference_frequency'] /= mratio
        parameters['luminosity_distance']=d_l
        parameters['chirp_mass'] *= mratio

        self.context_waveform_generator1.waveform_arguments['minimum_frequency'] = self.context_f_low
        self.context_waveform_generator2.waveform_arguments['minimum_frequency'] = self.context_f_low
        return h_dict, error_dict


class NPUeqMixWaveformGenerator():
    def __init__(self, model_path,
                context_waveform_generator1,
                context_waveform_generator2,
                context_fraction = 0.3,
                device='cpu',
                R_DIM=128,
                npmodel_total_mass=40, 
                npmodel_f_low=20,
                npmodel_f_ref = 50, 
                npmodel_duration = 32,
                npmodel_sampling_frequency = 8192,
                example_det = None,
                zero_pad_factor = 0
                ):

        self.context_fraction = context_fraction
        self.zero_pad_factor = zero_pad_factor
        self.R_DIM = R_DIM
        npmodel_dict = get_trained_gwmodels(model_path,device=device,R_DIM=R_DIM)
        self.npmodel_dict = npmodel_dict
        self.npmodel_total_mass = npmodel_total_mass
        self.npmodel_f_low = npmodel_f_low
        self.npmodel_f_ref = npmodel_f_ref
        self.npmodel_duration = npmodel_duration
        self.npmodel_sampling_frequency = npmodel_sampling_frequency
        #self.npmodel_frequency_array =\
        #    np.linspace(0, npmodel_sampling_frequency/2, npmodel_sampling_frequency//2 * duration + 1)

        self.context_waveform_generator1 = context_waveform_generator1
        self.context_waveform_generator2 = context_waveform_generator2
        self.context_duration = context_waveform_generator1.duration
        self.context_sampling_frequency = context_waveform_generator1.sampling_frequency
        self.context_f_low = context_waveform_generator1.waveform_arguments['minimum_frequency']
        self.context_f_ref = context_waveform_generator1.waveform_arguments['reference_frequency']
        self.context_frequency_array = context_waveform_generator1.frequency_array
        self.context_frequency_mask = context_waveform_generator1.frequency_array>=self.context_f_low

        if example_det is None:
            duration=self.context_duration
            f_lower=self.context_f_low
            sampling_frequency=self.context_sampling_frequency
            ifos = bilby.gw.detector.InterferometerList(['L1'])
            self.example_det = ifos[0]
            self.example_det.duration = duration
            self.example_det.sampling_frequency=sampling_frequency
            self.example_det.frequency_mask = self.example_det.frequency_array>=f_lower
        else:
            self.example_det = example_det
        
    
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
        
        fs_rd = 0.4 # ISCO freq for 40 Msun, =110Hz
        new_fs1 = np.linspace(fs_min, fs_rd, int(len(f_mono_new)*fs_rd/(fs_max-fs_min)) )
        new_fs2 = np.linspace(fs_rd, fs_max, len(f_mono_new)//10)
        new_fs = np.append(new_fs1,new_fs2[1:])
        
        interpolator = scipy.interpolate.CubicSpline(f_mono_new, h_mono_new)
        new_h = interpolator(new_fs)
        
        return new_fs[::-1], new_h[::-1]
        #return new_fs, new_h
    
    def unresample_mono_fdwaveforms(self, f_mono_resampled, h_mono_resampled, f_mono):
        interpolator = scipy.interpolate.CubicSpline(f_mono_resampled[::-1], h_mono_resampled[::-1])
        h_mono = interpolator(f_mono)
        return h_mono


    def mix_waveforms(self, farray, h1_dict, h2_dict, mode, part, injection_parameters, fraction=0.01, fmin2=None):

        l = len(farray)
        if fmin2 is None:
            fmin2 = farray[-1]
        else:
            fmin4SUR = fmin2**(-5/3) * 1e3
        sur_index = np.where(farray<fmin4SUR)[0]
        lsur = len(sur_index)

        farray_m1to1 = rescale_range(farray, (farray.min(),farray.max()), (-1,1))
        ctxtindex = np.arange(0,l,int(1/fraction))
        ctxtindexsur = np.arange(0,lsur,int(1/fraction))

        ffarray = np.array(list(zip(farray_m1to1[ctxtindex], farray_m1to1[sur_index][ctxtindexsur]))).flatten()
        if part == 'real':
            return ffarray, np.real(np.array(list(zip(h1_dict[mode][ctxtindex], h2_dict[mode][sur_index][ctxtindexsur]))).flatten())
        elif part == 'imag':
            return ffarray, np.imag(np.array(list(zip(h1_dict[mode][ctxtindex], h2_dict[mode][sur_index][ctxtindexsur]))).flatten())

    def frequency_domain_strain(self, parameters):
        mtot = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_total_mass(
            parameters['chirp_mass'],
            parameters['mass_ratio']
        )
        mratio = mtot/self.npmodel_total_mass
        #self.context_waveform_generator1.waveform_arguments['reference_frequency'] *= mratio
        #self.context_waveform_generator2.waveform_arguments['reference_frequency'] *= mratio

        # However, keep in mind that our model learns weaveforms from 40Msun, f_low=20Hz - these two parameters do not span over the whole space. We have to scale them. 
        self.context_waveform_generator1.waveform_arguments['minimum_frequency'] = self.npmodel_f_low
        #self.context_waveform_generator2.waveform_arguments['minimum_frequency'] = self.npmodel_f_low

        if mratio<1 and self.context_f_low<self.npmodel_f_low/mratio: 
            #logging.warning(f"Mtot={mtot} < Training Mtot {self.npmodel_total_mass}, starting frequency is at least {self.npmodel_f_low/mratio}Hz, which is higher than given {self.context_f_low}Hz. Setting zeros between two frequencies.")
            print(f"Warning: Mtot={mtot} < Training Mtot {self.npmodel_total_mass}, starting frequency should be at least {self.npmodel_f_low/mratio}Hz, which is higher than given {self.context_f_low}Hz. Setting zeros between two frequencies.")

        mask = self.context_frequency_array>=self.npmodel_f_low
        mask_output = self.context_frequency_array>=max(self.context_f_low,self.npmodel_f_low/mratio)
        d_l = parameters['luminosity_distance']
        parameters['luminosity_distance'] = 100
        parameters['chirp_mass'] = parameters['chirp_mass']/mratio

        fmin4SUR = np.ceil(safe_fmin_NRSur7dq4(parameters))
        self.context_waveform_generator2.waveform_arguments['minimum_frequency'] = fmin4SUR
        print(f'Assuming waveform generator 2 is for NRSur7dq4. Changing starting frequency to {fmin4SUR}Hz.')

        h_dict_context1 = {}
        h_dict_context2 = {}
        t1=time.time()
        h_dict_context1 = self.context_waveform_generator1.frequency_domain_strain(parameters)
        h_dict_context2 = self.context_waveform_generator2.frequency_domain_strain(parameters)
        t2=time.time()
        print(f'Context waveform generation time cost: {t2-t1}')
        for key in ['plus', 'cross']:
            h_dict_context2[key] = get_shifted_h2_zeropad(h_dict_context1[key], h_dict_context2[key], self.example_det, zero_pad_factor=self.zero_pad_factor)
        t3=time.time()
        print(f'Context waveform shifting time cost: {t3-t2}')
        farray0 = self.context_frequency_array[mask]

        h_dict_context1_processed = {}
        h_dict_context2_processed = {}
        for mode2scale in ['plus','cross']:
            # mask
            h_dict_context1_processed[mode2scale] = h_dict_context1[mode2scale][mask]
            h_dict_context2_processed[mode2scale] = h_dict_context2[mode2scale][mask]

            # monochromatize
            f_mono, h_dict_context1_processed[mode2scale] = self.monochromatize_waveform(farray0, h_dict_context1_processed[mode2scale], parameters['chirp_mass'])
            f_mono, h_dict_context2_processed[mode2scale] = self.monochromatize_waveform(farray0, h_dict_context2_processed[mode2scale], parameters['chirp_mass'])

            # resample
            f_mono_resampled, h1_mono_resampled_real = self.resample_mono_fdwaveforms(f_mono, np.real(h_dict_context1_processed[mode2scale]))
            f_mono_resampled, h1_mono_resampled_imag = self.resample_mono_fdwaveforms(f_mono, np.imag(h_dict_context1_processed[mode2scale]))
            h_dict_context1_processed[mode2scale] = h1_mono_resampled_real + h1_mono_resampled_imag*1j

            f_mono_resampled, h2_mono_resampled_real = self.resample_mono_fdwaveforms(f_mono, np.real(h_dict_context2_processed[mode2scale]))
            f_mono_resampled, h2_mono_resampled_imag = self.resample_mono_fdwaveforms(f_mono, np.imag(h_dict_context2_processed[mode2scale]))
            h_dict_context2_processed[mode2scale] = h2_mono_resampled_real + h2_mono_resampled_imag*1j
            


        h_dict_context_components={}
        ff_mono_resampled, h_dict_context_components['plus_real'] = self.mix_waveforms(f_mono_resampled, h_dict_context1_processed, h_dict_context2_processed, 'plus', 'real', self.context_fraction, fmin2=fmin4SUR)
        ff_mono_resampled, h_dict_context_components['plus_imag'] = self.mix_waveforms(f_mono_resampled, h_dict_context1_processed, h_dict_context2_processed, 'plus', 'imag', self.context_fraction, fmin2=fmin4SUR)
        ff_mono_resampled, h_dict_context_components['cross_real'] = self.mix_waveforms(f_mono_resampled, h_dict_context1_processed, h_dict_context2_processed, 'cross', 'real', self.context_fraction, fmin2=fmin4SUR)
        ff_mono_resampled, h_dict_context_components['cross_imag'] = self.mix_waveforms(f_mono_resampled, h_dict_context1_processed, h_dict_context2_processed, 'cross', 'imag', self.context_fraction, fmin2=fmin4SUR)
        
        #h_dict_context_components['plus_real'] = np.zeros(len(ff_mono_resampled))
        #h_dict_context_components['plus_imag'] = np.zeros(len(ff_mono_resampled))
        #h_dict_context_components['cross_real'] = np.zeros(len(ff_mono_resampled))
        #h_dict_context_components['cross_imag'] = np.zeros(len(ff_mono_resampled))

        mean_dict={}
        std_dict={}
        t4=time.time()
        for label,h_mono_resampled in h_dict_context_components.items():
            model = self.npmodel_dict[label]
            target_f_mono_resampled = rescale_range(f_mono_resampled, (f_mono_resampled.min(),f_mono_resampled.max()), (-1,1))
            mean_resampled, std_resampled = get_predictions(model,
                                        torch.from_numpy(ff_mono_resampled).unsqueeze(-1).unsqueeze(0).type(torch.float32), 
                                        torch.from_numpy(h_mono_resampled).unsqueeze(-1).unsqueeze(0).type(torch.float32), 
                                        torch.from_numpy(target_f_mono_resampled).unsqueeze(-1).unsqueeze(0).type(torch.float32), 
                                        1)

            farray2, mean_scaled = self.unmonochromatize_waveform(f_mono_resampled, mean_resampled, parameters['chirp_mass'])
            farray2, std_scaled = self.unmonochromatize_waveform(f_mono_resampled, std_resampled, parameters['chirp_mass'])

            # unscale to injection mass
            # farray3 should == farray (?)
            farray3, mean = self.scale_waveform(target_mass=mtot, base_f=farray2, base_h=mean_scaled, base_mass=self.npmodel_total_mass)
            farray3, std = self.scale_waveform(target_mass=mtot, base_f=farray2, base_h=std_scaled, base_mass=self.npmodel_total_mass)

            interpolator_mean = scipy.interpolate.CubicSpline(farray3, mean)
            mean_at_freq = interpolator_mean(self.context_frequency_array[mask_output])

            interpolator_std = scipy.interpolate.CubicSpline(farray3, std)
            std_at_freq = interpolator_std(self.context_frequency_array[mask_output])

            #zerolenth = len(np.where(self.context_frequency_array<min(fmin4inj,self.context_f_low))[0])
            zerolenth = len(np.where(self.context_frequency_array<max(self.context_f_low,self.npmodel_f_low/mratio))[0])
            zero_paddings = np.zeros(zerolenth)
            
            mean_dict[label] = np.append(zero_paddings, mean_at_freq) * 100 / d_l
            #mean_dict[label] = mean_resampled
            std_dict[label] = np.append(zero_paddings, std_at_freq) * 100 / d_l
        t5=time.time()
        print(f'NP prediction time cost: {t5-t4}')
        h_dict = {}
        error_dict = {}
        for key in ['plus', 'cross']:
            h_dict[key] = mean_dict[f'{key}_real'] + mean_dict[f'{key}_imag']*1j
            error_dict[key] = std_dict[f'{key}_real'] + std_dict[f'{key}_imag']*1j

        #self.context_waveform_generator1.waveform_arguments['reference_frequency'] /= mratio
        #self.context_waveform_generator2.waveform_arguments['reference_frequency'] /= mratio
        parameters['luminosity_distance']=d_l
        parameters['chirp_mass'] *= mratio

        self.context_waveform_generator1.waveform_arguments['minimum_frequency'] = self.context_f_low
        self.context_waveform_generator2.waveform_arguments['minimum_frequency'] = self.context_f_low
        return h_dict, error_dict