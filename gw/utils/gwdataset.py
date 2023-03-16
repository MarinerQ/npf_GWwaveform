import logging
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from npf.utils.helpers import rescale_range
import h5py
import time

from .gwutils import *

logging.disable(logging.ERROR)

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
            if "NRSur7dq4" in self.waveform_models:
                self.fmin_NRSur = []
                for i in range(len(self.source_parameters['mass_ratio'])):
                    tempinjdict = {}
                    tempinjdict['mass_ratio'] = self.source_parameters['mass_ratio'][i]
                    tempinjdict['chirp_mass'] = self.source_parameters['chirp_mass'][i]
                    tempinjdict['a_1'] = self.source_parameters['a_1'][i]
                    tempinjdict['a_2'] = self.source_parameters['a_2'][i]
                    tempinjdict['tilt_1'] = self.source_parameters['tilt_1'][i]
                    tempinjdict['tilt_2'] = self.source_parameters['tilt_2'][i]
                    self.fmin_NRSur.append(np.ceil(safe_fmin_NRSur7dq4(tempinjdict)))
                self.maxfmin_NRSur = max(self.fmin_NRSur)
                self.maxcutofflength = len(np.where( self.frequency_array >= self.maxfmin_NRSur**(-5/3)*1e3 )[0])

        
        
        
    def __len__(self):
        return self.n_samples
    
    def remove_nan(self, array):
        return array[~np.isnan(array)]
    
    def __getitem__(self, index):
        f = np.array([])
        h = np.array([])
        if 'NRSur7dq4' in self.waveform_models:
            fcut = self.fmin_NRSur[index]
            fcut = fcut**(-5/3) * 1e3
            nonzeroindex = np.where(self.frequency_array<fcut)[0]
            f = np.append(f, self.frequency_array[nonzeroindex])
            h = np.append(h, self.fdwaveforms['NRSur7dq4'][index][nonzeroindex])
            needcutlength = self.maxcutofflength - (len(self.frequency_array)-len(nonzeroindex))
            for approx in self.waveform_models:
                if approx != 'NRSur7dq4':
                    index_remain = np.random.permutation(len(self.frequency_array))[needcutlength:]
                    f = np.append(f, self.frequency_array[index_remain])
                    h = np.append(h, self.fdwaveforms[approx][index][index_remain])

        else:
            for approx in self.waveform_models:
                f = np.append(f, self.frequency_array)
                h = np.append(h, self.fdwaveforms[approx][index])

        arg_sorted = np.argsort(f)
        f = f[arg_sorted]
        h = h[arg_sorted] #/ 100

        f = rescale_range(f, (f.min(),f.max()), (-1,1))
        return torch.from_numpy(f).unsqueeze(-1).type(torch.float32), torch.from_numpy(h).unsqueeze(-1).type(torch.float32)

    
    def get_specific_model(self,index, approx, rescalex=True):
        if approx == 'NRSur7dq4':
            fcut = self.fmin_NRSur[index]
            fcut = fcut**(-5/3) * 1e3
            nonzeroindex = np.where(self.frequency_array<fcut)[0]
            f = self.frequency_array[nonzeroindex]
            h = self.fdwaveforms[approx][index][nonzeroindex]
        else:
            f = self.frequency_array
            h = self.fdwaveforms[approx][index] #/ 100

        if rescalex:
            f = rescale_range(f, (f.min(),f.max()), (-1,1))

        return torch.from_numpy(f).unsqueeze(-1).type(torch.float32), torch.from_numpy(h).unsqueeze(-1).type(torch.float32)





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