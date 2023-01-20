import numpy as np
import torch
from torch.utils.data import Dataset
from npf.utils.helpers import rescale_range
import h5py
from npf.neuralproc.base import LatentNeuralProcessFamily

def save_dict_to_hdf5(dic, filename):
    """
    https://codereview.stackexchange.com/questions/120802/recursively-save-python-dictionaries-to-hdf5-files-using-h5py
    """
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)

def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    https://codereview.stackexchange.com/questions/120802/recursively-save-python-dictionaries-to-hdf5-files-using-h5py
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
    https://codereview.stackexchange.com/questions/120802/recursively-save-python-dictionaries-to-hdf5-files-using-h5py
    """
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

def recursively_load_dict_contents_from_group(h5file, path):
    """
    https://codereview.stackexchange.com/questions/120802/recursively-save-python-dictionaries-to-hdf5-files-using-h5py
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



class GWDatasetFDMultimodel(Dataset):
    def __init__(self, h5file,
                root_dir=None, indcies=None,
                mode='plus', part='real', wavetype='scaled_resampled'):
        """
        Gravitational wave dataset. Waveforms are in FD.
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
            self.waveform_models = np.array(list(f['waveform_fd']))
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
        h = h[arg_sorted] / 100

        f = rescale_range(f, (f.min(),f.max()), (-1,1))
        return torch.from_numpy(f).unsqueeze(-1).type(torch.float32), torch.from_numpy(h).unsqueeze(-1).type(torch.float32)

    
    def get_specific_model(self,index, approx):
        f = self.frequency_array
        h = self.fdwaveforms[approx][index] / 100

        f = rescale_range(f, (f.min(),f.max()), (-1,1))

        return torch.from_numpy(f).unsqueeze(-1).type(torch.float32), torch.from_numpy(h).unsqueeze(-1).type(torch.float32)
