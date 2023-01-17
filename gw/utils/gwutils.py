import numpy as np
import torch
from torch.utils.data import Dataset
from npf.utils.helpers import rescale_range
import h5py
from npf.neuralproc.base import LatentNeuralProcessFamily


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