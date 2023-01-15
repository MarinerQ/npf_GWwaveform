import logging
import os
import warnings

import torch
from torch.utils.data import Dataset, DataLoader

os.chdir("/Users/qianhu/Documents/Glasgow/research/np_waveform/npf_GWwaveform")

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
logging.disable(logging.ERROR)

N_THREADS = 8
IS_FORCE_CPU = False  # Nota Bene : notebooks don't deallocate GPU memory

if IS_FORCE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

torch.set_num_threads(N_THREADS)
import h5py
#10:2:1=train:test:valid

import h5py
import matplotlib.pyplot as plt
import numpy as np
from utils import visualize_1d
from utils.ntbks_helpers import get_all_gp_datasets
from npf.utils.helpers import rescale_range

gp_datasets, gp_test_datasets, gp_valid_datasets = get_all_gp_datasets()
for ds in [gp_datasets, gp_test_datasets, gp_valid_datasets]:
    ds.pop('All_Kernels')
    ds.pop('Noisy_Matern_Kernel')
    ds.pop('RBF_Kernel')
    ds.pop('Variable_Matern_Kernel')
    #ds.pop('Periodic_Kernel')

from utils.ntbks_helpers import get_all_gp_datasets, get_img_datasets

from npf.utils.datasplit import (
    CntxtTrgtGetter,
    GetRandomIndcs,
    GridCntxtTrgtGetter,
    RandomMasker,
    get_all_indcs,
    no_masker,
)
from utils.data import cntxt_trgt_collate, get_test_upscale_factor

# CONTEXT TARGET SPLIT
get_cntxt_trgt_1d = cntxt_trgt_collate(
    CntxtTrgtGetter(
        contexts_getter=GetRandomIndcs(a=0, b=0.5), targets_getter=get_all_indcs,
    )
)

from functools import partial

from npf import ConvCNP, GridConvCNP
from npf.architectures import CNN, MLP, ResConvBlock, SetConv, discard_ith_arg
from npf.utils.helpers import CircularPad2d, make_abs_conv, make_padded_conv
from utils.helpers import count_parameters

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

n_params_1d = count_parameters(model_1d())
print(f"Number Parameters (1D): {n_params_1d:,d}")

import skorch
from npf import CNPFLoss
from utils.ntbks_helpers import add_y_dim
from utils.train import train_models

KWARGS = dict(
    is_retrain=True,  # whether to load precomputed model or retrain
    criterion=CNPFLoss,
    chckpnt_dirname="/Users/qianhu/Documents/Glasgow/research/np_waveform/1d_q/trained_models/",
    device=None,
    lr=1e-3,
    decay_lr=10,
    seed=123,
    batch_size=32,
)

# 1D
# gw <-> gp
trainers_1d = train_models(
    gp_datasets,
    {"ConvCNP": model_1d},
    test_datasets=gp_test_datasets,
    iterator_train__collate_fn=get_cntxt_trgt_1d,
    iterator_valid__collate_fn=get_cntxt_trgt_1d,
    max_epochs=100,
    **KWARGS
)

gp_models = trainers_1d['Periodic_Kernel/ConvCNP/run_0']
gp_model = gp_models.module_

x_test, y_test = gp_test_datasets['Periodic_Kernel'][0]

len_data = len(x_test)
len_context = int(len_data*0.8)
context_index = np.sort( np.random.permutation(len_data)[:len_context] )
target_index = np.arange(len_data)

x_test_context = x_test[context_index]
y_test_context = y_test[context_index]

x_test_target = x_test[target_index]


n_samples=100
p_yCc, *_ = model_gp_rbf.forward(x_test_context.unsqueeze(0), y_test_context.unsqueeze(0), x_test_target.unsqueeze(0))
sampled_y = p_yCc.sample_n(n_samples).detach().numpy()
mean_ys = p_yCc.base_dist.loc.detach().squeeze().numpy()
std_ys = p_yCc.base_dist.scale.detach().squeeze().numpy()

plt.figure(figsize=(12,8))
plt.plot(x_test, y_test, label='Target function', color='k')
plt.scatter(x_test_context, y_test_context, label='Context', color='r')
plt.plot(x_test_target, mean_ys, label='Predicted mean', color='r')
#plt.plot(x_test_target, sampled_y[0].squeeze(), label='Prediction id: 0', color='r')
plt.fill_between(x=x_test_target.squeeze(), y1=mean_ys-std_ys, y2=mean_ys+std_ys, label='1 sigma', alpha=0.1, color='r')

plt.legend()
plt.savefig('gp_reprod.png')
plt.show()