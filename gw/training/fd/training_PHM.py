import logging
import os
import warnings
import sys

import torch
import numpy as np

os.chdir("/home/qian.hu/neuron_process_waveform/npf_GWwaveform/")
sys.path.append('/home/qian.hu/neuron_process_waveform/npf_GWwaveform/')

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
logging.disable(logging.ERROR)

N_THREADS = 10
torch.set_num_threads(N_THREADS)

from gw.utils import gwutils
from npf import CNPFLoss
from utils.train import train_models
from functools import partial
from npf import ConvCNP
from npf.architectures import CNN, MLP, ResConvBlock, SetConv, discard_ith_arg
from npf.utils.datasplit import (
    CntxtTrgtGetter,
    GetRandomIndcs,
    GridCntxtTrgtGetter,
    RandomMasker,
    get_all_indcs, ##
    no_masker,
)
from utils.data import cntxt_trgt_collate




root_dir = '/home/qian.hu/neuron_process_waveform/npf_GWwaveform/data/'
h5filename = root_dir + 'gw_fd_8D_q25a8M40_2N10k_IMREOB_PHM.h5'
output_dir = "/home/qian.hu/neuron_process_waveform/npf_GWwaveform/gw/trained_models/FULLFD_IMREOB_PHM_q25a8M40_2N10k/"

Ngw = gwutils.get_gwfdh5_nsample(h5filename)
Ntrain = int(Ngw*0.7)
Ntest = int(Ngw*0.15)
Nvalid = Ngw - Ntrain - Ntest


random_index = np.random.permutation(Ngw)
train_index = random_index[:Ntrain]
test_index = random_index[Ntrain:Ntrain+Ntest]
valid_index = random_index[-Nvalid:]
np.savetxt(f'{output_dir}trainindex.txt', train_index)
np.savetxt(f'{output_dir}testindex.txt', test_index)
np.savetxt(f'{output_dir}validindex.txt', valid_index)

gw_datasets = {}
gw_test_datasets = {}
gw_valid_datasets = {}
for mode in ['plus', 'cross']:
    for part in ['real', 'imag']:
        train_label = f'{mode}_{part}'
        print(f"Loading {train_label}...")
        gw_dataset = gwutils.GWDatasetFDMultimodel(h5file=h5filename, indcies=train_index, mode=mode, part=part)
        gw_test_dataset = gwutils.GWDatasetFDMultimodel(h5file=h5filename, indcies=test_index, mode=mode, part=part)
        gw_valid_dataset = gwutils.GWDatasetFDMultimodel(h5file=h5filename, indcies=valid_index, mode=mode, part=part)
        
        gw_datasets[train_label] = gw_dataset
        gw_test_datasets[train_label] = gw_test_dataset
        gw_valid_datasets[train_label] = gw_valid_dataset



# CONTEXT TARGET SPLIT
get_cntxt_trgt_1d = cntxt_trgt_collate(
    CntxtTrgtGetter(
        contexts_getter=GetRandomIndcs(a=0.6, b=0.8), targets_getter=GetRandomIndcs(a=0.6, b=0.8), #GetRandomIndcs(a=0.8, b=0.9)
    )
)

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


KWARGS = dict(
    is_retrain=True,  # whether to load precomputed model or retrain
    is_continue_train=False,
    criterion=CNPFLoss,
    chckpnt_dirname=output_dir,
    device="cuda",
    lr=5e-5,
    decay_lr=10,
    seed=123,
    batch_size=2,
)

# 1D
trainers_1d = train_models(
    gw_datasets,
    {"ConvCNP": model_1d},
    test_datasets=gw_test_datasets,
    iterator_train__collate_fn=get_cntxt_trgt_1d,
    iterator_valid__collate_fn=get_cntxt_trgt_1d,
    max_epochs=50,
    **KWARGS
)

