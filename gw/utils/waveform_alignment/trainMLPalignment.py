
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import h5py
import scipy
from torch.utils.data import Dataset, DataLoader


#device = torch.device("cpu")
device = torch.device("cuda:0")

DT_FACTOR = 300
class DTDPHI_5Dinput_q10a99M40(Dataset):
    def __init__(self, h5file, index, transform=None, mode='plus', device='cpu'):
        
        with h5py.File(h5file, "r") as f:
            a1 = torch.tensor(list(f['source_parameters']['a_1']))
            a2 = torch.tensor(list(f['source_parameters']['a_2'])) 
            mass_ratio = torch.tensor(list(f['source_parameters']['mass_ratio']))
            theta_jn = torch.tensor(list(f['source_parameters']['theta_jn'])) / np.pi
            f_ref = (torch.tensor(list(f['source_parameters']['f_ref'])) - 5 )/195
            self.x = torch.stack((mass_ratio,a1,a2,theta_jn,f_ref),dim=1).to(device)
            
            
            dphi = torch.tensor(list(f['outputs'][f'dphi_{mode}'])) / np.pi
            dt = torch.tensor(list(f['outputs'][f'dt_{mode}'])) * DT_FACTOR
            
            self.y = torch.stack((dphi,dt),dim=1).to(device)
            
        self.transform = transform
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, index):
        
        if self.transform is not None:
            pass
                                      
        return self.x[index], self.y[index]


N=5000

rand_index = np.random.permutation(N)
train_index = rand_index[:int(N*0.7)]
valid_index = rand_index[int(N*0.7):int(N*0.85)]
test_index = rand_index[int(N*0.85):]

input_folder = '/home/qian.hu/neuron_process_waveform/npf_GWwaveform/gw/utils/waveform_alignment/training_data/'
output_folder = '/home/qian.hu/neuron_process_waveform/npf_GWwaveform/gw/utils/waveform_alignment/outputs/'
phy='ASHM'
#phy='AS'
#mode='plus'
mode='cross'
h5filename = f'{input_folder}dtdphi_5D_q10a99M40_IMREOB_{phy}.h5'

train_dataset = DTDPHI_5Dinput_q10a99M40(h5filename, train_index, mode=mode, device=device)
valid_dataset = DTDPHI_5Dinput_q10a99M40(h5filename, valid_index, mode=mode, device=device)
test_dataset = DTDPHI_5Dinput_q10a99M40(h5filename, test_index, mode=mode, device=device)

batch_size = 512
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 2)
        )


    def forward(self, x):
        return self.layers(x)



mlp = MLP().to(device)
loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
epochs = 3000
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)



mean_train_losses = []
mean_valid_losses = []
valid_acc_list = []

checkpoint_filename = f"{output_folder}checkpoint_{phy}_{mode}_5d_q10a99.pickle"
best_epoch = 0
print("Training started.")
for epoch in range(epochs):
    mlp.train()
    
    train_losses = []
    valid_losses = []
    for i, data in enumerate(train_loader):
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()
        optimizer.zero_grad()
        
        outputs = mlp(inputs)
        loss = loss_function(outputs, targets)
        #loss = weighted_loss_function(outputs, targets, weight_for_dt)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
    
            
    mlp.eval()

    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            inputs_valid, targets_valid = data
            inputs_valid, targets_valid = inputs_valid.float(), targets_valid.float()
            outputs_valid = mlp(inputs_valid)
            loss_valid = loss_function(outputs_valid, targets_valid)
            #loss_valid = weighted_loss_function(outputs_valid, targets_valid, weight_for_dt)

            valid_losses.append(loss_valid.item())
    
    if epoch>0:
        if np.mean(valid_losses)<min(mean_valid_losses):
            torch.save(mlp.state_dict(), checkpoint_filename)
            best_epoch = epoch

    mean_train_losses.append(np.mean(train_losses))
    mean_valid_losses.append(np.mean(valid_losses))
    
    if not epoch%500 or epoch==(epochs-1):
        print('epoch : {}, train loss : {:.8f}, valid loss : {:.8f}'\
             .format(epoch, np.mean(train_losses), np.mean(valid_losses)))
print("Training finished.")
print(f"Loading from the best epoch: {best_epoch}, best loss: {min(mean_valid_losses)}")
mlp.load_state_dict(torch.load(checkpoint_filename))




save_model_name = f'{output_folder}alignmodel_{phy}_{mode}_5d_q10a99.pickle'
torch.save(mlp, save_model_name)
#mlp = torch.load("my_model.pickle")





