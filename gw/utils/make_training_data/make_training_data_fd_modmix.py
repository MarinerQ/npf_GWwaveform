import numpy as np
from functools import partial
import multiprocessing
from multiprocessing import Pool
import sys

import bilby
from pesummary.gw.conversions import spins as pespin
from bilby.gw import conversion
import lal
import lalsimulation

import os
os.environ['LAL_DATA_PATH'] = '/home/qian.hu/lalsuite_extra_tempfiles/'
#os.environ['LAL_DATA_PATH'] = '/Users/qianhu/Documents/lalsuite-extra-master/data/lalsimulation/'
#try:
#    os.chdir('/home/qian.hu/neuron_process_waveform/npf_GWwaveform/gw')
#except:
#    os.chdir('/Users/qianhu/Documents/Glasgow/research/np_waveform/npf_GWwaveform/gw')

from cnmdgwutils import (
    safe_fmin_NRSur7dq4,
    get_shifted_h2_zeropad,
    save_dict_to_hdf5,
    scale_aligned_fdwaveforms,
    resample_scaled_fdwaveforms,
    generate_random_spin,
    generate_random_angle,
    get_inj_paras
    )



def make_FDaligned_waveforms(injection_parameters,
        duration, f_lower,sampling_frequency,
        approximant_list=['IMRPhenomPv2','SEOBNRv4P'],
        mode='plus',f_ref=50, mode_array=None):
    ''' 
    Waveforms are aligned with the 1st model in approximant_list.
    '''
    ifos = bilby.gw.detector.InterferometerList(['L1'])
    det = ifos[0]
    det.duration = duration
    det.sampling_frequency=sampling_frequency
    det.frequency_mask = det.frequency_array>=f_lower
    mask = det.frequency_mask
    freq_array = det.frequency_array[mask]
    
    h_list = []
    
    if mode in ['plus', 'cross']:
        for i,approx in enumerate(approximant_list):
            if approx=='NRSur7dq4':
                fmin_laltd = np.ceil(safe_fmin_NRSur7dq4(injection_parameters))
                if mode_array:
                    waveform_arguments = dict(waveform_approximant=approx,
                                reference_frequency=f_ref, minimum_frequency=fmin_laltd,mode_array=mode_array) 
                else:
                    waveform_arguments = dict(waveform_approximant=approx,
                                reference_frequency=f_ref, minimum_frequency=fmin_laltd)  
            else:
                if mode_array:
                    waveform_arguments = dict(waveform_approximant=approx,
                              reference_frequency=f_ref, minimum_frequency=f_lower,mode_array=mode_array)
                else:
                    waveform_arguments = dict(waveform_approximant=approx,
                              reference_frequency=f_ref, minimum_frequency=f_lower)


            waveform_generator = bilby.gw.WaveformGenerator(
                duration=duration, sampling_frequency=sampling_frequency,
                frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                waveform_arguments=waveform_arguments)


            h = waveform_generator.frequency_domain_strain(parameters=injection_parameters)[mode]
            if i==0:
                h_list.append(h)
            else:
                h = get_shifted_h2_zeropad(h_list[0],h,det)
                h_list.append(h)

        h_masked = []
        for h_to_mask in h_list:
            h_masked.append(h_to_mask[mask])

        return freq_array, h_masked
    
    
    elif mode == 'both':
        # h_list = [approx1_plus, approx1_cross, approx2_plus, approx2_cross, ...]
        for i,approx in enumerate(approximant_list):
            if approx=='NRSur7dq4':
                fmin_laltd = np.ceil(safe_fmin_NRSur7dq4(injection_parameters))
                if mode_array:
                    waveform_arguments = dict(waveform_approximant=approx,
                                reference_frequency=f_ref, minimum_frequency=fmin_laltd,mode_array=mode_array) 
                else:
                    waveform_arguments = dict(waveform_approximant=approx,
                                reference_frequency=f_ref, minimum_frequency=fmin_laltd)  
            else:
                if mode_array:
                    waveform_arguments = dict(waveform_approximant=approx,
                              reference_frequency=f_ref, minimum_frequency=f_lower,mode_array=mode_array)
                else:
                    waveform_arguments = dict(waveform_approximant=approx,
                              reference_frequency=f_ref, minimum_frequency=f_lower)


            waveform_generator = bilby.gw.WaveformGenerator(
                duration=duration, sampling_frequency=sampling_frequency,
                frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                waveform_arguments=waveform_arguments)


            h = waveform_generator.frequency_domain_strain(parameters=injection_parameters)
            for j,m in enumerate(['plus', 'cross']):
                if i==0:
                    h_list.append(h[m])
                else:
                    hh = get_shifted_h2_zeropad(h_list[j],h[m],det)
                    h_list.append(hh)
        
        h_masked = []
        for h_to_mask in h_list:
            h_masked.append(h_to_mask[mask])

        return freq_array, h_masked


# nohup python make_training_data_fd_modmix.py PHM 40 >nohup_PHM40.out &
# nohup python make_training_data_fd_modmix.py P 40 >nohup_P40.out &
# nohup python make_training_data_fd_modmix.py PHMsur 40 >nohup_PHMsur40.out &
if __name__ == '__main__':
    phy = str(sys.argv[1])
    Mtot = int(sys.argv[2])

    N=5000
    #q = np.logspace(np.log10(0.5),0,N)  # q from 0.5 to 1
    q = np.linspace(0.25,1,N)

    #Mtot=60
    #Mtot=25  # minimum m2=5 for qmin=0.25
    #Mtot=40  # minimum m2=8 for qmin=0.25

    mass_1 = Mtot/(1+q)
    mass_2 = mass_1*q

    mass_ratio = np.zeros(N) + q
    chirp_mass = conversion.component_masses_to_chirp_mass(mass_1,mass_2)
        
    spin1x,spin1y,spin1z = generate_random_spin(N, a_max=0.99)
    spin2x,spin2y,spin2z = generate_random_spin(N, a_max=0.99)


    iota = generate_random_angle(N, 'cos')
    #fref_list = np.zeros(N)+50.0
    fref_list = np.zeros(N)+ 6**(-3/2)/np.pi/lal.MRSUN_SI*lal.C_SI/Mtot # = 110 for 40Msun
    phiref_list = np.zeros(N)
    converted_spin = pespin.spin_angles(mass_1,mass_2,iota , spin1x, spin1y, spin1z, spin2x, spin2y,spin2z, fref_list,phiref_list)
    theta_jn = converted_spin[:,0]
    phi_jl = converted_spin[:,1]
    tilt_1 = converted_spin[:,2]
    tilt_2 = converted_spin[:,3]
    phi_12 = converted_spin[:,4]
    a_1 = converted_spin[:,5]
    a_2 = converted_spin[:,6]
            
            
    luminosity_distance = np.zeros(N) + 100
    phase = np.zeros(N)

    # unimportant paras
    geocent_time= np.zeros(N)
    psi = np.zeros(N)
    ra = np.zeros(N)
    dec = np.zeros(N)


    para_list = [chirp_mass,mass_ratio,a_1,a_2,tilt_1,tilt_2,phi_12,phi_jl,
                    theta_jn, psi, phase, ra, dec, luminosity_distance, geocent_time]

    samples = np.zeros(shape=(N,len(para_list)) )

    for i in range(len(para_list)):
        samples[:,i] = para_list[i] 

    mode_array = [[2,2],[2,-2]]
    duration=16*2
    f_lower=20
    sampling_frequency=4096/2

    if phy=="P":
        approximant_list = ['IMRPhenomPv2','SEOBNRv4P']
    elif phy=="PHM":
        approximant_list = ['IMRPhenomXPHM','SEOBNRv4PHM']
    elif phy=="PHMsur":
        approximant_list = ['IMRPhenomXPHM','NRSur7dq4']
    else:
        raise Exception("Wrong phy!")
    #approximant_list = ['IMRPhenomXPHM','SEOBNRv4PHM','NRSur7dq4']
    #
    n_approx = len(approximant_list)

    data_dict = dict()

    data_dict['frequency'] = dict()
    data_dict['frequency']['frequency_array_original'] = []
    data_dict['frequency']['frequency_array_scaled'] = []
    data_dict['frequency']['frequency_array_scaled_resampled'] = []

    data_dict['waveform_fd'] = dict()
    for approx in approximant_list:
        data_dict['waveform_fd'][approx] = dict()
        data_dict['waveform_fd'][approx]['plus'] = dict()
        data_dict['waveform_fd'][approx]['cross'] = dict()
        for mode in ['plus', 'cross']:
            data_dict['waveform_fd'][approx][mode] = dict()
            #data_dict['waveform_fd'][approx][mode]['original'] = []
            #data_dict['waveform_fd'][approx][mode]['scaled'] = []
            data_dict['waveform_fd'][approx][mode]['scaled_resampled'] = []


    data_dict['source_parameters'] = dict()
    parameter_name_list = ['chirp_mass','mass_ratio','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl','theta_jn']
    for paraname in parameter_name_list:
        data_dict['source_parameters'][paraname] = []


    for waveform_index in range(N):
        # calculation
        if i%100 ==0:
            print(f"Flag: {i}-th simulation.")
        injection_para = get_inj_paras(samples[waveform_index])
        
        freq_array, h_list = make_FDaligned_waveforms(injection_para,
            duration, f_lower, sampling_frequency,
            approximant_list=approximant_list, mode='both',mode_array=mode_array)
        freq_array_scaled, h_list_scaled = scale_aligned_fdwaveforms(freq_array, h_list, injection_para['chirp_mass'])
        freq_array_scaled_resampled, h_list_scaled_resampled = resample_scaled_fdwaveforms(freq_array_scaled, h_list_scaled)
        
        
        # save to data_dic
        data_dict['frequency']['frequency_array_original'] = freq_array
        data_dict['frequency']['frequency_array_scaled'] = freq_array_scaled
        data_dict['frequency']['frequency_array_scaled_resampled'] = freq_array_scaled_resampled
        
        for approx_index,approx in enumerate(approximant_list):
            #data_dict['waveform_fd'][approx]['plus']['original'].append(h_list[2*approx_index])
            #data_dict['waveform_fd'][approx]['cross']['original'].append(h_list[2*approx_index+1])
            
            #data_dict['waveform_fd'][approx]['plus']['scaled'].append(h_list_scaled[2*approx_index])
            #data_dict['waveform_fd'][approx]['cross']['scaled'].append(h_list_scaled[2*approx_index+1])
            
            data_dict['waveform_fd'][approx]['plus']['scaled_resampled'].append(h_list_scaled_resampled[2*approx_index])
            data_dict['waveform_fd'][approx]['cross']['scaled_resampled'].append(h_list_scaled_resampled[2*approx_index+1])
        
        for paraname in parameter_name_list:
            data_dict['source_parameters'][paraname].append(injection_para[paraname])

    save_folder = '/home/qian.hu/neuron_process_waveform/npf_GWwaveform/data/'
    #save_folder = '/Users/qianhu/Documents/Glasgow/research/np_waveform/npf_GWwaveform/data/'
    #h5filename = save_folder + f'gw_fd_8D_q25a8M{Mtot}_2N10k_IMREOB_{phy}.h5'
    h5filename = save_folder + f'gw_fd_8D_q4a99M{Mtot}_2N10k_IMRSUR_{phy}22_2ksr.h5'
    # 1: 4s, 4096Hz
    # 2: 16s, 4096Hz, //3
    # 3: 32s, 8192Hz, //10
    save_dict_to_hdf5(data_dict, h5filename)