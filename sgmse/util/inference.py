import torch
import numpy as np
from torchaudio import load

from pesq import pesq
from pystoi import stoi
import torch.nn.functional as F
import random

from .other import si_sdr, pad_spec
import wespeakerruntime as wespeaker
import torchaudio.compliance.kaldi as kaldi
import random
from scipy import signal
from scipy.io import wavfile
import torchaudio.sox_effects as sox_effects
from sgmse.data_module import mel_spectrogram


# Settings
sr = 16000
snr = 0.5
N = 30
corrector_steps = 1


def evaluate_model(model, num_eval_files):

    clean_files = model.data_module.valid_set.clean_files
    noisy_files = model.data_module.valid_set.noisy_files
    
    # Select test files uniformly accros validation files
    total_num_files = len(clean_files)
    indices = torch.linspace(0, total_num_files-1, num_eval_files, dtype=torch.int)
    clean_files = list(clean_files[i] for i in indices)
    noisy_files = list(noisy_files[i] for i in indices)
    spk_emb_extractor=wespeaker.Speaker(lang='en')

    _pesq = 0
    _si_sdr = 0
    _estoi = 0
    # iterate over files
    for (clean_file, noisy_file) in zip(clean_files, noisy_files):
        # Load wavs
        x, _ = load(clean_file)
        y, _ = load(noisy_file) 
        T_orig = x.size(1)   
        spk_emb=spk_emb_extractor.extract_embedding(clean_files[i])

        # Normalize per utterance
        norm_factor = y.abs().max()
        y = y / norm_factor

        # Prepare DNN input
        Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        y = y * norm_factor

        # Reverse sampling
        sampler = model.get_pc_sampler(
            'reverse_diffusion', 'ald', Y.cuda(), N=N, 
            corrector_steps=corrector_steps, snr=snr)
        sample, _ = sampler()

        x_hat = model.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor

        x_hat = x_hat.squeeze().cpu().numpy()
        x = x.squeeze().cpu().numpy()
        y = y.squeeze().cpu().numpy()

        _si_sdr += si_sdr(x, x_hat)
        _pesq += pesq(sr, x, x_hat, 'wb') 
        _estoi += stoi(x, x_hat, sr, extended=True)
        
    return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files



def evaluate_model_on_condition(model, num_eval_files,spk_encoder=False):

    clean_files = model.data_module.valid_set.clean_files
    noisy_files = model.data_module.valid_set.noisy_files
    
    # Select test files uniformly accros validation files
    total_num_files = len(clean_files)
    indices = torch.linspace(0, total_num_files-1, num_eval_files, dtype=torch.int)
    clean_files = list(clean_files[i] for i in indices)
    noisy_files = list(noisy_files[i] for i in indices)
    spk_emb_extractor=wespeaker.Speaker(lang='en')

    _pesq = 0
    _si_sdr = 0
    _estoi = 0
    # iterate over files
    for (clean_file, noisy_file) in zip(clean_files, noisy_files):
        # Load wavs
        x, _ = load(clean_file)
        y, _ = load(noisy_file) 

        if spk_encoder:
            target_len = (512 - 1) * 128
            current_len = x.size(-1)
            pad = max(target_len - current_len, 0)
            if pad == 0:
                # extract random part of the audio file
                start = int((current_len-target_len)/2)
                spk_emb = x[..., start:start+target_len]
            else:
                # pad audio if the length T is smaller than num_frames
                spk_emb = F.pad(x, (pad//2, pad//2+(pad%2)), mode='constant')
            spk_emb = mel_spectrogram(spk_emb, n_fft=1024, num_mels=80, sampling_rate=16000,
                            hop_size=256, win_size=1024, fmin=0, fmax=8000).to(y.device)
            
        else:
            spk_emb=torch.tensor(spk_emb_extractor.extract_embedding(clean_file)).unsqueeze(1).to(y.device)

        x_hat = model.enhance(y, spk_emb)
        x_hat = np.squeeze(x_hat)
        x = x.squeeze().cpu().numpy()
        x = np.squeeze(x)
        #print("x_hat.shape",x_hat.shape, "x.shape",x.shape, x_hat.dtype, x.dtype)
        #print("x",x, "x_hat",x_hat)
        _si_sdr += si_sdr(x, x_hat)
        _pesq += pesq(sr, x, x_hat, 'wb') 
        _estoi += stoi(x, x_hat, sr, extended=True)
        
    return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files


def reverberate(audio, rir_audio_file):
        audio = audio.squeeze().numpy()
        audio = audio.astype(np.float32)
        audio_len = audio.shape[0]
        
        #rir_audio_file = random.choice(rir_files)
        rir_audio, _ = load(rir_audio_file) 
        rir_audio = rir_audio.squeeze().numpy()

        rir_audio = rir_audio.astype(np.float32)
        rir_audio = rir_audio / np.sqrt(np.sum(rir_audio ** 2))

        return signal.convolve(audio, rir_audio, mode='full')[:audio_len]



def evaluate_model_on_condition_rir(model, num_eval_files):

    clean_files = model.data_module.valid_set.clean_files
    noisy_files = model.data_module.valid_set.noisy_files
    rir_files = model.data_module.valid_set.rir_files
    
    # Select test files uniformly accros validation files
    total_num_files = len(clean_files)
    indices = torch.linspace(0, total_num_files-1, num_eval_files, dtype=torch.int)
    clean_files = list(clean_files[i] for i in indices)
    noisy_files = list(noisy_files[i] for i in indices)
    spk_emb_extractor=wespeaker.Speaker(lang='en')

    _pesq = 0
    _si_sdr = 0
    _estoi = 0
    # iterate over files
    for (clean_file, noisy_file) in zip(clean_files, noisy_files):
        # Load wavs
        x, _ = load(clean_file)
        i = random.randint(1,20)
        y = reverberate(x, rir_files[-i])
        y = torch.tensor(y).unsqueeze(0)
        
        spk_emb=torch.tensor(spk_emb_extractor.extract_embedding(clean_file)).unsqueeze(1).to(y.device)

        # # Reverse sampling
        # sampler = model.get_pc_sampler(
        #     'reverse_diffusion', 'ald', Y.cuda(), condition = spk_emb, N=N, 
        #     corrector_steps=corrector_steps, snr=snr)
        # sample, _ = sampler()

        # x_hat = model.to_audio(sample.squeeze(), T_orig)
        # x_hat = x_hat * norm_factor

        # x_hat = x_hat.squeeze().cpu().numpy()
        
        
        # x = x.squeeze().cpu().numpy()
        # y = y.squeeze().cpu().numpy()
        
        x_hat = model.enhance(y, spk_emb)
        x_hat = np.squeeze(x_hat)
        x = x.squeeze().cpu().numpy()
        x = np.squeeze(x)
        #print("x_hat.shape",x_hat.shape, "x.shape",x.shape, x_hat.dtype, x.dtype)
        #print("x",x, "x_hat",x_hat)
        _si_sdr += si_sdr(x, x_hat)
        _pesq += pesq(sr, x, x_hat, 'wb') 
        _estoi += stoi(x, x_hat, sr, extended=True)
        
    return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files







def evaluate_model_on_condition_DDPM(model, num_eval_files):

    clean_files = model.data_module.valid_set.clean_files
    noisy_files = model.data_module.valid_set.noisy_files
    
    # Select test files uniformly accros validation files
    total_num_files = len(clean_files)
    indices = torch.linspace(0, total_num_files-1, num_eval_files, dtype=torch.int)
    clean_files = list(clean_files[i] for i in indices)
    noisy_files = list(noisy_files[i] for i in indices)
    spk_emb_extractor=wespeaker.Speaker(lang='en')

    _pesq = 0
    _si_sdr = 0
    _estoi = 0
    # iterate over files
    for (clean_file, noisy_file) in zip(clean_files, noisy_files):
        # Load wavs
        x, _ = load(clean_file)
        y, _ = load(noisy_file) 
        
        spk_emb=torch.tensor(spk_emb_extractor.extract_embedding(clean_file)).unsqueeze(1).to(y.device)

        x_hat = model.TSE_DDPM(y, spk_emb)
        x_hat = np.squeeze(x_hat)
        x = x.squeeze().cpu().numpy()
        x = np.squeeze(x)
        #print("x_hat.shape",x_hat.shape, "x.shape",x.shape, x_hat.dtype, x.dtype)
        #print("x",x, "x_hat",x_hat)
        _si_sdr += si_sdr(x, x_hat)
        _pesq += pesq(sr, x, x_hat, 'wb') 
        _estoi += stoi(x, x_hat, sr, extended=True)
        #sprint("si_sdr",_si_sdr, "pesq",_pesq, "estoi",_estoi)
        
    return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files

