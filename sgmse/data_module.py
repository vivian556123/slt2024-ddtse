
from os.path import join
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from glob import glob
from torchaudio import load
import numpy as np
import torch.nn.functional as F
import wespeakerruntime as wespeaker
import os
from typing import Optional
import numpy as np
import torchaudio
import onnxruntime as ort
import torchaudio.compliance.kaldi as kaldi
import random
from scipy import signal
from scipy.io import wavfile
import torchaudio.sox_effects as sox_effects
from librosa.filters import mel as librosa_mel_fn


def get_window(window_type, window_length):
    if window_type == 'sqrthann':
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    elif window_type == 'hann':
        return torch.hann_window(window_length, periodic=True)
    else:
        raise NotImplementedError(f"Window type {window_type} not implemented!")


class Specs(Dataset):
    def __init__(self, data_dir, subset, dummy, shuffle_spec, num_frames,
            format='default', normalize="noisy", spec_transform=None, only_enhancement="no",
            stft_kwargs=None, train_noisy_data = "mix_both", **ignored_kwargs):

        # Read file paths according to file naming format.
        if format == "default":
            self.clean_files = sorted(glob(join(data_dir, subset) + '/s1/*.wav'))
            if only_enhancement=="yes":
                self.noisy_files = sorted(glob(join(data_dir, subset) + '/mix_single/*.wav'))
                print("use_mix_single_data")
            else: 
                self.noisy_files = sorted(glob(join(data_dir, subset) + '/'+train_noisy_data+'/*.wav'))
        else:
            # Feel free to add your own directory format
            raise NotImplementedError(f"Directory format {format} unknown!")

        self.dummy = dummy
        self.num_frames = num_frames
        self.shuffle_spec = shuffle_spec
        self.normalize = normalize
        self.spec_transform = spec_transform

        assert all(k in stft_kwargs.keys() for k in ["n_fft", "hop_length", "center", "window"]), "misconfigured STFT kwargs"
        self.stft_kwargs = stft_kwargs
        self.hop_length = self.stft_kwargs["hop_length"]
        assert self.stft_kwargs.get("center", None) == True, "'center' must be True for current implementation"

    def __getitem__(self, i):
        x, _ = load(self.clean_files[i])
        y, _ = load(self.noisy_files[i])

        # formula applies for center=True
        target_len = (self.num_frames - 1) * self.hop_length
        current_len = x.size(-1)
        pad = max(target_len - current_len, 0)
        if pad == 0:
            # extract random part of the audio file
            if self.shuffle_spec:
                start = int(np.random.uniform(0, current_len-target_len))
            else:
                start = int((current_len-target_len)/2)
            x = x[..., start:start+target_len]
            y = y[..., start:start+target_len]
        else:
            # pad audio if the length T is smaller than num_frames
            x = F.pad(x, (pad//2, pad//2+(pad%2)), mode='constant')
            y = F.pad(y, (pad//2, pad//2+(pad%2)), mode='constant')

        # normalize w.r.t to the noisy or the clean signal or not at all
        # to ensure same clean signal power in x and y.
        if self.normalize == "noisy":
            normfac = y.abs().max()
        elif self.normalize == "clean":
            normfac = x.abs().max()
        elif self.normalize == "not":
            normfac = 1.0
        x = x / normfac
        y = y / normfac

        X = torch.stft(x, **self.stft_kwargs)
        Y = torch.stft(y, **self.stft_kwargs)

        X, Y = self.spec_transform(X), self.spec_transform(Y)
        return X, Y

    def __len__(self):
        if self.dummy:
            # for debugging shrink the data set size
            return int(len(self.clean_files)/150)
        else:
            return len(self.clean_files)


def compute_fbank(waveform, sample_rate,
                    resample_rate: int = 16000,
                    num_mel_bins: int = 80,
                    frame_length: int = 25,
                    frame_shift: int = 10,
                    dither: float = 0.0,
                    cmn: bool = True):
    """ Extract fbank, simlilar to the one in wespeaker.dataset.processor,
        While integrating the wave reading and CMN.
    """
    if sample_rate != resample_rate:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=resample_rate)(waveform)
    waveform = waveform * (1 << 15)
    mat = kaldi.fbank(waveform,
                        num_mel_bins=num_mel_bins,
                        frame_length=frame_length,
                        frame_shift=frame_shift,
                        dither=dither,
                        sample_frequency=resample_rate,
                        window_type='hamming',
                        use_energy=False)
    mat = mat.numpy()
    if cmn:
        # CMN, without CVN
        mat = mat - np.mean(mat, axis=0)
    return mat



MAX_WAV_VALUE = 32768.0

def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    # if torch.min(y) < -1.:
    #     print('min value is ', torch.min(y))
    # if torch.max(y) > 1.:
    #     print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True,return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec



def extract_spk_emb(spk_emb_extractor, waveform, sample_rate, wav_num=1):
    
    feats = compute_fbank(waveform, sample_rate)
    feats = np.expand_dims(feats, 0)
    spk_emb = spk_emb_extractor.extract_embedding_feat(feats)
    
    return spk_emb

class ConditionalSpecs(Dataset):
    def __init__(self, data_dir, subset, dummy, shuffle_spec, num_frames,
            format='default', normalize="noisy", spec_transform=None,
            stft_kwargs=None, return_time=False, only_enhancement="no", 
            return_prompt="no",return_interference=False, train_noisy_data = "mix_both",
            extend_tasks = False, rir = False, rir_dir=None,  **ignored_kwargs):
        print("format", format, "data_dir", data_dir, "subset", subset)
        # Read file paths according to file naming format.
        if format == "default":
            self.clean_files = sorted(glob(join(data_dir, subset) + '/s1/*.wav'))
            self.clean_files_s2 = sorted(glob(join(data_dir, subset) + '/s2/*.wav'))
            self.only_noisy_files = sorted(glob(join(data_dir, subset) + '/noise/*.wav'))
            self.non_personalized_clean_files = sorted(glob(join(data_dir, subset) + '/mix_clean/*.wav'))
            self.prompt_files = sorted(glob(join(data_dir, subset) + '/s1_prompt/*.wav'))
            if only_enhancement=="yes":
                self.noisy_files = sorted(glob(join(data_dir, subset) + '/mix_single/*.wav'))
                print("use_mix_single_data")
            else: 
                self.noisy_files = sorted(glob(join(data_dir, subset) + '/'+train_noisy_data+'/*.wav'))
            print("len(train data)", len(self.clean_files), len(self.noisy_files))
        elif format == "default_gt_enroll":
            self.clean_files = sorted(glob(join(data_dir, subset) + '/s1/*.wav'))
            self.clean_files_s2 = sorted(glob(join(data_dir, subset) + '/s2/*.wav'))
            self.only_noisy_files = sorted(glob(join(data_dir, subset) + '/noise/*.wav'))
            self.non_personalized_clean_files = sorted(glob(join(data_dir, subset) + '/mix_clean/*.wav'))
            self.prompt_files = self.clean_files
            if only_enhancement=="yes":
                self.noisy_files = sorted(glob(join(data_dir, subset) + '/mix_single/*.wav'))
                print("use_mix_single_data")
            else: 
                self.noisy_files = sorted(glob(join(data_dir, subset) + '/'+train_noisy_data+'/*.wav'))
            print("len(train data)", len(self.clean_files), len(self.noisy_files))
        elif format == "s1_and_s2":
            clean_files_s1 = sorted(glob(join(data_dir, subset) + '/s1/*.wav'))
            clean_files_s2 = sorted(glob(join(data_dir, subset) + '/s2/*.wav'), key=lambda x: x.split('_')[-1])
            self.clean_files = clean_files_s1 + clean_files_s2
            self.only_noisy_files =  [element.replace("/s1/", "/noise/").replace("/s2/", "/noise/") for element in self.clean_files]
            self.non_personalized_clean_files = [element.replace("/s1/", "/mix_clean/").replace("/s2/", "/mix_clean/") for element in self.clean_files]

            if only_enhancement=="yes":
                self.noisy_files = [element.replace("/s1/", "/mix_single/").replace("/s2/", "/mix_single/") for element in self.clean_files]
                #self.noisy_files = sorted(glob(join(data_dir, subset) + '/mix_single/*.wav'))
                print("use_mix_single_data")
            else: 
                self.noisy_files = [element.replace("/s1/", '/'+train_noisy_data+'/').replace("/s2/", '/'+train_noisy_data+'/') for element in self.clean_files]
            print("len(train data)", len(self.clean_files), len(self.noisy_files))
        elif format == "s1_and_s2_gt_enroll":
            clean_files_s1 = sorted(glob(join(data_dir, subset) + '/s1/*.wav'))
            clean_files_s2 = sorted(glob(join(data_dir, subset) + '/s2/*.wav'), key=lambda x: x.split('_')[-1])
            self.clean_files = clean_files_s1 + clean_files_s2
            self.only_noisy_files =  [element.replace("/s1/", "/noise/").replace("/s2/", "/noise/") for element in self.clean_files]
            self.non_personalized_clean_files = [element.replace("/s1/", "/mix_clean/").replace("/s2/", "/mix_clean/") for element in self.clean_files]
            self.prompt_files= self.clean_files
            if only_enhancement=="yes":
                self.noisy_files = [element.replace("/s1/", "/mix_single/").replace("/s2/", "/mix_single/") for element in self.clean_files]
                #self.noisy_files = sorted(glob(join(data_dir, subset) + '/mix_single/*.wav'))
                print("use_mix_single_data")
            else: 
                self.noisy_files = [element.replace("/s1/", '/'+train_noisy_data+'/').replace("/s2/", '/'+train_noisy_data+'/') for element in self.clean_files]
            print("len(train data)", len(self.clean_files), len(self.noisy_files))
        elif format == "voicebank_gt_enroll":
            if subset == "dev":
                self.clean_files = sorted(glob(data_dir + '/clean_dev_wav/*.wav'))
                self.noisy_files = sorted(glob(data_dir + '/noisy_dev_wav/*.wav'))
            else :
                self.clean_files = sorted(glob(data_dir + '/clean_trainset_28spk_wav/*.wav'))
                self.noisy_files = sorted(glob(data_dir+ '/noisy_trainset_28spk_wav/*.wav'))

            self.clean_files_s2 = self.clean_files
            self.only_noisy_files = self.noisy_files
            self.non_personalized_clean_files = self.clean_files
            self.prompt_files = self.clean_files
            print("len(train data)", len(self.clean_files), len(self.noisy_files))
        else:
            # Feel free to add your own directory format
            raise NotImplementedError(f"Directory format {format} unknown!")

        self.dummy = dummy
        self.num_frames = num_frames
        self.shuffle_spec = shuffle_spec
        self.normalize = normalize
        self.spec_transform = spec_transform

        assert all(k in stft_kwargs.keys() for k in ["n_fft", "hop_length", "center", "window"]), "misconfigured STFT kwargs"
        self.stft_kwargs = stft_kwargs
        self.hop_length = self.stft_kwargs["hop_length"]
        assert self.stft_kwargs.get("center", None) == True, "'center' must be True for current implementation"
        
        self.spk_emb_extractor=wespeaker.Speaker(lang='en')
        self.return_time = return_time
        self.only_enhancement = only_enhancement
        self.return_prompt = return_prompt
        self.return_interference = return_interference
        self.extend_tasks = extend_tasks
        self.format = format
        self.rir = rir
        if self.rir:
            with open(os.path.join(rir_dir,'rir_files_shuf_train.txt'), 'r') as file:
                self.rir_files=[]
                for line in file:
                    line = line.strip()
                    self.rir_files.append(line)

    def __getitem__(self, i):
        if self.extend_tasks:
            options = ["s1","s4"]
            probabilities = [0.8, 0.4]
            s1_or_s2 = random.choices(options, probabilities)[0]
        if self.format == "voicebank_gt_enroll":
            s1_or_s2 = "s1"
        else: 
            if self.clean_files[i].split("/")[-2] == "s1":
                s1_or_s2 = "s1"
            elif self.clean_files[i].split("/")[-2] == "s2":
                s1_or_s2 = "s2"
            else: 
                raise NotImplementedError("s1_or_s2 not defined")
            #s1_or_s2 = random.choice(["s1","s2"])
        
        # s1 and s2: target speaker in mixture
        # s3: target speaker not in mixture
        # s4: non personalized enhancement (only input the speech, denoise all background noise)

        if s1_or_s2 == "s3":    
            y, _ = load(self.noisy_files[i])
            x, _ = torch.zeros_like(y), None
            inf_file = random.choice([self.clean_files_s2[i],self.clean_files[i]])
            x_interference, _ = load(inf_file)
        elif s1_or_s2 == "s4":
            y, _ = load(self.noisy_files[i])
            x, _ = load(self.non_personalized_clean_files[i])
            x_interference, _ = load(self.only_noisy_files[i])
        else: 
            x, _ = load(self.clean_files[i])
            y, _ = load(self.noisy_files[i])
            if s1_or_s2 == "s1":
                x_interference, _ = load(self.clean_files[i].replace("s1","s2"))
            elif s1_or_s2 == "s2":
                x_interference, _ = load(self.clean_files[i].replace("s2","s1"))
            else: 
                raise NotImplementedError("s1_or_s2 not defined")
        
        if self.rir : 
            y = reverberate(x, self.rir_files[i%len(self.rir_files)])
            y = torch.tensor(y).unsqueeze(0)
        
        if self.only_enhancement=="no":
            if self.format == "s1_and_s2": 
                prompt_file = random.choice(self.clean_files[min(max(1,i-50),len(self.clean_files)-3):min(i+50,len(self.clean_files)-2)])
                count = 0
                if s1_or_s2 == "s3":
                    while prompt_file.split("/")[-1].split("-")[0] == self.clean_files[i].split("/")[-1].split("-")[0] or prompt_file.split("/")[-1].split("-")[0] == self.clean_files[i].split("/")[-1].split("_")[-1].split("-")[0] and count < 10: 
                        count = count + 1
                        # the prompt speaker is not in the mixture
                        prompt_file = random.choice(self.clean_files[min(max(1,i-100),len(self.clean_files)-3):min(i+100,len(self.clean_files)-2)])
                elif s1_or_s2 == "s2":
                    while prompt_file.split("/")[-1].split("_")[-1].split("-")[0] != self.clean_files[i].split("/")[-1].split("_")[-1].split("-")[0]  and count < 10: 
                        count = count + 1
                        prompt_file = random.choice(self.clean_files[min(max(1,i-50),len(self.clean_files)-3):min(i+50,len(self.clean_files)-2)])
                    if count >= 9:
                        #print("s2","count",count, "prompt_file", prompt_file, "clean_files", self.clean_files[i], self.clean_files[i], self.clean_files[min(max(1,i-10),len(self.clean_files)-3):min(i+10,len(self.clean_files)-2)])
                        prompt_file = self.clean_files[i]
                    prompt_file = prompt_file.replace("s1","s2")
                else:     
                    while prompt_file.split("/")[-1].split("-")[0] != self.clean_files[i].split("/")[-1].split("-")[0] and count < 10: 
                        count = count + 1
                        prompt_file = random.choice(self.clean_files[min(max(1,i-50),len(self.clean_files)-3):min(i+50,len(self.clean_files)-2)])
                    if count >= 9:
                        #print("s1","count",count, "prompt_file", prompt_file, "clean_files", self.clean_files[i], self.clean_files[i], self.clean_files[min(max(1,i-10),len(self.clean_files)-3):min(i+10,len(self.clean_files)-2)])
                        prompt_file = self.clean_files[i]
                    prompt_file = prompt_file.replace("s2","s1")
            else: 
                prompt_file = self.prompt_files[i]
            count = 0
            z, _ = load(prompt_file)
            target_len = (self.num_frames - 1) * self.hop_length
            current_len = z.size(-1)
            pad = max(target_len - current_len, 0)
            if pad == 0:
                # extract random part of the audio file
                if self.shuffle_spec:
                    start = int(np.random.uniform(0, current_len-target_len))
                else:
                    start = int((current_len-target_len)/2)
                z = z[..., start:start+target_len]
            else:
                # pad audio if the length T is smaller than num_frames
                z = F.pad(z, (pad//2, pad//2+(pad%2)), mode='constant')
            spk_emb = torch.tensor(extract_spk_emb(self.spk_emb_extractor, z, _)[0]).unsqueeze(0)
            
        elif self.only_enhancement=="yes" and not self.rir:
            spk_emb = self.spk_emb_extractor.extract_embedding(self.noisy_files[i])
        elif self.only_enhancement=="yes" and self.rir:
            spk_emb = torch.tensor(extract_spk_emb(self.spk_emb_extractor, y, _)[0]).unsqueeze(0)
        else:
            raise NotImplementedError("only_enhancement not defined")

        # formula applies for center=True
        target_len = (self.num_frames - 1) * self.hop_length
        current_len = x.size(-1)
        pad = max(target_len - current_len, 0)
        if pad == 0:
            # extract random part of the audio file
            if self.shuffle_spec:
                start = int(np.random.uniform(0, current_len-target_len))
            else:
                start = int((current_len-target_len)/2)
            x = x[..., start:start+target_len]
            y = y[..., start:start+target_len]
            x_interference = x_interference[..., start:start+target_len]
        else:
            # pad audio if the length T is smaller than num_frames
            x = F.pad(x, (pad//2, pad//2+(pad%2)), mode='constant')
            y = F.pad(y, (pad//2, pad//2+(pad%2)), mode='constant')
            x_interference = F.pad(x_interference, (pad//2, pad//2+(pad%2)), mode='constant')


        # normalize w.r.t to the noisy or the clean signal or not at all
        # to ensure same clean signal power in x and y.
        if self.normalize == "noisy":
            normfac = y.abs().max()
        elif self.normalize == "clean":
            normfac = x.abs().max()
        elif self.normalize == "not":
            normfac = 1.0
        x = x / normfac
        y = y / normfac
        if self.return_time:
            return x, y

        X = torch.stft(x, **self.stft_kwargs)
        Y = torch.stft(y, **self.stft_kwargs)

        X, Y = self.spec_transform(X), self.spec_transform(Y)
        
        if s1_or_s2 == "s4":
            spk_emb = torch.zeros_like(spk_emb)
        
        
        
        if self.return_prompt == "yes":
            z = z / normfac
            Z = torch.stft(z, **self.stft_kwargs)
            return X, Y, spk_emb, Z
        elif self.return_interference and self.only_enhancement=="no":
            z = z / normfac
            Z = torch.stft(z, **self.stft_kwargs)
            Z_mel = mel_spectrogram(z,n_fft=1024, num_mels=80, sampling_rate=16000,
                            hop_size=256, win_size=1024, fmin=0, fmax=8000)
            spk_emb_tgt = torch.tensor(extract_spk_emb(self.spk_emb_extractor, x, _)[0]).unsqueeze(0)
            spk_emb_interference = torch.tensor(extract_spk_emb(self.spk_emb_extractor, x_interference, _)[0]).unsqueeze(0)
            return X, Y, spk_emb, x, y, x_interference, spk_emb_tgt, spk_emb_interference, Z, Z_mel
        elif self.return_interference and self.only_enhancement=="yes":
            return X, Y, spk_emb, x, y, x, spk_emb, spk_emb, X, x
        else:
            return X, Y, spk_emb

    def __len__(self):
        if self.dummy:
            # for debugging shrink the data set size
            return int(len(self.clean_files)/150)
        else:
            return len(self.clean_files)


class SpecsDataModule(pl.LightningDataModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--base_dir", type=str, required=True, help="The base directory of the dataset. Should contain `train`, `valid` and `test` subdirectories, each of which contain `clean` and `noisy` subdirectories.")
        parser.add_argument("--format", type=str, choices=("default", "dns", "s1_and_s2","s1_and_s2_gt_enroll","default_gt_enroll","voicebank_gt_enroll"), default="default", help="Read file paths according to file naming format.")
        parser.add_argument("--batch_size", type=int, default=8, help="The batch size. 8 by default.")
        parser.add_argument("--n_fft", type=int, default=510, help="Number of FFT bins. 510 by default.")   # to assure 256 freq bins
        parser.add_argument("--hop_length", type=int, default=128, help="Window hop length. 128 by default.")
        parser.add_argument("--num_frames", type=int, default=256, help="Number of frames for the dataset. 256 by default.")
        parser.add_argument("--window", type=str, choices=("sqrthann", "hann"), default="hann", help="The window function to use for the STFT. 'hann' by default.")
        parser.add_argument("--num_workers", type=int, default=4, help="Number of workers to use for DataLoaders. 4 by default.")
        parser.add_argument("--dummy", action="store_true", help="Use reduced dummy dataset for prototyping.")
        parser.add_argument("--spec_factor", type=float, default=0.15, help="Factor to multiply complex STFT coefficients by. 0.15 by default.")
        parser.add_argument("--spec_abs_exponent", type=float, default=0.5, help="Exponent e for the transformation abs(z)**e * exp(1j*angle(z)). 0.5 by default.")
        parser.add_argument("--normalize", type=str, choices=("clean", "noisy", "not"), default="noisy", help="Normalize the input waveforms by the clean signal, the noisy signal, or not at all.")
        parser.add_argument("--transform_type", type=str, choices=("exponent", "log", "none"), default="exponent", help="Spectogram transformation for input representation.")
        parser.add_argument("--condition_on_spkemb", type=str, choices=("no", "yes"), default="no", help="no for Spec, yes for ConditionalSpec")
        parser.add_argument("--return_time", action="store_true", help="Return the waveform instead of the STFT")
        parser.add_argument("--only_enhancement", type=str, choices=("no", "yes"), default="no", help="training and testing using mix_single")
        parser.add_argument("--return_prompt", type=str, choices=("no", "yes"), default="no", help="return prompt stft for dataloader")
        parser.add_argument("--return_interference", action="store_true", help="Return the interference speech")
        parser.add_argument("--train_subset", type=str, choices=("train-360", "train-100"), default="train-360", help="Return the interference speech")
        parser.add_argument("--extend_tasks", action="store_true", help="Extend tasks to non-personalized se and non target extraction")
        parser.add_argument("--train_noisy_data", type=str, choices=("mix_both", "mix_clean","mix_single"), default="mix_both", help="Extend tasks to non-personalized se and non target extraction")
        parser.add_argument("--rir_dir", type=str, help="The base directory of the rir dataset")
        parser.add_argument("--rir", action="store_true", help="Extend tasks to non-personalized se and non target extraction")



        return parser

    def __init__(
        self, base_dir, format='default', batch_size=8,
        n_fft=510, hop_length=128, num_frames=256, window='hann',
        num_workers=4, dummy=False, spec_factor=0.15, spec_abs_exponent=0.5,
        gpu=True, normalize='noisy', transform_type="exponent",condition_on_spkemb="no", 
        return_time=False, only_enhancement="no", train_subset = "train-360",
        train_noisy_data = "mix_both", **kwargs
    ):
        super().__init__()
        self.base_dir = base_dir
        print("self.base_dir", self.base_dir)
        self.format = format
        self.batch_size = batch_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.window = get_window(window, self.n_fft)
        self.windows = {}
        self.num_workers = num_workers
        self.dummy = dummy
        self.spec_factor = spec_factor
        self.spec_abs_exponent = spec_abs_exponent
        self.gpu = gpu
        self.normalize = normalize
        self.transform_type = transform_type
        self.kwargs = kwargs
        self.condition_on_spkemb = condition_on_spkemb
        self.return_time= return_time
        self.only_enhancement=only_enhancement
        self.train_subset=train_subset
        self.train_noisy_data = train_noisy_data

    def setup(self, stage=None):
        specs_kwargs = dict(
            stft_kwargs=self.stft_kwargs, num_frames=self.num_frames,
            spec_transform=self.spec_fwd, **self.kwargs
        )
        if stage == 'fit' or stage is None:
            print("preparing dataset, condition_on_spkemb: " + self.condition_on_spkemb)
            if  self.condition_on_spkemb == "no":
                self.train_set = Specs(data_dir=self.base_dir, subset=self.train_subset,
                    dummy=self.dummy, shuffle_spec=True, format=self.format,only_enhancement=self.only_enhancement,
                    normalize=self.normalize, train_noisy_data = self.train_noisy_data,
                    **specs_kwargs)
                self.valid_set = Specs(data_dir=self.base_dir, subset='dev',
                    dummy=self.dummy, shuffle_spec=False, format=self.format,only_enhancement=self.only_enhancement,
                    normalize=self.normalize, train_noisy_data = self.train_noisy_data,
                    **specs_kwargs)
            elif  self.condition_on_spkemb == "yes":
                self.train_set = ConditionalSpecs(data_dir=self.base_dir, subset=self.train_subset,
                    dummy=self.dummy, shuffle_spec=True, format=self.format,only_enhancement=self.only_enhancement,
                    normalize=self.normalize, return_time = self.return_time, train_noisy_data = self.train_noisy_data,
                    **specs_kwargs)
                self.valid_set = ConditionalSpecs(data_dir=self.base_dir, subset='dev',
                    dummy=self.dummy, shuffle_spec=False, return_time = self.return_time,format=self.format,
                    only_enhancement=self.only_enhancement, train_noisy_data = self.train_noisy_data,
                    normalize=self.normalize, **specs_kwargs)
        if stage == 'validate' or stage is None:
            print("preparing dataset, condition_on_spkemb: " + self.condition_on_spkemb)
            if  self.condition_on_spkemb == "no":
                self.valid_set = Specs(data_dir=self.base_dir, subset='dev',
                    dummy=True, shuffle_spec=False, format=self.format, only_enhancement=self.only_enhancement,
                    normalize=self.normalize, train_noisy_data = self.train_noisy_data,
                    **specs_kwargs)
            elif  self.condition_on_spkemb == "yes":
                self.valid_set = ConditionalSpecs(data_dir=self.base_dir, subset='dev',only_enhancement=self.only_enhancement,
                    dummy=True, shuffle_spec=False, return_time = self.return_time,format=self.format,
                    normalize=self.normalize, train_noisy_data = self.train_noisy_data,
                    **specs_kwargs)
        
        if stage == 'test' or stage is None:
            if  self.condition_on_spkemb == "no":
                self.test_set = Specs(data_dir=self.base_dir, subset='test',only_enhancement=self.only_enhancement,
                    dummy=self.dummy, shuffle_spec=False, format=self.format,
                    normalize=self.normalize, train_noisy_data = self.train_noisy_data,
                    **specs_kwargs)
            elif self.condition_on_spkemb == "yes":
                self.test_set = ConditionalSpecs(data_dir=self.base_dir, subset='test',only_enhancement=self.only_enhancement,
                    dummy=self.dummy, shuffle_spec=False, format=self.format,
                    normalize=self.normalize,return_time = self.return_time, 
                    train_noisy_data = self.train_noisy_data, **specs_kwargs)

    def spec_fwd(self, spec):
        if self.transform_type == "exponent":
            if self.spec_abs_exponent != 1:
                # only do this calculation if spec_exponent != 1, otherwise it's quite a bit of wasted computation
                # and introduced numerical error
                e = self.spec_abs_exponent
                spec = spec.abs()**e * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "log":
            spec = torch.log(1 + spec.abs()) * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "none":
            spec = spec
        return spec

    def spec_back(self, spec):
        if self.transform_type == "exponent":
            spec = spec / self.spec_factor
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                spec = spec.abs()**(1/e) * torch.exp(1j * spec.angle())
        elif self.transform_type == "log":
            spec = spec / self.spec_factor
            spec = (torch.exp(spec.abs()) - 1) * torch.exp(1j * spec.angle())
        elif self.transform_type == "none":
            spec = spec
        return spec

    @property
    def stft_kwargs(self):
        return {**self.istft_kwargs, "return_complex": True}

    @property
    def istft_kwargs(self):
        return dict(
            n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, center=True
        )

    def _get_window(self, x):
        """
        Retrieve an appropriate window for the given tensor x, matching the device.
        Caches the retrieved windows so that only one window tensor will be allocated per device.
        """
        window = self.windows.get(x.device, None)
        if window is None:
            window = self.window.to(x.device)
            self.windows[x.device] = window
        return window

    def stft(self, sig):
        window = self._get_window(sig)
        return torch.stft(sig, **{**self.stft_kwargs, "window": window})

    def istft(self, spec, length=None):
        window = self._get_window(spec)
        return torch.istft(spec, **{**self.istft_kwargs, "window": window, "length": length})

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False
        )
        
        
        
        
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
