import glob
from argparse import ArgumentParser
from os.path import join

import torch
from soundfile import write
from torchaudio import load
import torchaudio
from tqdm import tqdm
import torch.nn.functional as F
import time

from sgmse.data_module import mel_spectrogram
from sgmse.model import ScoreModel, DiscriminativeModel
from sgmse.conditional_model import ConditionalScoreModel, ConditionalDiscriminativeModel, ConditionalDDTSEModel

from sgmse.util.other import ensure_dir, pad_spec
import random
import wespeakerruntime as wespeaker
import numpy as np
from sgmse.util.other import energy_ratios, mean_std

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test_dir", type=str, default="/LibriMix/Libri2Mix_mel_data/Libri2Mix/wav16k/min/test", help='Directory containing the test data (must have subdirectory noisy/)')
    parser.add_argument("--clean_wav_dir", type=str, default = "no", help='Directory containing the test data')
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory containing the enhanced data')
    parser.add_argument("--ckpt", type=str,  help='Path to model checkpoint.')
    parser.add_argument("--corrector", type=str, choices=("ald", "langevin", "none"), default="ald", help="Corrector class for the PC sampler.")
    parser.add_argument("--corrector_steps", type=int, default=1, help="Number of corrector steps")
    parser.add_argument("--snr", type=float, default=0.5, help="SNR value for (annealed) Langevin dynmaics.")
    parser.add_argument("--N", type=int, default=30, help="Number of reverse steps")
    parser.add_argument("--train_or_test", type=str, choices=("train-100","train-360", "test"), default="test", help="Number of reverse steps")
    parser.add_argument("--condition", type=str, choices=("no", "yes"), default="yes", help="no for Spec, yes for ConditionalSpec")
    parser.add_argument("--discriminatively", type=str, choices=("no", "yes"), default="no",  help="Train the backbone as a discriminative model instead")  
    parser.add_argument("--DDPM", type=str,  default="DDTSE",  help="model configuration")  
    parser.add_argument("--only_enhanDDTSEent", type=str,  default="no",  help="single-speaker enhanDDTSEent")  
    parser.add_argument("--voicebank", type=str,  default="no",  help="Use voicebank-demand for test")  
    parser.add_argument("--infer_step", type=int, default=10, help="Number of inference steps for DDTSE")
    parser.add_argument("--recursive_infer", type=int, default=1, help="Number of recursion of inference for DDTSE")
    parser.add_argument("--seed", type=int, default=20, help="random seed")
    parser.add_argument("--model_sr", type=int, default=16000, help="sample rate of traning data for model")
    parser.add_argument("--speaker_encoder",action='store_true', help="use a speaker encoder to extract speaker embedding instead of a pretrained model")


    
    args = parser.parse_args()
    
    setup_seed(args.seed)


    if (args.condition == "no" and args.only_enhanDDTSEent == "no" ) or args.voicebank=="yes":
        noisy_dir = args.test_dir
        clean_dir = noisy_dir
        prompt_dir = noisy_dir
    else:
        if args.clean_wav_dir=="no" and args.only_enhanDDTSEent == "no":
            noisy_dir = join(args.test_dir, 'mix_both/')
            clean_dir = join(args.test_dir, 's1/')
            prompt_dir = join(args.test_dir, 's1_prompt/')
        elif args.clean_wav_dir=="mix_clean":
            noisy_dir = join(args.test_dir, 'mix_clean/')
            clean_dir = join(args.test_dir, 's1/')
            prompt_dir = join(args.test_dir, 's1_prompt/')
        elif args.only_enhanDDTSEent == "yes" or args.only_enhanDDTSEent == "pse":
            noisy_dir = join(args.test_dir, 'mix_single/')
            clean_dir = join(args.test_dir, 's1/')
            prompt_dir = join(args.test_dir, 's1_prompt/')
        else:
            noisy_dir = args.test_dir
            clean_dir = join(args.clean_wav_dir, 's1/')
            prompt_dir = join(args.clean_wav_dir, 's1_prompt/')
    print("noisy_dir",noisy_dir)
    checkpoint_file = args.ckpt
    corrector_cls = args.corrector

    target_dir = args.enhanced_dir
    print("target_dir",target_dir)
    ensure_dir(target_dir)
    

    # Settings
    sr = args.model_sr
    
    snr = args.snr
    N = args.N
    corrector_steps = args.corrector_steps

    # Load score model
    print("model loading...")
    if args.algorithm_type=="DDTSE":
        model = ConditionalDDTSEModel.load_from_checkpoint(checkpoint_file, base_dir='', batch_size=16, num_workers=0, kwargs=dict(gpu=False))
    else: 
        if args.condition == "no" and args.discriminatively == "no": 
            model = ScoreModel.load_from_checkpoint(checkpoint_file, base_dir='', batch_size=16, num_workers=0, kwargs=dict(gpu=False))
        elif args.condition == "no" and args.discriminatively == "yes":
            model = DiscriminativeModel.load_from_checkpoint(checkpoint_file, base_dir='', batch_size=16, num_workers=0, kwargs=dict(gpu=False))

        elif args.condition == "yes" and args.discriminatively == "no":
            model = ConditionalScoreModel.load_from_checkpoint(checkpoint_file, base_dir='', batch_size=16, num_workers=0, kwargs=dict(gpu=False))
        elif args.condition == "yes" and args.discriminatively == "yes":
            model = ConditionalDiscriminativeModel.load_from_checkpoint(checkpoint_file, base_dir='', batch_size=16, num_workers=0, kwargs=dict(gpu=False))
    
    model.eval(no_ema=False)
    model.cuda()
    print("Model.device",model.device)

    if  args.condition == "no": 
        noisy_files = sorted(glob.glob('{}/*.wav'.format(noisy_dir)))
        print(len(noisy_files))
    else:
        noisy_files = sorted(glob.glob('{}/*.wav'.format(noisy_dir)))

    rtf_total_list = []

    noisy_files = noisy_files[:55]
    for noisy_file in tqdm(noisy_files):
        filename = noisy_file.split('/')[-1]
        
        # Load wav
        y, ori_sr = load(noisy_file) 
                   
        
        if ori_sr != sr:    
            print("ori_sr ",ori_sr, "!= sr ",sr)
            y = torchaudio.transforms.Resample(orig_freq=ori_sr, new_freq=sr)(y)

        T_orig = y.size(1)   

        # Normalize
        norm_factor = y.abs().max()
        y = y / norm_factor
        
        # Prepare DNN input
        Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        #print("Y.device",Y.device)
        
        
        # spk embedding extraction
        if args.condition == "yes" and args.only_enhanDDTSEent == "no": 
            spk_emb_extractor=wespeaker.Speaker(lang='en')
            prompt_file = join(prompt_dir,filename)
            spk_emb = torch.tensor(spk_emb_extractor.extract_embedding(prompt_file)).unsqueeze(1).to(Y.device)
            
            if args.speaker_encoder:
                z, _ = load(prompt_file) 
                spk_emb = mel_spectrogram(z, n_fft=1024, num_mels=80, sampling_rate=16000,
                            hop_size=256, win_size=1024, fmin=0, fmax=8000).to(Y.device)
            
        elif args.only_enhanDDTSEent == "yes":
            spk_emb_extractor=wespeaker.Speaker(lang='en')
            spk_emb = torch.tensor(spk_emb_extractor.extract_embedding(noisy_file)).unsqueeze(1).to(Y.device)
        elif args.only_enhanDDTSEent == "pse":
            spk_emb_extractor=wespeaker.Speaker(lang='en')
            prompt_file = join(prompt_dir,filename)
            #prompt_file = join(clean_dir,"1089-134691-0020_908-31957-0004.wav")
            spk_emb = torch.tensor(spk_emb_extractor.extract_embedding(prompt_file)).unsqueeze(1).to(Y.device)
        else:
            spk_emb = None
            
        #spk_emb = torch.randn_like(spk_emb)
        
        if args.discriminatively == "no" and args.algorithm_type=="no":
            # Reverse sampling
            if args.condition == "no": 
                sampler = model.get_pc_sampler(
                    'reverse_diffusion', corrector_cls, Y.cuda(), N=N, 
                    corrector_steps=corrector_steps, snr=snr)
            elif args.condition == "yes": 
                sampler = model.get_pc_sampler(
                    'reverse_diffusion', corrector_cls, Y.cuda(), condition=spk_emb, N=N, 
                    corrector_steps=corrector_steps, snr=snr)
            sample, _ = sampler()
            
            # reuse this sample to do something
            
            
            # Backward transform in time domain
            x_hat = model.to_audio(sample.squeeze(), T_orig)

            # Renormalize
            x_hat = x_hat * norm_factor

            # Write enhanced wav file
            x_hat = x_hat.squeeze().cpu().detach().numpy()
            
        else:
                start_time = time.time()
                if args.algorithm_type=="DDTSE":
                    x_hat = model.enhance(y = y.to(Y.device),condition= spk_emb.to(Y.device), infer_step = args.infer_step, recursive_infer = args.recursive_infer)
                else: 
                    x_hat = model.enhance(y = y.to(Y.device),condition= spk_emb.to(Y.device))
                process_time = time.time() - start_time
                audio_length = x_hat.shape[-1] / sr
                rtf = process_time / audio_length
                #print("rtf", rtf)
                rtf_total_list.append(rtf)
                x_hat = np.squeeze(x_hat)
                
        
        write(join(target_dir, filename), x_hat, sr)
    
    rtf_total_list = rtf_total_list[5:]
    rtf = np.mean(rtf_total_list)
    print("rtf_average", rtf)