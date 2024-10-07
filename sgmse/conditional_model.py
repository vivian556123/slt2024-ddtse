import time
from math import ceil
import warnings
from typing import List, Tuple, Union
from espnet2.enh.extractor.abs_extractor import AbsExtractor
from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.layers.tcn import ChannelwiseLayerNorm

import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage

from sgmse import conditional_sampling as sampling
from sgmse import conditional_sampling_DDTSE as sampling_DDTSE
from sgmse.sdes import SDERegistry
from sgmse.backbones import BackboneRegistry
from sgmse.util.inference import evaluate_model, evaluate_model_on_condition, evaluate_model_on_condition_DDPM, evaluate_model_on_condition_rir
from sgmse.util.other import pad_spec
import wespeakerruntime as wespeaker
import numpy as np
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from einops import rearrange, repeat
from diffusers import AutoencoderKL, DDPMScheduler
import matplotlib.pyplot as plt
import os
from sgmse.util.other import energy_ratios
from torch import nn
import torch.nn.functional as F

from sgmse.util.DDPM_utils import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from sgmse.util.DDPM_utils import make_beta_schedule, extract_into_tensor, noise_like
from typing import Optional
import numpy as np
import torchaudio
import onnxruntime as ort
import torchaudio.compliance.kaldi as kaldi
import random
import deepspeed

class ConditionalScoreModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum time (3e-2 by default)")
        parser.add_argument("--num_eval_files", type=int, default=20, help="Number of files for speech enhancement performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).")
        parser.add_argument("--loss_type", type=str, default="mse", choices=("mse", "mae"), help="The type of loss function to use.")
        parser.add_argument("--classifier_free", type=str, default="no", choices=("yes", "no"), help="Use classifier free guidance or not")
        parser.add_argument("--sisdr", type=float, default=0.0,  help="The weight of sisdr loss")
        parser.add_argument("--spk_encoder", action='store_true',   help="Use a speaker encoder (train simultaneously with the model) instead of pretrained SV net")

        return parser

    def __init__(
        self, backbone, sde, lr=1e-4, ema_decay=0.999, t_eps=3e-2,
        num_eval_files=20, loss_type='mse', data_module_cls=None,spk_encoder =False,
        **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: Backbone DNN that serves as a score-based model.
            sde: The SDE that defines the diffusion process.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            loss_type: The type of loss to use (wrt. noise z/std). Options are 'mse' (default), 'mae'
        """
        super().__init__()
        # Initialize Backbone DNN
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.dnn = dnn_cls(**kwargs)
        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.loss_type = loss_type
        self.num_eval_files = num_eval_files

        self.save_hyperparameters(ignore=['no_wandb'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)
        
        self.spk_emb_extractor=wespeaker.Speaker(lang='en')
        self.spk_encoder = spk_encoder

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.parameters())        # store current params in EMA
                self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _loss(self, err):
        if self.loss_type == 'mse':
            losses = torch.square(err.abs())
        elif self.loss_type == 'mae':
            losses = err.abs()
        # taken from reduce_op function: sum over channels and position and mean over batch dim
        # presumably only important for absolute loss number, not for gradients
        loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss

    def _step(self, batch, batch_idx):
        x, y, spk_emb, x_audio, y_audio, x_interference_audio,spk_emb_tgt, spk_emb_interf, prompt_stft, prompt_mel = batch
        t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
        mean, std = self.sde.marginal_prob(x, t, y)
        z = torch.randn_like(x)  # i.i.d. normal distributed with var=0.5
        sigmas = std[:, None, None, None]
        perturbed_data = mean + sigmas * z
        if self.spk_encoder:
            score = self(perturbed_data, t, y, prompt_mel)
        else:
            score = self(perturbed_data, t, y, spk_emb)
        
        
        err = score * sigmas + z
        loss = self._loss(err)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True)
        # print("val_dataloader = model.val_dataloader()")
        # Evaluate speech enhancement performance
        if batch_idx == 0 and self.num_eval_files != 0:
            # pesq = 0.1
            # si_sdr = 0.1
            # estoi = 0.1
            pesq, si_sdr, estoi = evaluate_model_on_condition(self, self.num_eval_files, spk_encoder = self.spk_encoder)
            self.log('pesq', pesq, on_step=False, on_epoch=True)
            self.log('si_sdr', si_sdr, on_step=False, on_epoch=True)
            self.log('estoi', estoi, on_step=False, on_epoch=True)
            print("pesq, si_sdr, estoi",pesq, si_sdr, estoi)

        return loss

    def forward(self, x, t, y, spk_emb):
        # Concatenate y as an extra channel
        dnn_input = torch.cat([x, y], dim=1)
        
        # the minus is most likely unimportant here - taken from Song's repo
        score = -self.dnn(dnn_input, t, spk_emb)
        return score
    
    def sample_forward(self, x, t, y, spk_emb):
        # Concatenate y as an extra channel
        dnn_input = torch.cat([x, y], dim=1)
                
        # the minus is most likely unimportant here - taken from Song's repo
        score = -self.dnn(dnn_input, t, spk_emb)
                
        return score

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_pc_sampler(self, predictor_name, corrector_name, y, condition, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y,condition = condition, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    condition_mini = condition[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y_mini, condition = condition_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    def get_ode_sampler(self, y, condition, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_ode_sampler(sde, self, y=y,condition=condition, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    condition_mini =condition[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_ode_sampler(sde, self, y=y_mini, condition=condition_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return sample, ns
            return batched_sampling_fn

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)

    def enhance(self, y, condition, sampler_type="pc", predictor="reverse_diffusion",
        corrector="ald", N=30, corrector_steps=1, snr=0.5, timeit=False,
        **kwargs
    ):
        """
        One-call speech enhancement of noisy speech `y`, for convenience.
        """
        sr=16000
        start = time.time()
        T_orig = y.size(1) 
        norm_factor = y.abs().max().item()
        y = y / norm_factor
        Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        if sampler_type == "pc":
            sampler = self.get_pc_sampler(predictor, corrector, Y.cuda(), condition.cuda(), N=N, 
                corrector_steps=corrector_steps, snr=snr, intermediate=False,
                **kwargs)
        elif sampler_type == "ode":
            sampler = self.get_ode_sampler(Y.cuda(), condition.cuda(), N=N, **kwargs)
        else:
            print("{} is not a valid sampler type!".format(sampler_type))
        sample, nfe = sampler()
        x_hat = self.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu().detach().numpy()
        end = time.time()
        if timeit:
            rtf = (end-start)/(len(x_hat)/sr)
            return x_hat, nfe, rtf
        else:
            return x_hat




class ConditionalDiscriminativeModel(ConditionalScoreModel):

    def forward(self, y, spk_emb):
        t = torch.ones(y.shape[0], device=y.device)
        return self.dnn(y, t, spk_emb)
    
    def sample_forward(self, y, spk_emb):
        t = torch.ones(y.shape[0], device=y.device)
        return self.dnn(y, t, spk_emb)

    def _step(self, batch, batch_idx):
        X, Y, spk_emb, x_audio, y_audio, x_interference_audio,spk_emb_tgt, spk_emb_interf, prompt_stft, prompt_mel = batch
        Xhat = self(Y, spk_emb)
        err = Xhat - X
        loss = self._loss(err)
        return loss

    def enhance(self, y, condition=None, timeit=False, **ignored_kwargs):
        sr=16000
        start = time.time()
        norm_factor = y.abs().max().item()
        T_orig = y.size(1)
        y = y / norm_factor

        if self.data_module.return_time:
            Y = torch.unsqueeze(y.cuda(), 0) #1,D=1,T
        else:
            Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0) #1,D,F,T
            Y = pad_spec(Y)
        X_hat = self(Y, condition.to(Y.device))

        if self.data_module.return_time:
            x_hat = X_hat.squeeze()
        else:
            x_hat = self.to_audio(X_hat.squeeze(), T_orig)
        x_hat = x_hat * norm_factor

        x_hat = x_hat.squeeze().cpu().detach().numpy()
        end = time.time()
        if timeit:
            rtf = (end-start)/(len(x_hat)/sr)
            return x_hat, .0, rtf
        else:
            return x_hat
        
        
    def enhance_twice(self, y, spk_emb, timeit=False, **ignored_kwargs):
        sr=16000
        start = time.time()
        norm_factor = y.abs().max().item()
        T_orig = y.size(1)
        y = y / norm_factor

        if self.data_module.return_time:
            Y = torch.unsqueeze(y.cuda(), 0) #1,D=1,T
        else:
            Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0) #1,D,F,T
            Y = pad_spec(Y)
        X_hat = self(Y, spk_emb.to(Y.device))
        X_hat = self(X_hat, spk_emb.to(Y.device))

        if self.data_module.return_time:
            x_hat = X_hat.squeeze()
        else:
            x_hat = self.to_audio(X_hat.squeeze(), T_orig)
        x_hat = x_hat * norm_factor

        x_hat = x_hat.squeeze().cpu().detach().numpy()
        end = time.time()
        if timeit:
            rtf = (end-start)/(len(x_hat)/sr)
            return x_hat, .0, rtf
        else:
            return x_hat
        
        
        
        
        


    
class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        #Input dimension: 
        # Anchor: (batch_size, embedding_size)
        # Positive: (batch_size, embedding_size)
        # Negative: (batch_size, embedding_size)

    def forward(self, anchor, positive, negative):
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negative = F.normalize(negative, dim=-1)

        # Compute cosine similarities
        pos_similarity = F.cosine_similarity(anchor, positive, dim= -1)
        neg_similarity = F.cosine_similarity(anchor, negative, dim= -1)

        # Compute triplet loss
        loss = torch.relu(self.margin - (pos_similarity - neg_similarity)).mean()

        return loss

def compute_fbank(waveform, sample_rate,
                    resample_rate: int = 16000,
                    num_mel_bins: int = 80,
                    frame_length: int = 25,
                    frame_shift: int = 10,
                    dither: float = 0.0,
                    cmn: bool = True):
    """ Extract fbank, simlilar to the one in wespeaker.dataset.processor,
        While integrating the wave reading and CMN.
        waveform: (channel, time)
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


def extract_spk_emb(spk_emb_extractor, waveform, sample_rate, wav_num=1):
    #print("waveform.dim()", waveform.dim(),waveform.shape,waveform.shape[0])
    if waveform.dim() == 3:
        spk_emb_list = []
        for i in range(waveform.shape[0]):
            waveform_process = waveform[i]
            feats = compute_fbank(waveform_process, sample_rate)
            feats = np.expand_dims(feats, 0)
            spk_emb_list.append(torch.tensor(spk_emb_extractor.extract_embedding_feat(feats)))
        spk_emb = torch.cat(spk_emb_list, dim=0)
    elif waveform.dim() == 2: 
        feats = compute_fbank(waveform, sample_rate)
        feats = np.expand_dims(feats, 0)
        spk_emb = torch.tensor(spk_emb_extractor.extract_embedding_feat(feats))
    else: 
        raise ValueError("The input waveform should be 2D or 3D tensor")
    
    return spk_emb.unsqueeze(1) # [batchsize,1,emb_dim]   
    

def si_sdr_components(s_hat, s, n):
    # Ensure input tensors have shape (batch_size, num_channels, signal_length)
    assert s_hat.dim() == 3 and s.dim() == 3 and n.dim() == 3, f"Input tensors should have shape (batch_size, num_channels, signal_length), but has shape {s_hat.shape}, {s.shape}, {n.shape}"
    
    # s_target
    alpha_s = torch.sum(s_hat * s, dim=(1, 2)) / torch.sum(s**2, dim=(1, 2))
    s_target = alpha_s.unsqueeze(1).unsqueeze(2) * s

    # e_noise
    alpha_n = torch.sum(s_hat * n, dim=(1, 2)) / torch.sum(n**2, dim=(1, 2))
    e_noise = alpha_n.unsqueeze(1).unsqueeze(2) * n

    # e_art
    e_art = s_hat - s_target - e_noise

    return s_target, e_noise, e_art

def training_sisdr_loss(s_hat, s, n):
    s_target, e_noise, e_art = si_sdr_components(s_hat, s, n)

    si_sdr = 10 * torch.log10(torch.sum(s_target**2, dim=(1, 2)) / torch.sum((e_noise + e_art)**2, dim=(1, 2)))
    loss = -torch.mean(si_sdr)
    return loss




class ConditionalDDTSEModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum time (3e-2 by default)")
        parser.add_argument("--num_eval_files", type=int, default=20, help="Number of files for speech enhancement performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).")
        parser.add_argument("--loss_type", type=str, default="mse", choices=("mse", "mae"), help="The type of loss function to use.")
        parser.add_argument("--sisdr", type=float, default=0.0,  help="The weight of sisdr loss")
        parser.add_argument("--triplet", type=float, default=0.0,  help="The weight of triplet loss")
        parser.add_argument("--suppression_loss", type=float, default=0.0,  help="The weight of triplet loss")
        parser.add_argument("--spk_encoder", action='store_true',   help="Use a speaker encoder (train simultaneously with the model) instead of pretrained SV net")
        parser.add_argument("--inference_step", type=int, default=10, help="inference step for DDTSE")
        parser.add_argument("--teacher_forcing", action='store_true',   help="Use teacher forcing and introduce xt-pred into training")
        parser.add_argument("--max_threshold", type=float, default=1.0,  help="teacher forcing and max threshold")
        parser.add_argument("--TF_version2", action='store_true',     help="teacher forcing for stage 2")
        parser.add_argument("--valid_rir", action='store_true',     help="add reverberation or not")
        parser.add_argument("--speaker_encoder_type", type=str, default="BLSTM",  help="type of speaker encoder")


        return parser

    def __init__(
        self, backbone, sde, lr=1e-4, ema_decay=0.999, t_eps=3e-2,
        num_eval_files=20, loss_type='mse', sisdr = 0.0,triplet=0.0, suppression_loss=0.0, data_module_cls=None,  
        spk_encoder=False,inference_step = 10, 
        teacher_forcing = False, max_threshold=1.0, TF_version2=False, deepspeed_optimizer=False, valid_rir=False, **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: Backbone DNN that serves as a score-based model.
            sde: The SDE that defines the diffusion process.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            loss_type: The type of loss to use (wrt. noise z/std). Options are 'mse' (default), 'mae'
        """
        super().__init__()
        # Initialize Backbone DNN
        self.spk_encoder = spk_encoder
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.dnn = dnn_cls(**kwargs)
        #self.dnn.to("cuda")
        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.loss_type = loss_type
        self.num_eval_files = num_eval_files

        self.save_hyperparameters(ignore=['no_wandb'])
        print("data_module_cls", data_module_cls)
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)
        
        self.sisdr = sisdr
        self.triplet = triplet
        if self.triplet > 0.0:
            self.triplet_loss_calculator = TripletLoss(margin=0.2)
            self.spk_emb_extractor=wespeaker.Speaker(lang='en')
        self.spk_encoder = spk_encoder
        
        self.inference_step = inference_step
        self.teacher_forcing = teacher_forcing
        self.max_threshold=max_threshold
        self.deepspeed_optimizer = deepspeed_optimizer
        self.TF_version2 = TF_version2
        self.rir = valid_rir
    
    def configure_optimizers(self):
        if self.deepspeed_optimizer:
            optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(self.parameters(), lr=self.lr)
        else:    
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.parameters())        # store current params in EMA
                self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _loss(self, err, lt = None):
        if self.loss_type == 'mse':
            losses = torch.square(err.abs())
        elif self.loss_type == 'mae':
            losses = err.abs()
        # taken from reduce_op function: sum over channels and position and mean over batch dim
        # presumably only important for absolute loss number, not for gradients
        if lt != None: 
            losses = losses * lt.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss

    def get_perturbed_data_basedon_pred_x0(self, x_pred, t, y):
        # Implement any transformations or manipulations on the output1 here
        mean, std = self.sde.marginal_prob(x_pred, t, y)
        z = torch.randn_like(x_pred)  # i.i.d. normal distributed with var=0.5
        sigmas = std[:, None, None, None]
        perturbed_data = mean + sigmas * z
        return perturbed_data
    
    def _step(self, batch, batch_idx):
        #if self.triplet > 0:
        x, y, spk_emb, x_audio, y_audio, x_interference_audio,spk_emb_tgt, spk_emb_interf, prompt_stft, prompt_mel = batch
        t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
        random_value = random.random() 
        threshold = min(self.current_epoch / 100, self.max_threshold)
        threshold_pred = threshold / 2
        if self.teacher_forcing and random_value < threshold:
            if self.TF_version2:
                xT = self.sde.prior_sampling(y.shape, y)
                x_pred = self(xT, torch.ones_like(t), y, spk_emb)
                score = x_pred
            else:
                if random_value < threshold_pred :
                    with torch.no_grad():
                        xT = self.sde.prior_sampling(y.shape, y)
                        x_pred = self(xT, torch.ones_like(t), y, spk_emb)
                    perturbed_data = self.get_perturbed_data_basedon_pred_x0(x_pred, t, y)
                    score = self(perturbed_data, t, y, spk_emb)
                else: 
                    xT = self.sde.prior_sampling(y.shape, y)
                    x_pred = self(xT, torch.ones_like(t), y, spk_emb)
                    score = x_pred
        else: 
            mean, std = self.sde.marginal_prob(x, t, y)
            z = torch.randn_like(x)  # i.i.d. normal distributed with var=0.5
            sigmas = std[:, None, None, None]
            perturbed_data = mean + sigmas * z
            if self.spk_encoder:
                score = self(perturbed_data, t, y, prompt_mel)
            else:
                score = self(perturbed_data, t, y, spk_emb)
        #print("x.shape",x.shape,"score",score.shape)
        err = x - score
        lt = 1/(torch.exp(t) -1) # Decreasing function of t
        if self.sisdr > 0.0 or self.triplet > 0.0:
            sisdr_loss = 0.0
            triplet_loss = 0.0
            sr = 16000
            pred_audio = torch.tensor(self.to_audio(score.squeeze())) #of shape [batchsize, T]
            if pred_audio.dim() == 2:
                pred_audio = pred_audio.unsqueeze(1)
            elif pred_audio.dim() == 1:
                pred_audio = pred_audio.unsqueeze(0).unsqueeze(0)
            sisdr_loss = training_sisdr_loss(pred_audio, x_audio, (y_audio - x_audio))
            if self.triplet > 0.0:
                with torch.no_grad():
                    spk_emb_pred = extract_spk_emb(self.spk_emb_extractor, pred_audio.cpu(), sr).to(x.device)
                triplet_loss = self.triplet_loss_calculator(spk_emb_tgt.squeeze(1), spk_emb.squeeze(1), spk_emb_interf.squeeze(1)) + self.triplet_loss_calculator(spk_emb_tgt.squeeze(1), spk_emb_pred.squeeze(1), spk_emb_interf.squeeze(1)) + self.triplet_loss_calculator(spk_emb_pred.squeeze(1), spk_emb_tgt.squeeze(1), spk_emb_interf.squeeze(1)) 
            loss = self._loss(err, lt) + self.sisdr * torch.tensor(sisdr_loss).to(err.device) + self.triplet * triplet_loss
            return loss, sisdr_loss, triplet_loss

        else: 
            loss = self._loss(err, lt)
            return loss, 0.0, 0.0

    def training_step(self, batch, batch_idx):
        loss, sisdr_loss, triplet_loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('sisdr_loss', sisdr_loss, on_step=True, on_epoch=True)
        self.log('triplet_loss', triplet_loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, sisdr_loss, triplet_loss  = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True)
        self.log('valid_sisdr_loss', sisdr_loss, on_step=True, on_epoch=True)
        self.log('valid_triplet_loss', triplet_loss, on_step=True, on_epoch=True)

        # Evaluate speech enhancement performance
        if batch_idx == 0 and self.num_eval_files != 0:
            if self.rir:
                pesq, si_sdr, estoi = evaluate_model_on_condition_rir(self, self.num_eval_files)
            pesq, si_sdr, estoi = evaluate_model_on_condition(self, self.num_eval_files, self.spk_encoder)
            self.log('pesq', pesq, on_step=False, on_epoch=True)
            self.log('si_sdr', si_sdr, on_step=False, on_epoch=True)
            self.log('estoi', estoi, on_step=False, on_epoch=True)
            print("pesq, si_sdr, estoi",pesq, si_sdr, estoi)

        return loss

    def forward(self, x, t, y, spk_emb):

        # Concatenate y as an extra channel
        dnn_input = torch.cat([x, y], dim=1)
        # the minus is most likely unimportant here - taken from Song's repo
        score = self.dnn(dnn_input, t, spk_emb)

        
        return score
    
    def sample_forward(self, x, t, y, spk_emb):
        # Concatenate y as an extra channel
        dnn_input = torch.cat([x, y], dim=1)
        
        # the minus is most likely unimportant here - taken from Song's repo
        score = self.dnn(dnn_input, t, spk_emb)
        
        
        mean, std = self.sde.marginal_prob(score, t, y)
        std = torch.tensor(max(std.squeeze().cpu().numpy(), 0.1)).to(y.device)
        score =  -3*((x-y) - torch.exp(-t*1.5) * (score-y) ) / (1 - torch.exp(-3*t)) /(std**2)
        
        return score

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_pc_sampler(self, predictor_name, corrector_name, y, condition, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling_DDTSE.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y,condition = condition, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    condition_mini = condition[i*minibatch:(i+1)*minibatch]
                    sampler = sampling_DDTSE.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y_mini, condition = condition_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    def get_ode_sampler(self, y, condition, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling_DDTSE.get_ode_sampler(sde, self, y=y,condition=condition, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    condition_mini =condition[i*minibatch:(i+1)*minibatch]
                    sampler = sampling_DDTSE.get_ode_sampler(sde, self, y=y_mini, condition=condition_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return sample, ns
            return batched_sampling_fn

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)

    def enhance(self, y, condition, sampler_type="pc", predictor="reverse_diffusion",
        corrector="ald", N=30, corrector_steps=1, snr=0.5, timeit=False,x = None,
        infer_step = None, recursive_infer = 1,
        **kwargs
    ):
        """
        One-call speech enhancement of noisy speech `y`, for convenience.
        """
        N = 30
        sr=16000
        start = time.time()
        T_orig = y.size(1) 
        norm_factor = y.abs().max().item()
        y = y / norm_factor
        Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        
          
        
        # Improved Sample only in the first step, and following training process in the following steps
        if x == None:
            latents = self.sde.prior_sampling(Y.shape, Y).to(Y.device)
        else:
            latents = x
        eps=3e-2
        if infer_step == None:
            inference_step = self.inference_step # Default 10
        else: 
            inference_step = infer_step
        timesteps = torch.linspace(self.sde.T, eps, inference_step, device=Y.device)
        
        if recursive_infer > 1:
            timesteps = timesteps.repeat(recursive_infer)
            inference_step = inference_step * recursive_infer
        
        
        #print("timesteps",timesteps)
        with torch.no_grad():
            if x == None:
                latents = self(latents, torch.tensor(1).unsqueeze(0).to(Y.device), Y, condition.to(Y.device))
            for i in range(inference_step):
                t = timesteps[i].unsqueeze(0).to(Y.device)
                mean, std = self.sde.marginal_prob(latents, t, Y)
                z = torch.randn_like(Y)  # i.i.d. normal distributed with var=0.5
                sigmas = std[:, None, None, None]
                perturbed_data = mean + sigmas * z
                latents = self(perturbed_data.to(Y.device), t, Y, condition.to(Y.device))

        X_hat = latents
        if self.data_module.return_time:
            x_hat = X_hat.squeeze()
        else:
            x_hat = self.to_audio(X_hat.squeeze(), T_orig)
        x_hat = x_hat * norm_factor

        x_hat = x_hat.squeeze().cpu().detach().numpy()
        end = time.time()
        if timeit:
            rtf = (end-start)/(len(x_hat)/sr)
            return x_hat, .0, rtf
        else:
            return x_hat





class Conv1dBlock(nn.Module):
    def __init__(self,
                in_channels: int = 128,
                out_channels: int = 257,
                ):
        super(Conv1dBlock, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.norm = ChannelwiseLayerNorm(out_channels, shape="BDT")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1d(x)
        x = self.norm(x)

        return x

class Conv2dBlock(nn.Module):
    def __init__(self, 
                in_dims: int = 16,
                out_dims: int = 32,
                kernel_size: Tuple[int] = (3, 3),
                stride: Tuple[int] = (1, 1),
                padding: Tuple[int] = (1, 1)) -> None:
        super(Conv2dBlock, self).__init__() 
        self.conv2d = nn.Conv2d(in_dims, out_dims, kernel_size, stride, padding)     
        self.elu = nn.ELU()
        self.norm = nn.InstanceNorm2d(out_dims)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv2d(x)
        x = self.elu(x)
        
        return self.norm(x)


