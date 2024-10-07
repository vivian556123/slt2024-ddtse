# pylint: skip-file

from .ncsnpp_utils import layers, layerspp, normalization
import torch.nn as nn
import functools
import torch
import numpy as np

from .shared import BackboneRegistry
from collections import OrderedDict
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torch_complex.tensor import ComplexTensor

from espnet2.enh.extractor.abs_extractor import AbsExtractor
from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.layers.tcn import ChannelwiseLayerNorm

default_init = layers.default_init
ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
ResnetBlockConditionalBigGAN = layerspp.ResnetBlockConditionalBigGANpp
ResnetBlockConditionalConcatBigGAN = layerspp.ResnetBlockConditionalConcatBigGANpp
ResnetBlockConditionalConcatAttnBigGAN = layerspp.ResnetBlockConditionalConcatAttnBigGANpp
ResnetBlockConditionalFiLMBigGAN = layerspp.ResnetBlockConditionalFiLMBigGANpp
ResnetBlockConditionalCrossAttnBigGAN = layerspp.ResnetBlockConditionalCrossAttnBigGANpp

Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init

class DPCCNExtractor(AbsExtractor):
    def __init__(
        self,
        input_dim: int,
        kernel_size: Tuple[int] = (3, 3),
        stride1: Tuple[int] = (1, 1),
        stride2: Tuple[int] = (1, 2),
        paddings: Tuple[int] = (1, 0),
        paddings_spk: Tuple[int] = (0, 1),
        output_padding: Tuple[int] = (0, 0),
        tcn_dims: int = 384,
        tcn_blocks: int = 10,
        tcn_layers: int = 2,
        causal: bool = False,
        channels: int = 2,     
        pool_size: Tuple[int] = (4, 8, 16, 32),
        aux_hidden_dim: int = 128,
    ):
        super().__init__()
        self.dpccn = DenseUNet(
            kernel_size=kernel_size,
            stride1=stride1,
            stride2=stride2,
            paddings=paddings,
            paddings_spk=paddings_spk,
            output_padding=output_padding,
            tcn_dims=tcn_dims,
            tcn_blocks=tcn_blocks,
            tcn_layers=tcn_layers,
            causal=causal,
            channels=channels,
            pool_size=pool_size,
            aux_input_dim=input_dim,
            aux_hidden_dim=aux_hidden_dim,
        )

        self.channels = channels

    def forward(
        self,
        input: Union[torch.Tensor, ComplexTensor],
        ilens: torch.Tensor,
        input_aux: torch.Tensor,
        ilens_aux: torch.Tensor,
        suffix_tag: str = "",
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        if is_complex(input):
            real = input.real
            imag = input.imag
            feature = torch.stack([real, imag], dim=1)
        else:
            feature = input.unsqueeze(1)

        aux_feature = abs(input_aux) if is_complex(input_aux) else input_aux
        
        out = self.dpccn(feature, aux_feature)
        
        if self.channels == 2:
            F2 = out.shape[-1]
            masked = ComplexTensor(out[..., :F2//2], out[..., F2//2:])
        else:
            masked = out

        others = {
            "enroll_emb{}".format(suffix_tag): aux_feature.detach(),
        }

        return masked, ilens, others


class Conv1dBlock(nn.Module):
    def __init__(self,
                in_channels: int = 128,
                out_channels: int = 257,
                ):
        super(Conv1dBlock, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.norm = ChannelwiseLayerNorm(out_channels, shape="BDT")

    def forward(self, input_encoder) -> torch.Tensor:
        if isinstance(input_encoder, torch.Tensor):
            x = input_encoder
        else: 
            x = input_encoder['x']
        x = self.conv1d(x)
        x = self.norm(x)

        return x

class Conv2dBlock(nn.Module):
    def __init__(self, 
                in_dims  = 16,
                out_dims = 32,
                kernel_size  = (3, 3),
                stride  = (1, 1),
                padding  = (1, 1)):
        super(Conv2dBlock, self).__init__() 
        self.conv2d = nn.Conv2d(in_dims, out_dims, kernel_size, stride, padding)     
        self.elu = nn.ELU()
        self.norm = nn.InstanceNorm2d(out_dims)
        
    def forward(self, input_encoder):
        if isinstance(input_encoder, torch.Tensor):
            x = input_encoder
        else: 
            x = input_encoder['x']
        #print("Conv2dBlock input",x.shape)
        x = self.conv2d(x)
        x = self.elu(x)
        #print("Conv2dBlock Output",x.shape,self.norm(x).shape)
        return self.norm(x)


class ConvTrans2dBlock(nn.Module):
    def __init__(self, 
                in_dims = 32,
                out_dims  = 16,
                kernel_size   = (3, 3),
                stride  = (1, 2),
                padding = (1, 0),
                output_padding  = (0, 0)) :
        super(ConvTrans2dBlock, self).__init__() 
        self.convtrans2d = nn.ConvTranspose2d(in_dims, out_dims, kernel_size, stride, padding, output_padding)     
        self.elu = nn.ELU()
        self.norm = nn.InstanceNorm2d(out_dims)
        
    def forward(self, input_encoder):
        if isinstance(input_encoder, torch.Tensor):
            x = input_encoder
        else: 
            x = input_encoder['x']
        #print("ConvTrans2dBlock input",x.shape)
        x = self.convtrans2d(x)
        x = self.elu(x)
        #print("ConvTrans2dBlock Output",x.shape,self.norm(x).shape)
        return self.norm(x)
    
    
class DenseBlock(nn.Module):
    def __init__(self, in_dims, out_dims, mode = "enc", temb_dim = None, **kargs):
        super(DenseBlock, self).__init__()
        if mode not in ["enc", "dec"]:
            raise RuntimeError("The mode option must be 'enc' or 'dec'!")
            
        n = 1 if mode == "enc" else 2
        self.conv1 = Conv2dBlock(in_dims=in_dims*n, out_dims=in_dims, **kargs)
        self.conv2 = Conv2dBlock(in_dims=in_dims*(n+1), out_dims=in_dims, **kargs)
        self.conv3 = Conv2dBlock(in_dims=in_dims*(n+2), out_dims=in_dims, **kargs)
        self.conv4 = Conv2dBlock(in_dims=in_dims*(n+3), out_dims=in_dims, **kargs)
        self.conv5 = Conv2dBlock(in_dims=in_dims*(n+4), out_dims=out_dims, **kargs)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, in_dims)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)
            self.act = get_act('swish')
        
    def forward(self, input_encoder) -> torch.Tensor:
        if isinstance(input_encoder, torch.Tensor):
            x = input_encoder
            temb = None
        else: 
            x = input_encoder['x']
            temb = input_encoder['temb']
        y1 = self.conv1(x)
        if temb is not None:
            y1 += self.Dense_0(self.act(temb))[:, :, None, None]
        y2 = self.conv2(torch.cat([x, y1], 1))
        if temb is not None:
            y2 += self.Dense_0(self.act(temb))[:, :, None, None]
        y3 = self.conv3(torch.cat([x, y1, y2], 1))
        if temb is not None:
            y3 += self.Dense_0(self.act(temb))[:, :, None, None]
        y4 = self.conv4(torch.cat([x, y1, y2, y3], 1))
        if temb is not None:
            y4 += self.Dense_0(self.act(temb))[:, :, None, None]
        y5 = self.conv5(torch.cat([x, y1, y2, y3, y4], 1))
        
        #print("y1.y2.y3,t4,y5", y1.shape, y2.shape,y3.shape,y4.shape,y5.shape)
        return y5
    
        
class TCNBlock(nn.Module):
    """
    TCN block:
        IN - ELU - Conv1D - IN - ELU - Conv1D
    """

    def __init__(self,
                in_dims: int = 384,
                out_dims: int = 384,
                kernel_size: int = 3,
                stride: int = 1,
                paddings: int = 1,
                dilation: int = 1,
                causal: bool = False,
                temb_dim = None) -> None:
        super(TCNBlock, self).__init__()
        self.norm1 = nn.InstanceNorm1d(in_dims)
        self.elu1 = nn.ELU()
        dconv_pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))
        # dilated conv
        self.dconv1 = nn.Conv1d(
            in_dims,
            out_dims,
            kernel_size,
            padding=dconv_pad,
            dilation=dilation,
            groups=in_dims,
            bias=True)
        
        self.norm2 = nn.InstanceNorm1d(in_dims)
        self.elu2 = nn.ELU()    
        self.dconv2 = nn.Conv1d(in_dims, out_dims, 1, bias=True)
        
        # different padding way
        self.causal = causal
        self.dconv_pad = dconv_pad
        
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_dims)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)
            self.act = get_act('swish')

    def forward(self, input_encoder) -> torch.Tensor:
        if isinstance(input_encoder, torch.Tensor):
            x = input_encoder
        else: 
            x = input_encoder['x']
        y = self.elu1(self.norm1(x))
        
        y = self.dconv1(y)
        if self.causal:
            y = y[:, :, :-self.dconv_pad]
        y = self.elu2(self.norm2(y))
        y = self.dconv2(y)    
        x = x + y

        return x 
   
class GaussianFourierProjection(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""

  def __init__(self, embedding_size=256, scale=1.0):
    super().__init__()
    self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

  def forward(self, x):
    #print("x.device",x.device, self.W.device)
    x_proj = x[:, None] * self.W[None, :].to(x.device) * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
   
    
@BackboneRegistry.register("dpccn")    
class DenseUNet(nn.Module):
    def __init__(self,
                kernel_size: Tuple[int] = (3, 3),
                stride1: Tuple[int] = (1, 1),
                stride2: Tuple[int] = (1, 2),
                paddings: Tuple[int] = (1, 0),
                paddings_conv2d: Tuple[int] = (1, 1),
                paddings_spk: Tuple[int] = (1, 1),
                output_padding: Tuple[int] = (0, 0),
                tcn_dims: int = 384,
                tcn_blocks: int = 10,
                tcn_layers: int = 2,
                causal: bool = False,     
                pool_size: Tuple[int] = (4, 8, 16, 32),
                channels: int = 2,
                # enrollment related arguments
                aux_input_dim: int = 256,
                aux_hidden_dim: int = 128,
                embedding_type = 'fourier',
                nf = 128,
                fourier_scale = 16,
                discriminative = False,
                conditional = True, 
                scale_by_sigma = False,
                spk_encoder_backbone = False,
                **unused_kwargs
                ):
        super(DenseUNet, self).__init__()
        
        self.discriminative = discriminative
        if self.discriminative:
            # overwrite options that make no sense for a discriminative model
            conditional = False
            scale_by_sigma = False
            self.temb_dim = None
        else:
            self.temb_dim = 4* nf
            
        self.channels = channels
        
        #self.stft = ConvSTFT(win_len, win_inc, fft_len, win_type, 'complex')
        if conditional:
            self.conv2d = nn.Conv2d(2*channels, 16, kernel_size, stride1, paddings)
        else: 
            self.conv2d = nn.Conv2d(channels, 16, kernel_size, stride1, paddings)
        
        self.encoder = self._build_encoder(
                    kernel_size=kernel_size,
                    stride=stride2,
                    padding=paddings
                )
        self.tcn_layers = self._build_tcn_layers(
                    tcn_layers,
                    tcn_blocks,
                    in_dims=tcn_dims,
                    out_dims=tcn_dims,
                    causal=causal,
                )
        if self.channels == 2:
            #print("self.channels == 2, build_decoder")
            self.decoder = self._build_decoder(
                        kernel_size=kernel_size,
                        stride=stride2,
                        padding=paddings,
                        output_padding=(0,1)
                        #output_padding=output_padding
                    )
        else:
            #print("self.channels == 2, _build_mel_decoder")
            self.decoder = self._build_mel_decoder(
                        kernel_size=kernel_size,
                        stride=stride2,
                        padding=paddings,
                        output_padding=output_padding
                    )
        self.avg_pool = self._build_avg_pool(pool_size)
        self.avg_proj = nn.Conv2d(64, 32, 1, 1)

        #self.speaker_encoder = self._build_speaker_encoder(aux_input_dim, aux_hidden_dim, kernel_size, stride1, paddings_spk)
        
        self.deconv2d = nn.ConvTranspose2d(32, channels, kernel_size, stride1, (1,0))
        #self.istft = ConviSTFT(win_len, win_inc, fft_len, win_type, 'complex')
        
        
        self.conditional=conditional
        self.scale_by_sigma=scale_by_sigma
        self.embedding_type = embedding_type
        self.nf = nf

        self.fourier_scale = fourier_scale
        # timestep/noise_level embedding
        self.time_encoder = self._build_temb_encoder(conditional = conditional)
        #print("self.time_encoder",self.time_encoder)
        self.act = get_act('swish')
        
        self.spk_encoder_backbone = spk_encoder_backbone
        if self.spk_encoder_backbone: 
            self.speaker_encoder_model = layerspp.SpeakerEncoder()

        print("DPCCN", "self.discriminative",self.discriminative, "conditional",self.conditional, "spk_encoder", self.spk_encoder_backbone)
            
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--centered", dest="centered", action="store_true", help="The data is centered [-1, 1]")
        parser.add_argument("--resblock_type", type=str, default="conditional_biggan", 
                            choices=("ddpm", "biggan","conditional_biggan","conditional_concat_biggan",
                                     "conditional_concat_biggan_attn","conditional_film_biggan",
                                     "conditional_film_biggan_cross_attn","conditional_biggan_cross_attn"), 
                            help="The type of resblock.")
        parser.add_argument("--middle_concat_attention", type=bool, default=False, 
                            help="middle block, concat the condition embedding before attention")

        parser.set_defaults(centered=True)
        return parser
        
    
    def _build_temb_encoder(self, conditional):
        """
        Build time encoder layers 
        """
        print("self.embedding_type",self.embedding_type, "conditional", conditional)
        time_encoder = nn.ModuleList()
        if self.embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            time_encoder.append(GaussianFourierProjection(embedding_size=self.nf, scale=self.fourier_scale))
            embed_dim = 2 * self.nf
        elif self.embedding_type == 'positional':
            embed_dim = self.nf
        else:
            raise ValueError(f'embedding type {self.embedding_type} unknown.')
        #print("time encoder after embedding type", time_encoder)
        if conditional:
            time_encoder.append(nn.Linear(embed_dim, self.nf * 4))
            time_encoder.append(nn.Linear(self.nf * 4, self.nf * 4))
            nn.init.xavier_uniform_(time_encoder[-2].weight)
            nn.init.zeros_(time_encoder[-2].bias)
            nn.init.xavier_uniform_(time_encoder[-1].weight)
            nn.init.zeros_(time_encoder[-1].bias)
        #print("time_encoder", time_encoder)
        return time_encoder    
        
    def _build_encoder(self, **enc_kargs):
        """
        Build encoder layers 
        """
        encoder = nn.ModuleList()
        encoder.append(DenseBlock(16, 16, "enc",temb_dim = self.temb_dim))
        for i in range(4):
            encoder.append(
                    nn.Sequential(
                            Conv2dBlock(in_dims=16 if i==0 else 32, 
                                    out_dims=32, **enc_kargs),
                            DenseBlock(32, 32, "enc", temb_dim = self.temb_dim)
                            )
                    )
        encoder.append(Conv2dBlock(in_dims=32, out_dims=64, **enc_kargs))
        encoder.append(Conv2dBlock(in_dims=64, out_dims=128, **enc_kargs))
        encoder.append(Conv2dBlock(in_dims=128, out_dims=384, **enc_kargs))

        return encoder
    
    def _build_decoder(self, **dec_kargs):
        """
        Build decoder layers 
        """
        decoder = nn.ModuleList()
        decoder.append(ConvTrans2dBlock(in_dims=384*2, out_dims=128, **dec_kargs))
        decoder.append(ConvTrans2dBlock(in_dims=128*2, out_dims=64, **dec_kargs))
        decoder.append(ConvTrans2dBlock(in_dims=64*2, out_dims=32, **dec_kargs))        
        for i in range(4):
            decoder.append(
                    nn.Sequential(
                            DenseBlock(32, 64, "dec", temb_dim = self.temb_dim),
                            ConvTrans2dBlock(in_dims=64, 
                                            out_dims=32  if i!=3 else 16,
                                             **dec_kargs)
                            )
                    )
        decoder.append(DenseBlock(16, 32, "dec", temb_dim = self.temb_dim))                            
        
        return decoder
    
    def _build_mel_decoder(self, kernel_size, stride, padding, output_padding):
        """
        Build decoder layers 
        """
        decoder = nn.ModuleList()
        decoder.append(ConvTrans2dBlock(in_dims=384*2, out_dims=128, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding))
        decoder.append(ConvTrans2dBlock(in_dims=128*2, out_dims=64, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=[0,0]))
        decoder.append(ConvTrans2dBlock(in_dims=64*2, out_dims=32, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=[0,0]))        
        for i in range(4):
            decoder.append(
                    nn.Sequential(
                            DenseBlock(32, 64, "dec", temb_dim = self.temb_dim),
                            ConvTrans2dBlock(in_dims=64, 
                                            out_dims=32  if i!=3 else 16,
                                            kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)
                            )
                    )
        decoder.append(DenseBlock(16, 32, "dec", temb_dim = self.temb_dim))                            
        
        return decoder

    def _build_speaker_encoder(self, aux_input_dim, aux_hidden_dim, kernel_size, stride, paddings):
        """
        Build speaker encoder layers
        """
        speaker_encoder = nn.ModuleList()
        speaker_encoder.append(
            nn.Sequential(
                nn.Conv1d(in_channels=aux_input_dim, out_channels=aux_hidden_dim, kernel_size=1),
                nn.ReLU()
            )
        )
        speaker_encoder.append(Conv1dBlock(aux_hidden_dim, aux_input_dim))
        speaker_encoder.append(Conv2dBlock(1, 16, kernel_size=kernel_size, stride=stride, padding=paddings))

        return speaker_encoder
    
    def _build_tcn_blocks(self, tcn_blocks, **tcn_kargs):
        """
        Build TCN blocks in each repeat (layer)
        """
        blocks = [
            TCNBlock(**tcn_kargs, dilation=(2**b))
            for b in range(tcn_blocks)
        ]
        
        return nn.Sequential(*blocks)
    
    def _build_tcn_layers(self, tcn_layers, tcn_blocks, **tcn_kargs):
        """
        Build TCN layers
        """
        layers = [
            self._build_tcn_blocks(tcn_blocks, **tcn_kargs)
            for _ in range(tcn_layers)
        ]
        
        return nn.Sequential(*layers)
    
    def _build_avg_pool(self, pool_size):
        """
        Build avg pooling layers
        """
        avg_pool = nn.ModuleList()
        for sz in pool_size:
            avg_pool.append(
                    nn.Sequential(
                            nn.AvgPool2d(sz),
                            nn.Conv2d(32, 8, 1, 1)                            
                            )
                )
        
        return avg_pool
    
    def sep(self, spec: torch.Tensor):
        """
        spec: (batchsize, channels, T, F)
        return [real, imag] or waveform for each speaker
        """
        spec = torch.einsum("hijk->hikj", spec)        # (batchsize, channels, F, T)
        spec = torch.chunk(spec, self.channels, 1)
        B, N, F, T = spec[0].shape
        est1 = torch.chunk(spec[0], 2, 1)      # [(B, 1, F, T), (B, 1, F, T)]
        est1 = torch.cat(est1, 2).reshape(B, -1, T)      # B, 1, 2F, T
        return est1
        
    def forward(self, x: torch.Tensor, time_cond: torch.Tensor, enroll: torch.Tensor) -> torch.Tensor:
        # process condition
        if self.spk_encoder_backbone: 
            enroll = self.speaker_encoder_model(enroll)
            enroll = enroll.unsqueeze(1)
        
        
        # Convert real and imaginary parts of (x,y) into four channel dimensions
        if self.discriminative:
            x = torch.cat((x[:,[0],:,:].real, x[:,[0],:,:].imag), dim=1)
        else:
            x = torch.cat((x[:,[0],:,:].real, x[:,[0],:,:].imag,
                x[:,[1],:,:].real, x[:,[1],:,:].imag), dim=1)
        
        #print("conv2d.device", self.conv2d.weight.device, type(self.time_encoder))
        if self.embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            used_sigmas = time_cond
            temb = self.time_encoder[0](torch.log(used_sigmas))
            #print("temb",temb, temb.shape, temb.device)

        elif self.embedding_type == 'positional':
            # Sinusoidal positional embeddings.
            timesteps = time_cond
            #used_sigmas = self.sigmas[time_cond.long()]
            temb = layers.get_timestep_embedding(timesteps, self.nf)

        else:
            raise ValueError(f'embedding type {self.embedding_type} unknown.')

        #print("conv2d.device", self.conv2d.weight.device)

        if self.conditional:
            temb = self.time_encoder[1](temb)
            temb = self.time_encoder[2](self.act(temb))
        else:
            temb = None
            
        out = self.conv2d(x)
        
        enr = enroll
        #print("enr",enr.shape)
        enr = enr.unsqueeze(-1)
        
 
        out_list = []
        for i, enc in enumerate(self.encoder):
            #print("temb",temb.shape)
            #print("out",out.shape)
            #print(i, enc)
            input_encoder = {'x':out, 'temb':temb}
            #print()
            out = enc(input_encoder) # [2,16,256,510]
            if i == 0:
                #print("i:", i,"out",out.shape, "enr.repeat(1,out.shape[1], 1, out.shape[3])",(enr.repeat(1,out.shape[1], 1, out.shape[3])).shape)
                out = out * enr.repeat(1,out.shape[1], 1, out.shape[3])
            #print("i: ",i," out",out.shape)
            out_list.append(out)
        B, N, T, F = out.shape

        tcn_layers_input = {'x':out.reshape(B, N, T*F), 'temb':temb}
        out = self.tcn_layers(tcn_layers_input)
        #print("out, tcnlayer",out.shape)
        out = out.reshape(B, N, T, F)
        #out = torch.unsqueeze(out, -1)
        out_list = out_list[::-1]
        for idx, dec in enumerate(self.decoder):
            #print("idx",idx,"out_list[idx]",out_list[idx].shape, "out", out.shape)
            decoder_input = {'x':torch.cat([out_list[idx], out], 1), 'temb':temb}
            out = dec(decoder_input)   

        # Pyramidal pooling
        B, N, T, F = out.shape
        upsample = nn.Upsample(size=(T, F), mode='bilinear')
        pool_list = []
        for avg in self.avg_pool:
            pool_list.append(upsample(avg(out)))
            
        out = torch.cat([out, *pool_list], 1)
        out = self.avg_proj(out)
        out = self.deconv2d(out)
        B, C, T, F = out.shape

        #out = out.transpose(1,2).reshape(B, T, -1)
        out = torch.permute(out, (0, 2, 3, 1)).contiguous()
        out = torch.view_as_complex(out)[:,None, :, :]
        return out

