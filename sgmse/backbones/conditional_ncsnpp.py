# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file

from .ncsnpp_utils import layers, layerspp, normalization
import torch.nn as nn
import functools
import torch
import numpy as np

from .shared import BackboneRegistry

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


@BackboneRegistry.register("conditionalncsnpp")
class ConditionalNCSNpp(nn.Module):
    """NCSN++ model, adapted from https://github.com/yang-song/score_sde repository"""

    @staticmethod
    def add_argparse_args(parser):
        # TODO: add additional arguments of constructor, if you wish to modify them.
       
        return parser

    def __init__(self,
        scale_by_sigma = True,
        nonlinearity = 'swish',
        nf = 128,
        ch_mult = (1, 1, 2, 2, 2, 2, 2),
        num_res_blocks = 2,
        attn_resolutions = (16,),
        resamp_with_conv = True,
        conditional = True,
        fir = True,
        fir_kernel = [1, 3, 3, 1],
        skip_rescale = True,
        resblock_type = 'conditional_biggan',
        progressive = 'output_skip',
        progressive_input = 'input_skip',
        progressive_combine = 'sum',
        init_scale = 0.,
        fourier_scale = 16,
        image_size = 256,
        embedding_type = 'fourier',
        dropout = .0,
        centered = True,
        middle_concat_attention = False,
        discriminative = False,
        spk_encoder_backbone = False,
        middle_film_only = False, 
        middle_concat_attn_only = False,
        conditional_timestep=False,
        speaker_encoder_type="BLSTM",
        **unused_kwargs
    ):
        super().__init__()
        self.act = act = get_act(nonlinearity)

        self.nf = nf = nf
        ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions = attn_resolutions
        dropout = dropout
        resamp_with_conv = resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [image_size // (2 ** i) for i in range(num_resolutions)]

        #self.conditional = conditional = conditional  # noise-conditional
        self.discriminative = discriminative
        self.conditional_timestep = conditional_timestep
        if self.discriminative and not self.conditional_timestep:
            # overwrite options that make no sense for a discriminative model
            conditional = False
            scale_by_sigma = False
        self.conditional = conditional
        
        self.centered = centered
        self.scale_by_sigma = scale_by_sigma
        print("discriminative",self.discriminative, "conditional", conditional, "scale_by_sigma",scale_by_sigma,"conditional_timestep",self.conditional_timestep)

        fir = fir
        fir_kernel = fir_kernel
        self.skip_rescale = skip_rescale = skip_rescale
        self.resblock_type = resblock_type = resblock_type.lower()
        self.progressive = progressive = progressive.lower()
        self.progressive_input = progressive_input = progressive_input.lower()
        self.embedding_type = embedding_type = embedding_type.lower()
        init_scale = init_scale
        assert progressive in ['none', 'output_skip', 'residual']
        assert progressive_input in ['none', 'input_skip', 'residual']
        assert embedding_type in ['fourier', 'positional']
        combine_method = progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)

        
        print("self.discriminative", self.discriminative )
        num_channels = 4 if not self.discriminative else 2 # x.real, x.imag, y.real, y.imag
        self.output_layer = nn.Conv2d(num_channels, 2, 1)

        modules = []
        # timestep/noise_level embedding
        if embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            modules.append(layerspp.GaussianFourierProjection(
                embedding_size=nf, scale=fourier_scale
            ))
            embed_dim = 2 * nf
        elif embedding_type == 'positional':
            embed_dim = nf
        else:
            raise ValueError(f'embedding type {embedding_type} unknown.')

        if conditional:
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

        AttnBlock = functools.partial(layerspp.AttnBlockpp,
            init_scale=init_scale, skip_rescale=skip_rescale)

        Upsample = functools.partial(layerspp.Upsample,
            with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive == 'output_skip':
            self.pyramid_upsample = layerspp.Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive == 'residual':
            pyramid_upsample = functools.partial(layerspp.Upsample, fir=fir,
                fir_kernel=fir_kernel, with_conv=True)

        Downsample = functools.partial(layerspp.Downsample, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive_input == 'input_skip':
            self.pyramid_downsample = layerspp.Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive_input == 'residual':
            pyramid_downsample = functools.partial(layerspp.Downsample,
                fir=fir, fir_kernel=fir_kernel, with_conv=True)

        if resblock_type == 'ddpm':
            ResnetBlock = functools.partial(ResnetBlockDDPM, act=act,
                dropout=dropout, init_scale=init_scale,
                skip_rescale=skip_rescale, temb_dim=nf * 4)

        elif resblock_type == 'biggan':
            ResnetBlock = functools.partial(ResnetBlockBigGAN, act=act,
                dropout=dropout, fir=fir, fir_kernel=fir_kernel,
                init_scale=init_scale, skip_rescale=skip_rescale, temb_dim=nf * 4)
            
        elif resblock_type == 'conditional_biggan':
            ResnetBlock = functools.partial(ResnetBlockConditionalBigGAN, act=act,
                dropout=dropout, fir=fir, fir_kernel=fir_kernel,
                init_scale=init_scale, skip_rescale=skip_rescale, temb_dim=nf * 4,
                cross_attention_dim=256, attn_num_head_channels=4)
        elif resblock_type == 'conditional_concat_biggan':
            ResnetBlock = functools.partial(ResnetBlockConditionalConcatBigGAN, act=act,
                dropout=dropout, fir=fir, fir_kernel=fir_kernel,
                init_scale=init_scale, skip_rescale=skip_rescale, temb_dim=nf * 4,
                cross_attention_dim=256, attn_num_head_channels=4)    
        elif resblock_type == 'conditional_concat_biggan_attn':
            ResnetBlock = functools.partial(ResnetBlockConditionalConcatAttnBigGAN, act=act,
                dropout=dropout, fir=fir, fir_kernel=fir_kernel,
                init_scale=init_scale, skip_rescale=skip_rescale, temb_dim=nf * 4,
                cross_attention_dim=256, attn_num_head_channels=4)  
        elif resblock_type == 'conditional_film_biggan':
            ResnetBlock = functools.partial(ResnetBlockConditionalFiLMBigGAN, act=act,
                dropout=dropout, fir=fir, fir_kernel=fir_kernel,
                init_scale=init_scale, skip_rescale=skip_rescale, temb_dim=nf * 4,
                cross_attention_dim=256, attn_num_head_channels=4)  
        elif resblock_type == 'conditional_film_biggan_cross_attn':
            ResnetBlock1 = functools.partial(ResnetBlockConditionalFiLMBigGAN, act=act,
                dropout=dropout, fir=fir, fir_kernel=fir_kernel,
                init_scale=init_scale, skip_rescale=skip_rescale, temb_dim=nf * 4,
                cross_attention_dim=256, attn_num_head_channels=4)  
            ResnetBlock2 = functools.partial(ResnetBlockConditionalCrossAttnBigGAN, act=act,
                dropout=dropout, fir=fir, fir_kernel=fir_kernel,
                init_scale=init_scale, skip_rescale=skip_rescale, temb_dim=nf * 4,
                cross_attention_dim=256, attn_num_head_channels=4)  
        elif resblock_type == 'conditional_biggan_cross_attn':
            ResnetBlock = functools.partial(ResnetBlockConditionalCrossAttnBigGAN, act=act,
                dropout=dropout, fir=fir, fir_kernel=fir_kernel,
                init_scale=init_scale, skip_rescale=skip_rescale, temb_dim=nf * 4,
                cross_attention_dim=256, attn_num_head_channels=4)   
        elif resblock_type == 'downsample_biggan_upsample_condbigganfilm':
            ResnetBlock_down = functools.partial(ResnetBlockBigGAN, act=act,
                dropout=dropout, fir=fir, fir_kernel=fir_kernel,
                init_scale=init_scale, skip_rescale=skip_rescale, temb_dim=nf * 4)
            ResnetBlock_up = functools.partial(ResnetBlockConditionalFiLMBigGAN, act=act,
                dropout=dropout, fir=fir, fir_kernel=fir_kernel,
                init_scale=init_scale, skip_rescale=skip_rescale, temb_dim=nf * 4,
                cross_attention_dim=256, attn_num_head_channels=4)  
        else:
            raise ValueError(f'resblock type {resblock_type} unrecognized.')
        print("resblock_type",resblock_type)
        
        if middle_film_only:
            ResnetBlock_film =functools.partial(ResnetBlockConditionalFiLMBigGAN, act=act,
                    dropout=dropout, fir=fir, fir_kernel=fir_kernel,
                    init_scale=init_scale, skip_rescale=skip_rescale, temb_dim=nf * 4,
                    cross_attention_dim=256, attn_num_head_channels=4)  
        
        # Downsampling block

        channels = num_channels
        if progressive_input != 'none':
            input_pyramid_ch = channels

        modules.append(conv3x3(channels, nf))
        hs_c = [nf]

        in_ch = nf
        for i_level in range(num_resolutions):
            if resblock_type == 'conditional_film_biggan_cross_attn':
                if i_level == num_resolutions-1 or i_level == num_resolutions-2:
                    ResnetBlock = ResnetBlock2
                else:
                    ResnetBlock = ResnetBlock1
            if resblock_type == 'downsample_biggan_upsample_condbigganfilm':
                ResnetBlock = ResnetBlock_down
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                if resblock_type == 'ddpm':
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))

                if progressive_input == 'input_skip':
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if combine_method == 'cat':
                        in_ch *= 2

                elif progressive_input == 'residual':
                    modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
                    input_pyramid_ch = in_ch

                hs_c.append(in_ch)
        
        
        # Middle block
        if resblock_type == 'conditional_film_biggan_cross_attn':
            ResnetBlock = ResnetBlock2
            
                    
        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        self.middle_concat_attention = middle_concat_attention
        self.middle_concat_attn_only = middle_concat_attn_only

        print("self.middle_concat_attention",middle_concat_attention,"middle_concat_attn_only",middle_concat_attn_only)
        if middle_concat_attention or self.middle_concat_attn_only:
            modules.append(torch.nn.Linear(in_ch+256, in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))


        pyramid_ch = 0
        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            if resblock_type == 'conditional_film_biggan_cross_attn':
                if i_level ==0 or i_level == 1:
                    ResnetBlock = ResnetBlock2
                else:
                    ResnetBlock = ResnetBlock1
            if resblock_type == 'downsample_biggan_upsample_condbigganfilm':
                ResnetBlock = ResnetBlock_up
            for i_block in range(num_res_blocks + 1):  # +1 blocks in upsampling because of skip connection from combiner (after downsampling)
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if progressive != 'none':
                if i_level == num_resolutions - 1:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                            num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name.')
                else:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                            num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, channels, bias=True, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name')

            if i_level != 0:
                if resblock_type == 'ddpm':
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        if progressive != 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                                    num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)
        
        self.spk_encoder_backbone = spk_encoder_backbone
        if self.spk_encoder_backbone: 
            print("speaker_encoder_type",speaker_encoder_type)
            self.speaker_encoder_model = layerspp.SpeakerEncoder(speaker_encoder_type = speaker_encoder_type)
            
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--no-centered", dest="centered", action="store_false", help="The data is not centered [-1, 1]")
        parser.add_argument("--centered", dest="centered", action="store_true", help="The data is centered [-1, 1]")
        parser.add_argument("--resblock_type", type=str, default="conditional_biggan", 
                            choices=("ddpm", "biggan","conditional_biggan","conditional_concat_biggan",
                                     "conditional_concat_biggan_attn","conditional_film_biggan",
                                     "conditional_film_biggan_cross_attn","conditional_biggan_cross_attn",
                                     "downsample_biggan_upsample_condbigganfilm"), 
                            help="The type of resblock.")
        parser.add_argument("--middle_concat_attention", type=bool, default=False, 
                            help="middle block, concat the condition embedding before attention")
        parser.add_argument("--middle_film_only", action='store_true',   help="use film only in the middle")
        parser.add_argument("--middle_concat_attn_only", action='store_true',   help="use film only in the middle")
        parser.add_argument("--spk_encoder_backbone", action='store_true',   help="use film only in the middle")
        parser.add_argument("--conditional_timestep", action='store_true',   help="use film only in the middle")

        
        
        
        parser.set_defaults(centered=True)
        return parser

    def forward(self, x, time_cond, condition = None):
        # process condition
        if self.spk_encoder_backbone: 
            #print("Condition before spk_encoder", condition.shape) #[batchsize, 1, L]
            condition = self.speaker_encoder_model(condition)
            #print("Condition after spk_encoder", condition.shape) [batchsize, 1, 256]
            
        
        # timestep/noise_level embedding; only for continuous training
        modules = self.all_modules
        m_idx = 0

        # Convert real and imaginary parts of (x,y) into four channel dimensions
        if self.discriminative:
            x = torch.cat((x[:,[0],:,:].real, x[:,[0],:,:].imag), dim=1)
        else:
            x = torch.cat((x[:,[0],:,:].real, x[:,[0],:,:].imag,
                x[:,[1],:,:].real, x[:,[1],:,:].imag), dim=1)

        if self.embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            used_sigmas = time_cond
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1

        elif self.embedding_type == 'positional':
            # Sinusoidal positional embeddings.
            timesteps = time_cond
            used_sigmas = self.sigmas[time_cond.long()]
            temb = layers.get_timestep_embedding(timesteps, self.nf)

        else:
            raise ValueError(f'embedding type {self.embedding_type} unknown.')

        if self.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if not self.centered:
            # If input data is in [0, 1]
            x = 2 * x - 1.

        # Downsampling block
        input_pyramid = None
        if self.progressive_input != 'none':
            input_pyramid = x

        # Input layer: Conv2d: 4ch -> 128ch
        hs = [modules[m_idx](x)]
        m_idx += 1

        # Down path in U-Net
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb, condition)
                m_idx += 1
                # Attention layer (optional)
                if h.shape[-2] in self.attn_resolutions: # edit: check H dim (-2) not W dim (-1)
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)

            # Downsampling
            if i_level != self.num_resolutions - 1:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    h = modules[m_idx](hs[-1], temb, condition)
                    m_idx += 1

                if self.progressive_input == 'input_skip':   # Combine h with x
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1

                elif self.progressive_input == 'residual':
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid
                hs.append(h)

        h = hs[-1] # actualy equal to: h = h
        h = modules[m_idx](h, temb, condition)  # ResNet block
        m_idx += 1
        if self.middle_concat_attention or self.middle_concat_attn_only:
            batch, inner_dim, height, width = h.shape
            h = h.reshape(batch, height * width, inner_dim)
            encoder_hidden_states = condition.repeat(1, height * width, 1)
            h = modules[m_idx](torch.concat((h, encoder_hidden_states), dim=2))
            h = h.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
            m_idx += 1
        h = modules[m_idx](h)  # Attention block
        m_idx += 1
        h = modules[m_idx](h, temb, condition)  # ResNet block
        m_idx += 1

        pyramid = None

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb, condition)
                m_idx += 1

            # edit: from -1 to -2
            if h.shape[-2] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if self.progressive != 'none':
                if i_level == self.num_resolutions - 1:
                    if self.progressive == 'output_skip':
                        pyramid = self.act(modules[m_idx](h))  # GroupNorm
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)  # Conv2D: 256 -> 4
                        m_idx += 1
                    elif self.progressive == 'residual':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name.')
                else:
                    if self.progressive == 'output_skip':
                        pyramid = self.pyramid_upsample(pyramid)  # Upsample
                        pyramid_h = self.act(modules[m_idx](h))  # GroupNorm
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.progressive == 'residual':
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name')

            # Upsampling Layer
            if i_level != 0:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    h = modules[m_idx](h, temb, condition)  # Upspampling
                    m_idx += 1

        assert not hs

        if self.progressive == 'output_skip':
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1

        assert m_idx == len(modules), "Implementation error"
        if self.scale_by_sigma:
            used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
            h = h / used_sigmas

        # Convert back to complex number
        h = self.output_layer(h)
        h = torch.permute(h, (0, 2, 3, 1)).contiguous()
        h = torch.view_as_complex(h)[:,None, :, :]
        return h





@BackboneRegistry.register("conditionalncsnpp12M")
class ConditionalNCSNppSmall(ConditionalNCSNpp):
    """Small-scale NCSN++ model. ~12M parameters"""

    def __init__(self, **kwargs):
        super().__init__( 
        nf = 96,
        ch_mult = (1, 2, 2, 1),
        num_res_blocks = 1,
        attn_resolutions = (0,),
        **kwargs)

    # @staticmethod
    # def add_argparse_args(parser):
    #     # parser.add_argument("--centered", action="store_true", help="The data is already centered [-1, 1]")
    #     return parser

