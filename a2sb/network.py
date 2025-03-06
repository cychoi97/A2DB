# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import pickle
import torch
import torch.nn as nn

from guided_diffusion.script_util import create_model, create_encoder
from guided_diffusion.fp16_util import convert_module_to_f16, convert_module_to_f32

from . import util
from functools import partial
from .ckpt_util import (
    I2SB_IMG256_UNCOND_PKL,
    I2SB_IMG256_UNCOND_CKPT,
    I2SB_IMG256_COND_PKL,
    I2SB_IMG256_COND_CKPT,
    I2SB_IMG512_UNCOND_PKL,
    I2SB_IMG512_UNCOND_CKPT,
    I2SB_IMG512_COND_PKL,
    I2SB_IMG512_COND_CKPT,
)

from ipdb import set_trace as debug

#===================================================================================================#
#                                     First Stage: A2SB Network                                     #
#===================================================================================================#

class Image256Net(torch.nn.Module):
    def __init__(self, log, noise_levels, image_size=256, in_channels=1, use_fp16=False, cond=False, pretrained_adm=False, ckpt_dir="data/"):
        super(Image256Net, self).__init__()

        # initialize model
        ckpt_pkl = os.path.join(ckpt_dir, I2SB_IMG256_COND_PKL if cond else I2SB_IMG256_UNCOND_PKL)
        with open(ckpt_pkl, "rb") as f:
            kwargs = pickle.load(f)
        kwargs["use_fp16"] = use_fp16
        if cond:
            in_channels = in_channels * 2
        kwargs["in_channels"] = in_channels
        self.diffusion_model = create_model(**kwargs)
        self.semantic_enc = create_encoder(image_size,
                                           in_channels=1,
                                           encoder_use_fp16=use_fp16,
                                           encoder_width=128,
                                           encoder_attention_resolutions="16")
        log.info(f"[Net] Initialized network from {ckpt_pkl}! Size={util.count_parameters(self.diffusion_model)}!")
        log.info(f"[Enc] Initialized network from create_encoder() in script_util.py! Size={util.count_parameters(self.semantic_enc)}!")

        # load (modified) adm ckpt
        if pretrained_adm:
            ckpt_pt = os.path.join(ckpt_dir, I2SB_IMG256_COND_CKPT if cond else I2SB_IMG256_UNCOND_CKPT)
            out = torch.load(ckpt_pt, map_location="cpu")
            self.diffusion_model.load_state_dict(out)
            log.info(f"[Net] Loaded pretrained adm {ckpt_pt}!")

        self.diffusion_model.eval()
        self.semantic_enc.eval()
        self.cond = cond
        self.noise_levels = noise_levels

    def forward(self, x1, xt, steps, cond=None, generated_style=None):

        t = self.noise_levels[steps].detach()
        assert t.dim()==1 and t.shape[0] == xt.shape[0]
        style_emb = self.semantic_enc(x1).detach() if generated_style is None else generated_style

        xt = torch.cat([xt, cond], dim=1) if self.cond else xt
        return self.diffusion_model(xt, t, style_emb), style_emb


class Image512Net(torch.nn.Module):
    def __init__(self, log, noise_levels, image_size=512, in_channels=1, use_fp16=False, cond=False, pretrained_adm=False, ckpt_dir="data/"):
        super(Image512Net, self).__init__()

        # initialize model
        ckpt_pkl = os.path.join(ckpt_dir, I2SB_IMG512_COND_PKL if cond else I2SB_IMG512_UNCOND_PKL)
        with open(ckpt_pkl, "rb") as f:
            kwargs = pickle.load(f)
        kwargs["use_fp16"] = use_fp16
        if cond:
            in_channels = in_channels * 2
        kwargs["in_channels"] = in_channels
        # channel size = sbae-xs:128, sbae-s:192, sbae-m:256, sbae-l:320, sbae-xl:384
        kwargs["num_channels"] = 256
        self.diffusion_model = create_model(**kwargs)
        self.semantic_enc = create_encoder(image_size,
                                           in_channels=1,
                                           encoder_use_fp16=use_fp16,
                                           encoder_width=256,
                                           encoder_attention_resolutions="32,16,8") # 32,16,8
        log.info(f"[Net] Initialized network from {ckpt_pkl}! Size={util.count_parameters(self.diffusion_model)}!")
        log.info(f"[Enc] Initialized network from create_encoder() in script_util.py! Size={util.count_parameters(self.semantic_enc)}!")

        # load (modified) adm ckpt
        if pretrained_adm:
            ckpt_pt = os.path.join(ckpt_dir, I2SB_IMG512_COND_CKPT if cond else I2SB_IMG512_UNCOND_CKPT)
            out = torch.load(ckpt_pt, map_location="cpu")
            self.diffusion_model.load_state_dict(out)
            log.info(f"[Net] Loaded pretrained adm {ckpt_pt}!")

        self.diffusion_model.eval()
        self.semantic_enc.eval()
        self.cond = cond
        self.noise_levels = noise_levels

    def forward(self, x1, xt, steps, cond=None, generated_style=None):

        t = self.noise_levels[steps].detach()
        assert t.dim()==1 and t.shape[0] == xt.shape[0]
        style_emb = self.semantic_enc(x1).detach() if generated_style is None else generated_style

        xt = torch.cat([xt, cond], dim=1) if self.cond else xt
        return self.diffusion_model(xt, t, style_emb), style_emb
    
#===================================================================================================#
#                                 Second Stage: Latent DDIM Network                                 #
#===================================================================================================#

class MLPNet(torch.nn.Module):
    def __init__(
        self,
        num_channels,
        num_hidden_channels,
        num_layers,
        skip_layers,
        image_size=512,
        num_time_emb_channels=64,
        num_time_layers=2,
        use_norm = True,
        condition_bias=1.,
        dropout=0.,
        time_last_act=False,
        use_fp16=False
    ):
        super(MLPNet, self).__init__()

        self.num_time_emb_channels = num_time_emb_channels
        self.skip_layers = skip_layers
        self.act = nn.SiLU()
        self.dropout = dropout
        self.last_act = nn.Identity()
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.n_stages = 3
        self.multiplier = 0.5
        self.downsize = int(image_size//(2**self.n_stages))

        # x1 projection
        self.interpolator = partial(nn.functional.interpolate, mode="bilinear")
        self.proj1 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.downsize**2, num_channels),
            )
        self.proj2 = nn.Linear(num_channels*2, num_channels)

        # time embedding
        layers = []
        for i in range(num_time_layers):
            if i == 0:
                in_channels = num_time_emb_channels
                out_channels = num_channels
            else:
                in_channels = num_channels
                out_channels = num_channels
            layers.append(nn.Linear(in_channels, out_channels))
            if i < num_time_layers - 1 or time_last_act:
                layers.append(self.act)
        self.time_embed = nn.Sequential(*layers)

        # mlp layers
        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            if i == 0:
                act = self.act
                norm = use_norm
                cond = True
                in_channels, out_channels = num_channels, num_hidden_channels
                dropout = self.dropout
            elif i == num_layers - 1:
                act = self.last_act
                norm = False
                cond = False
                in_channels, out_channels = num_hidden_channels, num_channels
                dropout = 0.
            else:
                act = self.act
                norm = use_norm
                cond = True
                in_channels, out_channels = num_hidden_channels, num_hidden_channels
                dropout = self.dropout

            if i in skip_layers:
                in_channels += num_channels

            self.layers.append(
                MLPLNAct(
                    in_channels,
                    out_channels,
                    norm=norm,
                    activation=act,
                    cond_channels=num_channels, # num_channels
                    use_cond=cond,
                    condition_bias=condition_bias,
                    dropout=dropout
                )
            )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.layers.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.layers.apply(convert_module_to_f32)

    def forward(self, x, t, x1=None):
        t = util.timestep_embedding(t, self.num_time_emb_channels)
        cond = self.time_embed(t)
        if x1 is not None:
            for stage in range(self.n_stages):
                x1 = self.interpolator(x1, scale_factor=self.multiplier) # (B, C, 64, 64) if 512x512
            z_sem = self.proj1(x1) # (B, 1024)
            x = torch.cat([x, z_sem], dim=1) # (B, 2048)
            # x += z_sem
            x = self.proj2(x)
        h = x.type(self.dtype).clone() # (B, 1024)
        for i in range(len(self.layers)):
            if i in self.skip_layers:
                h = torch.cat([h, x], dim=1)
            h = self.layers[i].forward(x=h, cond=cond)
        h = self.last_act(h)
        return h
    

class MLPLNAct(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm,
        activation,
        cond_channels,
        use_cond,
        condition_bias=0.,
        dropout=0.,
    ):
        super().__init__()
        self.activation = activation
        self.condition_bias = condition_bias
        self.use_cond = use_cond

        self.linear = nn.Linear(in_channels, out_channels)
        if self.use_cond:
            self.linear_emb = nn.Linear(cond_channels, out_channels)
            self.cond_layers = nn.Sequential(self.activation, self.linear_emb)
        if norm:
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = nn.Identity()
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation == nn.SiLU():
                    nn.init.kaiming_normal_(module.weight, a=0, nonlinearity='relu')
                else:
                    # leave it as default
                    pass

    def forward(self, x, cond=None):
        x = self.linear(x)
        if self.use_cond:
            # (n, c) or (n, c * 2)
            cond = self.cond_layers(cond)
            cond = (cond, None)

            # scale shift first
            x = x * (self.condition_bias + cond[0])
            if cond[1] is not None:
                x = x + cond[1]
            # then norm
            x = self.norm(x)
        else:
            # no condition
            x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x