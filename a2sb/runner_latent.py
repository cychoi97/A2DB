# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import math
import numpy as np
import pickle

import torch
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu
# import torchmetrics

import distributed_util as dist_util
# from evaluation import build_resnet50

from . import util
from .network import Image256Net, Image512Net, MLPNet
from .diffusion import Diffusion
from a2sb import ckpt_util
from guided_diffusion.script_util import create_gaussian_diffusion

from ipdb import set_trace as debug

def build_optimizer_sched(opt, net, log):
    
    optim_dict = {"lr": opt.lr, 'weight_decay': opt.l2_norm}
    optimizer = AdamW(net.parameters(), **optim_dict)
    log.info(f"[Opt] Built AdamW optimizer {optim_dict}!")

    if opt.lr_gamma < 1.0:
        sched_dict = {"step_size": opt.lr_step, 'gamma': opt.lr_gamma}
        sched = lr_scheduler.StepLR(optimizer, **sched_dict)
        log.info(f"[Opt] Built lr step scheduler {sched_dict}!")
    else:
        sched = None

    if opt.load:
        checkpoint = torch.load(opt.load, map_location="cpu")
        if "optimizer" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer"])
            log.info(f"[Opt] Loaded optimizer ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no optimizer!")
        if sched is not None and "sched" in checkpoint.keys() and checkpoint["sched"] is not None:
            sched.load_state_dict(checkpoint["sched"])
            log.info(f"[Opt] Loaded lr sched ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no lr sched!")

    return optimizer, sched

def all_cat_cpu(opt, log, t):
    if not opt.distributed: return t.detach().cpu()
    gathered_t = dist_util.all_gather(t.to(opt.device), log=log)
    return torch.cat(gathered_t).detach().cpu()

class Runner(object):
    def __init__(self, opt, log, save_opt=True):
        super(Runner,self).__init__()

        # Save opt.
        if save_opt:
            opt_pkl_path = opt.ckpt_path / "options.pkl"
            with open(opt_pkl_path, "wb") as f:
                pickle.dump(opt, f)
            log.info("Saved options pickle to {}!".format(opt_pkl_path))
        sbae_opt = ckpt_util.build_ckpt_option(opt, log, opt.sbae_load)
        self.diffusion = create_gaussian_diffusion(steps=opt.interval, noise_schedule=opt.schedule_name)
        log.info(f"[Diffusion] Built latent diffusion: steps={opt.interval}!")

        noise_levels = torch.linspace(opt.t0, opt.T, opt.interval, device=opt.device) * opt.interval
        
        if opt.image_size == 256:
            self.net = MLPNet(num_channels=512, num_hidden_channels=1024, num_layers=10, skip_layers=list(range(1, 10)), image_size=opt.image_size, use_fp16=False)
            self.a2sb = Image256Net(log, noise_levels=noise_levels, image_size=opt.image_size, in_channels=sbae_opt.in_channels,
                                    use_fp16=sbae_opt.use_fp16, cond=sbae_opt.cond_x1)
        elif opt.image_size == 512:
            self.net = MLPNet(num_channels=1024, num_hidden_channels=2048, num_layers=20, skip_layers=list(range(1, 20)), image_size=opt.image_size, use_fp16=False)
            self.a2sb = Image512Net(log, noise_levels=noise_levels, image_size=opt.image_size, in_channels=sbae_opt.in_channels,
                                    use_fp16=sbae_opt.use_fp16, cond=sbae_opt.cond_x1)

        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=opt.ema)
        self.sbae_ema = ExponentialMovingAverage(self.a2sb.parameters(), decay=sbae_opt.ema)

        if opt.load:
            checkpoint = torch.load(opt.load, map_location="cpu")
            self.net.load_state_dict(checkpoint['net'])
            log.info(f"[Net] Loaded network ckpt: {opt.load}!")
            self.ema.load_state_dict(checkpoint["ema"])
            log.info(f"[Ema] Loaded ema ckpt: {opt.load}!")

        sbae_checkpoint = torch.load(sbae_opt.load, map_location="cpu")
        self.a2sb.load_state_dict(sbae_checkpoint['net'])
        log.info(f"[Net] Loaded network ckpt: {sbae_opt.load}!")
        self.sbae_ema.load_state_dict(sbae_checkpoint["ema"])
        log.info(f"[Ema] Loaded ema ckpt: {sbae_opt.load}!")

        self.net.to(opt.device)
        self.a2sb.to(opt.device)
        self.ema.to(opt.device)
        self.sbae_ema.to(opt.device)

        self.log = log
        self.sbae_opt = sbae_opt

    @torch.no_grad()
    def sample_batch(self, opt, loader):
        clean_img, corrupt_img, _ = next(loader)

        # os.makedirs(".debug", exist_ok=True)
        # tu.save_image((clean_img+1)/2, ".debug/clean.png", nrow=4)
        # debug()

        x0 = clean_img.detach().to(opt.device)
        x1 = corrupt_img.detach().to(opt.device)
        with self.sbae_ema.average_parameters():
            self.a2sb.eval()
            z_sem = self.a2sb.semantic_enc(x1).detach().clone().cpu()
        return z_sem, x1

    def train(self, opt, train_dataset):
        self.writer = util.build_log_writer(opt)
        log = self.log

        net = DDP(self.net, device_ids=[opt.device])
        ema = self.ema
        optimizer, sched = build_optimizer_sched(opt, net, log)

        train_loader = util.setup_loader(train_dataset, opt.microbatch)

        net.train()

        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)
        for it in range(opt.start_itr, opt.num_itr + 1):
            optimizer.zero_grad()

            for _ in range(n_inner_loop):
                # ===== sample style embed =====
                z_sem, x1 = self.sample_batch(opt, train_loader)

                # ===== compute loss =====
                step = torch.randint(0, opt.interval, (z_sem.shape[0],))

                label = torch.randn_like(z_sem)
                xt = self.diffusion.q_sample(z_sem, step, noise=label).to(opt.device)

                pred = net(xt, step, x1)
                label = label.to(opt.device)
                assert z_sem.shape == label.shape == pred.shape

                loss = F.l1_loss(pred, label) # effective more than mse loss
                loss.backward()

            optimizer.step()
            ema.update()
            if sched is not None: sched.step()

            # -------- logging --------
            log.info("train_it {}/{} | lr:{} | loss:{}".format(
                it,
                opt.num_itr,
                "{:.2e}".format(optimizer.param_groups[0]['lr']),
                "{:+.4f}".format(loss.item()),
            ))
            if it % 10 == 0:
                self.writer.add_scalar(it, 'loss', loss.detach())

            if it % 5000 == 0 or it == opt.num_itr:
                if opt.global_rank == 0:
                    torch.save({
                        "net": self.net.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "sched": sched.state_dict() if sched is not None else sched,
                    }, opt.ckpt_path / f"{it:07}.pt")
                    log.info(f"Saved latest(it={it}) checkpoint to {opt.ckpt_path}!")
                if opt.distributed:
                    torch.distributed.barrier()
            torch.cuda.empty_cache()
        self.writer.close()
