# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import copy
import argparse
import random
import pydicom
from pathlib import Path
from easydict import EasyDict as edict

import numpy as np

import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.utils.data import DataLoader, Subset
from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu

from logger import Logger
import distributed_util as dist_util
from a2sb import download_ckpt
from a2sb.runner import Runner as A2SB_Runner
from a2sb.runner_latent import Runner as LDM_Runner
from dataset import dataset
from a2sb import ckpt_util
from guided_diffusion.script_util import create_gaussian_diffusion

import colored_traceback.always
from ipdb import set_trace as debug

RESULT_DIR = Path("results")

def set_seed(seed):
    # https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

def build_subset_per_gpu(opt, dataset, log):
    n_data = len(dataset)
    n_gpu  = opt.global_size
    n_dump = (n_data % n_gpu > 0) * (n_gpu - n_data % n_gpu)

    # create index for each gpu
    total_idx = np.concatenate([np.arange(n_data), np.zeros(n_dump)]).astype(int)
    idx_per_gpu = total_idx.reshape(-1, n_gpu)[:, opt.global_rank]
    log.info(f"[Dataset] Add {n_dump} data to the end to be devided by {n_gpu}. Total length={len(total_idx)}!")

    # build subset
    indices = idx_per_gpu.tolist()
    subset = Subset(dataset, indices)
    log.info(f"[Dataset] Built subset for gpu={opt.global_rank}! Now size={len(subset)}!")
    return subset

def collect_all_subset(sample, log):
    batch, *xdim = sample.shape
    gathered_samples = dist_util.all_gather(sample, log)
    gathered_samples = [sample.cpu() for sample in gathered_samples]
    # [batch, n_gpu, *xdim] --> [batch*n_gpu, *xdim]
    return torch.stack(gathered_samples, dim=1).reshape(-1, *xdim)

def build_partition(opt, full_dataset, log):
    n_samples = len(full_dataset)

    part_idx, n_part = [int(s) for s in opt.partition.split("_")]
    assert part_idx < n_part and part_idx >= 0
    assert n_samples % n_part == 0

    n_samples_per_part = n_samples // n_part
    start_idx = part_idx * n_samples_per_part
    end_idx = (part_idx+1) * n_samples_per_part

    indices = [i for i in range(start_idx, end_idx)]
    subset = Subset(full_dataset, indices)
    log.info(f"[Dataset] Built partition={opt.partition}, {start_idx}, {end_idx}! Now size={len(subset)}!")
    return subset

def build_val_dataset(opt, log):
    val_dataset = dataset.A2SBDataset(opt, log, mode='test')

    # build partition
    if opt.partition is not None:
        val_dataset = build_partition(opt, val_dataset, log)
    return val_dataset

def get_recon_imgs_fn(opt, nfe):
    recon_imgs_fn = RESULT_DIR / "s2b" / opt.sbae_ckpt / "samples_nfe{}{}_iter{}{}".format(
        nfe, "_clip" if opt.clip_denoise else "", opt.load_itr, "_" + str(opt.ldm_load_itr) if opt.use_ldm else ""
    )
    os.makedirs(recon_imgs_fn, exist_ok=True)

    return recon_imgs_fn

def generate_style(opt, log, ldm_runner, ldm_ckpt_opt, cond=None, nfe=2):
    diffusion = create_gaussian_diffusion(steps=ldm_ckpt_opt.interval, noise_schedule=ldm_ckpt_opt.schedule_name, timestep_respacing=f"ddim{nfe}")
    z_style = diffusion.ddim_sample_loop(ldm_runner.net, (opt.batch_size, 1024), cond=cond, clip_denoised=ldm_ckpt_opt.clip_denoise, progress=True)
    log.info("Generated style feature!")
    return z_style

def compute_batch(ckpt_opt, out):
    clean_img, corrupt_img, fpath = out
    x0 = clean_img.to(opt.device)
    x1 = corrupt_img.to(opt.device)
    cond = x1.detach() if ckpt_opt.cond_x1 else None

    # if ckpt_opt.add_x1_noise: # only for decolor
    #     x1 = x1 + torch.randn_like(x1)
 
    return x0, x1, cond, fpath

def save_dicom(dcm_path, predict_tensor, save_path):
    predict_img = (predict_tensor*4095.0-1024.0).detach().clone().cpu().numpy()
    dcm = pydicom.dcmread(dcm_path, force=True)

    intercept = dcm.RescaleIntercept
    slope = dcm.RescaleSlope
    
    predict_img -= np.float32(intercept)
    if slope != 1:
        predict_img = predict_img.astype(np.float32) / slope
    predict_img = predict_img.astype(np.int16)

    dcm.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    dcm.PixelData = predict_img.squeeze().tobytes()

    dcm.SmallestImagePixelValue = predict_img.min()
    dcm.LargestImagePixelValue = predict_img.max()
    dcm[0x0028,0x0106].VR = 'US'
    dcm[0x0028,0x0107].VR = 'US'

    dcm.save_as(save_path)

@torch.no_grad()
def main(opt):
    log = Logger(opt.global_rank, ".log")

    # get (default) ckpt option
    sbae_ckpt_opt = ckpt_util.build_ckpt_option(opt, log, RESULT_DIR / "s2b" / opt.sbae_ckpt)
    nfe = opt.nfe or sbae_ckpt_opt.interval-1

    # build imagenet val dataset
    val_dataset = build_val_dataset(opt, log)
    n_samples = len(val_dataset)

    # build dataset per gpu and loader
    subset_dataset = build_subset_per_gpu(opt, val_dataset, log)
    val_loader = DataLoader(subset_dataset,
        batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=1, drop_last=False,
    )

    # build runner
    a2sb_runner = A2SB_Runner(sbae_ckpt_opt, log, save_opt=False)

    # handle use_fp16 for ema
    if opt.use_fp16:
        a2sb_runner.ema.copy_to() # copy weight from ema to net
        a2sb_runner.net.diffusion_model.convert_to_fp16()
        a2sb_runner.net.semantic_enc.convert_to_fp16()
        a2sb_runner.ema = ExponentialMovingAverage(a2sb_runner.net.parameters(), decay=0.99) # re-init ema with fp16 weight

    # use ldm runner
    if opt.use_ldm:
        ldm_ckpt_opt = ckpt_util.build_ckpt_option(opt, log, RESULT_DIR / "latent-ddim" / opt.ldm_ckpt, net="latent-ddim")
        ldm_runner = LDM_Runner(ldm_ckpt_opt, log, save_opt=False)
        if opt.use_fp16:
            ldm_runner.ema.copy_to() # copy weight from ema to net
            ldm_runner.net.convert_to_fp16()
            ldm_runner.ema = ExponentialMovingAverage(ldm_runner.net.parameters(), decay=0.99) # re-init ema with fp16 weight

    # create save folder
    recon_imgs_fn = get_recon_imgs_fn(opt, nfe)
    log.info(f"Recon images will be saved to {recon_imgs_fn}!")

    recon_imgs = []
    num = 0

    for loader_itr, out in enumerate(val_loader):
        x0, x1, cond, fpath = compute_batch(sbae_ckpt_opt, out)

        # generate style using latent ddim network
        generated_style = generate_style(opt, log, ldm_runner=ldm_runner, ldm_ckpt_opt=ldm_ckpt_opt, cond=x1, nfe=nfe) if opt.use_ldm else None

        xs, _ = a2sb_runner.ddpm_sampling(
            sbae_ckpt_opt, x0, x1, cond=cond, generated_style=generated_style, clip_denoise=opt.clip_denoise, nfe=nfe, verbose=opt.n_gpu_per_node==1
        )
        recon_img = xs[:, 0, ...].to(opt.device)
        if opt.clip_denoise: recon_img.clamp_(-1., 1.)

        assert recon_img.shape == x1.shape == x0.shape

        tu.save_image((x1+1)/2, recon_imgs_fn / f"{fpath[0].split('/')[-1]}_source.png", value_range=(0, 1))
        tu.save_image((x0+1)/2, recon_imgs_fn / f"{fpath[0].split('/')[-1]}_target.png", value_range=(0, 1))
        tu.save_image((recon_img+1)/2, recon_imgs_fn / f"{fpath[0].split('/')[-1]}_target_recon.png", value_range=(0, 1))            
        
        # save as dicom
        if opt.save_dicom and opt.batch_size == 1:
            os.makedirs(recon_imgs_fn / "dcm" / opt.src / fpath[0].split("/")[-2], exist_ok=True)
            save_dicom(fpath[0], (recon_img+1)/2, recon_imgs_fn / "dcm" / opt.src / fpath[0].split("/")[-2] / fpath[0].split("/")[-1])
        log.info("Saved output images!")

        # [-1,1]
        gathered_recon_img = collect_all_subset(recon_img, log)
        recon_imgs.append(gathered_recon_img)

        num += len(gathered_recon_img)
        log.info(f"Collected {num} recon images!")
        dist.barrier()

    del a2sb_runner

    arr = torch.cat(recon_imgs, axis=0)[:n_samples]

    if opt.global_rank == 0:
        torch.save({"arr": arr}, recon_imgs_fn / "recon.pt")
        log.info(f"Save at {recon_imgs_fn}")
    dist.barrier()

    log.info(f"Sampling complete! Collect recon_imgs={arr.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,  default=0)
    parser.add_argument("--n-gpu-per-node", type=int,  default=1,           help="number of gpu on each node")
    parser.add_argument("--master-address", type=str,  default='localhost', help="address for master")
    parser.add_argument("--master-port",    type=str,  default='6020',      help="port for master")
    parser.add_argument("--node-rank",      type=int,  default=0,           help="the index of node")
    parser.add_argument("--num-proc-node",  type=int,  default=1,           help="The number of nodes in multi node env")

    # data
    parser.add_argument("--image-size",     type=int,  default=512)
    parser.add_argument("--dataset-dir",    type=Path, default="/data",     help="path to dataset")
    parser.add_argument("--src",            type=str,  default='src',       help="source folder name")
    parser.add_argument("--trg",            type=str,  default='trg',       help="target folder name")
    parser.add_argument("--partition",      type=str,  default=None,        help="e.g., '0_4' means the first 25% of the dataset")

    # latent ddim
    parser.add_argument("--ldm-load-itr",   type=int,  default=200000)
    parser.add_argument("--ldm-ckpt",       type=str,  default=None,        help="the checkpoint name from which we wish to sample from ldm")
    parser.add_argument("--use-ldm",        action="store_true",            help="use latent ddim network for generating semantic style")

    # sample
    parser.add_argument("--load-itr",       type=int,  default=50000)
    parser.add_argument("--batch-size",     type=int,  default=1)
    parser.add_argument("--s2b-ckpt",      type=str,  default=None,        help="the checkpoint name from which we wish to sample from s2b")
    parser.add_argument("--nfe",            type=int,  default=None,        help="sampling steps")
    parser.add_argument("--clip-denoise",   action="store_true",            help="clamp predicted image to [-1,1] at each")
    parser.add_argument("--use-fp16",       action="store_true",            help="use fp16 network weight for faster sampling")
    parser.add_argument("--save-dicom",     action="store_true",            help="save as dicom file")

    arg = parser.parse_args()

    opt = edict(
        distributed=(arg.n_gpu_per_node > 1),
        device="cuda",
    )
    opt.update(vars(arg))

    # one-time download: ADM checkpoint
    download_ckpt("data/")

    set_seed(opt.seed)

    if opt.distributed:
        size = opt.n_gpu_per_node

        processes = []
        for rank in range(size):
            opt = copy.deepcopy(opt)
            opt.local_rank = rank
            global_rank = rank + opt.node_rank * opt.n_gpu_per_node
            global_size = opt.num_proc_node * opt.n_gpu_per_node
            opt.global_rank = global_rank
            opt.global_size = global_size
            print('Node rank %d, local proc %d, global proc %d, global_size %d' % (opt.node_rank, rank, global_rank, global_size))
            p = Process(target=dist_util.init_processes, args=(global_rank, global_size, main, opt))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        torch.cuda.set_device(0)
        opt.global_rank = 0
        opt.local_rank = 0
        opt.global_size = 1
        dist_util.init_processes(0, opt.n_gpu_per_node, main, opt)
