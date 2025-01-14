# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import random
import argparse

import copy
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.multiprocessing import Process

from logger import Logger
from distributed_util import init_processes
from dataset import dataset
from sbae.runner_latent import Runner

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

def create_training_options():
    # --------------- basic ---------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--name",           type=str,   default=None,         help="experiment ID")
    parser.add_argument("--ckpt",           type=str,   default=None,         help="resumed checkpoint name")
    parser.add_argument("--gpu",            type=int,   default=None,         help="set only if you wish to run on a particular device")
    parser.add_argument("--n-gpu-per-node", type=int,   default=1,            help="number of gpu on each node")
    parser.add_argument("--master-address", type=str,   default='localhost',  help="address for master")
    parser.add_argument("--master-port",    type=str,   default='6020',       help="port for master")
    parser.add_argument("--node-rank",      type=int,   default=0,            help="the index of node")
    parser.add_argument("--num-proc-node",  type=int,   default=1,            help="The number of nodes in multi node env")

    # --------------- Latent ddim model ---------------
    parser.add_argument("--image-size",     type=int,   default=512)
    parser.add_argument("--trg",            type=str,   default='trg',        help="target folder name")
    parser.add_argument("--t0",             type=float, default=1e-4,         help="sigma start time in network parametrization")
    parser.add_argument("--T",              type=float, default=1.,           help="sigma end time in network parametrization")
    parser.add_argument("--interval",       type=int,   default=1000,         help="number of interval")
    parser.add_argument("--schedule-name",  type=str,   default='const0.008', help="target folder name")
    parser.add_argument("--beta-max",       type=float, default=0.3,          help="max diffusion for the diffusion model")
    parser.add_argument("--use-fp16",       action="store_true",              help="use fp16 network weight for faster sampling")
    parser.add_argument("--clip-denoise",   action="store_true",              help="clamp predicted style embedding to [-1,1] at each")

    # --------------- sementic encoder model ---------------
    parser.add_argument("--load-itr",       type=int,   default=50000,        help="checkpoint iteration for loading semantic encoder")
    parser.add_argument("--sbae-ckpt",      type=str,   default=None,         help="the checkpoint name from which we wish to load semantic encoder")
    
    # --------------- optimizer and loss ---------------
    parser.add_argument("--batch-size",     type=int,   default=256)
    parser.add_argument("--microbatch",     type=int,   default=16,           help="accumulate gradient over microbatch until full batch-size")
    parser.add_argument("--start-itr",      type=int,   default=0,            help="start or resumed iteration")
    parser.add_argument("--num-itr",        type=int,   default=1000000,      help="training iteration")
    parser.add_argument("--lr",             type=float, default=1e-4,         help="learning rate")
    parser.add_argument("--lr-gamma",       type=float, default=1.0,          help="learning rate decay ratio")
    parser.add_argument("--lr-step",        type=int,   default=1000,         help="learning rate decay step size")
    parser.add_argument("--l2-norm",        type=float, default=0.01)
    parser.add_argument("--ema",            type=float, default=0.99)

    # --------------- path and logging ---------------
    parser.add_argument("--dataset-dir",    type=Path,  default="/data",      help="path to dataset")
    parser.add_argument("--log-dir",        type=Path,  default=".log",       help="path to log std outputs and writer data")
    parser.add_argument("--log-writer",     type=str,   default=None,         help="log writer: can be tensorbard, wandb, or None")
    parser.add_argument("--wandb-api-key",  type=str,   default=None,         help="unique API key of your W&B account; see https://wandb.ai/authorize")
    parser.add_argument("--wandb-user",     type=str,   default=None,         help="user name of your W&B account")

    opt = parser.parse_args()

    # ========= auto setup =========
    opt.device='cuda' if opt.gpu is None else f'cuda:{opt.gpu}'
    assert opt.name is not None
    opt.distributed = opt.n_gpu_per_node > 1
    opt.use_fp16 = False # disable fp16 for training

    # log ngc meta data
    if "NGC_JOB_ID" in os.environ.keys():
        opt.ngc_job_id = os.environ["NGC_JOB_ID"]

    # ========= path handle =========
    os.makedirs(opt.log_dir, exist_ok=True)
    opt.ckpt_path = RESULT_DIR / "latent-ddim" / opt.name
    os.makedirs(opt.ckpt_path, exist_ok=True)

    if opt.ckpt is not None:
        ckpt_file = RESULT_DIR / "latent-ddim" / opt.ckpt / f"{opt.start_itr:0>7}.pt"
        assert ckpt_file.exists()
        opt.load = ckpt_file
    else:
        opt.load = None

    assert opt.sbae_ckpt is not None
    sbae_ckpt_folder = RESULT_DIR / "sbae" / opt.sbae_ckpt
    opt.sbae_load = sbae_ckpt_folder

    return opt

def main(opt):
    log = Logger(opt.global_rank, opt.log_dir)
    log.info("=======================================================")
    log.info("                      Latent DDIM                      ")
    log.info("=======================================================")
    log.info("Command used:\n{}".format(" ".join(sys.argv)))
    log.info(f"Experiment ID: {opt.name}")

    # set seed: make sure each gpu has differnet seed!
    if opt.seed is not None:
        set_seed(opt.seed + opt.global_rank)

    # build dataset
    train_dataset = dataset.LatentDataset(opt, log, mode='train')
    # note: images should be normalized to [-1,1] for corruption methods to work properly

    run = Runner(opt, log)
    run.train(opt, train_dataset)
    log.info("Finish!")

if __name__ == '__main__':
    opt = create_training_options()

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
            p = Process(target=init_processes, args=(global_rank, global_size, main, opt))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        torch.cuda.set_device(0)
        opt.global_rank = 0
        opt.local_rank = 0
        opt.global_size = 1
        init_processes(0, opt.n_gpu_per_node, main, opt)