"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random
import time
import datetime
import logging
import wandb

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.registry import registry
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners import *
from lavis.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.get("seed", 42)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across processes.
    job_id = now()

    cfg = Config(parse_args())
    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()
    logging.info("Started on {}".format(os.uname().nodename))

    wandb.init(
        project="evalmuse-competition",
        config={
            "architecture": "FGA-BLIP2",
            "dataset": "NITRE2025",
            "learning_rate": cfg.run_cfg.init_lr,
            "epochs": cfg.run_cfg.max_epoch,
            "batch_size": cfg.run_cfg.batch_size_train,
            "weight_decay": cfg.run_cfg.weight_decay,
            "warmup_steps": cfg.run_cfg.warmup_steps,
            "min_lr": cfg.run_cfg.min_lr,
            "warmup_lr": cfg.run_cfg.warmup_lr,
            "freeze_vit": cfg.model_cfg.freeze_vit,
            "image_size": cfg.model_cfg.image_size,
        }
    )

    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    runner_cls = get_runner_class(cfg)
    runner = runner_cls(cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets)

    # 添加wandb记录回调
    def log_metrics(metrics):
        if get_rank() == 0:  # 只在主进程记录
            wandb.log(metrics)
    
    runner.train(callback=log_metrics)
    wandb.finish()


if __name__ == "__main__":
    main()
