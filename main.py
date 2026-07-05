import argparse
import datetime
import glob
import os
import sys
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.trainer import Trainer

from utils.helpers import instantiate_from_config
from spamo.callbacks import SetupCallback, MetricsTableCallback

def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='SpaMo training and evaluation')
    parser.add_argument('-c', '--config', nargs='*', metavar='base_config.yaml', default=list())
    parser.add_argument('-t', '--train', type=str2bool, default=True, nargs='?')
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-f', '--fast_dev_run', action='store_true', default=False)
    parser.add_argument('-n', '--name', type=str, const=True, default='', nargs='?')
    parser.add_argument('--postfix', type=str, default='')
    parser.add_argument('-l', '--logdir', type=str, default='logs')
    parser.add_argument('-r', '--resume', default=None)
    parser.add_argument('--no_test', type=bool, default=True)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('-e', '--evaluation', type=str, default='mse')
    return parser

def load_configs(config_paths: List[str]) -> OmegaConf:
    configs = [OmegaConf.load(cfg) for cfg in config_paths]
    return OmegaConf.merge(*configs)

def setup_logging_dirs(opt: argparse.Namespace) -> tuple:
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError(f"Cannot find checkpoint directory: {opt.resume}")
            
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "checkpoints", opt.ckpt) if opt.ckpt else None
        nowname = logdir.split("/")[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.config:
            cfg_fname = os.path.split(opt.config[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)
        ckpt = opt.ckpt
    
    return logdir, ckpt, nowname

def configure_callbacks(opt: argparse.Namespace, model: pl.LightningModule, ckptdir: str, lightning_config: OmegaConf, logdir: str, now: str, config: OmegaConf) -> List:
    callbacks = [instantiate_from_config(lightning_config.callback[callback]) for callback in lightning_config.callback.keys()]
    callbacks.append(MetricsTableCallback())

    if opt.evaluation == "bleu":
        callbacks.append(ModelCheckpoint(
            dirpath=ckptdir, 
            filename="epoch={epoch:05}-step={step:07}-bleu4={val/bleu4:.2f}", 
            monitor=model.monitor, 
            auto_insert_metric_name=False, 
            save_top_k=1, 
            mode="max"
        ))
        callbacks.append(EarlyStopping(
            monitor=model.monitor, verbose=True, patience=50, mode="max"
        ))
    else:
        callbacks.append(ModelCheckpoint(
            dirpath=ckptdir, 
            filename="epoch={epoch:05}-step={step:07}-loss={val/contra_loss:.4f}", 
            monitor=model.monitor, 
            auto_insert_metric_name=False, 
            save_top_k=1, 
            mode="min"
        ))
        callbacks.append(EarlyStopping(
            monitor=model.monitor, verbose=True, patience=50, mode="min"
        ))
    
    callbacks.append(SetupCallback(
        resume=opt.resume, 
        now=now, 
        logdir=logdir, 
        ckptdir=ckptdir, 
        cfgdir=os.path.join(logdir, "configs"),
        config=config, 
        lightning_config=lightning_config
    ))
    
    return callbacks

def configure_logger(logger_type: str, logdir: str, nowname: str) -> Dict:
    logger_configs = {
        "wandb": {"target": "pytorch_lightning.loggers.WandbLogger", "params": {"name": nowname, "save_dir": logdir, "id": nowname}},
        "testtube": {"target": "pytorch_lightning.loggers.TestTubeLogger", "params": {"name": "testtube", "save_dir": logdir}},
        "tensorboard": {"target": "pytorch_lightning.loggers.TensorBoardLogger", "params": {"name": nowname, "save_dir": logdir}}
    }
    
    if logger_type not in logger_configs:
        logger_type = "tensorboard"
        
    return logger_configs[logger_type]

def main():
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())
    
    parser = get_parser()
    opt, _ = parser.parse_known_args()
    
    if opt.name and opt.resume:
        raise ValueError("-n/--name and -r/--resume cannot be specified both.")
    
    logdir, ckpt, nowname = setup_logging_dirs(opt)
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")

    seed_value = opt.seed
    pl.seed_everything(seed_value, workers=True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if opt.resume or opt.test:
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.config = base_configs + opt.config
        
    config = load_configs(opt.config)
    lightning_config = config.pop("lightning", OmegaConf.create())
    
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    if opt.fast_dev_run:
        trainer_config["fast_dev_run"] = True
    trainer_opt = argparse.Namespace(**trainer_config)
    
    from pytorch_lightning.strategies import DDPStrategy
    if not hasattr(trainer_opt, "strategy") or trainer_opt.strategy is None:
        trainer_opt.strategy = DDPStrategy(find_unused_parameters=True)

    lightning_config.trainer = trainer_config
    
    data = instantiate_from_config(config.data)
    data.setup()
    
    model = instantiate_from_config(config.model)
    
    if not opt.fast_dev_run:
        logger_cfg = configure_logger("tensorboard", logdir, nowname)
        trainer_opt.logger = instantiate_from_config(logger_cfg)
        trainer_opt.callbacks = configure_callbacks(opt, model, ckptdir, lightning_config, logdir, now, config)
    
    trainer = Trainer(**vars(trainer_opt))
    
    if opt.train:
        if opt.resume is not None:
            trainer.fit(model, data, ckpt_path=ckpt)
        else:
            if ckpt is not None:
                model.load_pretrained_weights(ckpt)
            trainer.fit(model, data)
            
            if not opt.no_test:
                trainer.test(model, data)
    elif opt.test:
        train_loader = data.train_dataloader()
        trainer.test(model, dataloaders=train_loader, ckpt_path=ckpt)
        trainer.validate(model, data, ckpt_path=ckpt) 
        trainer.test(model, data, ckpt_path=ckpt)

if __name__ == '__main__':
    main()
