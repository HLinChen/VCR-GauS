import os
import sys
import argparse
sys.path.append(os.getcwd())

from configs.config import Config, recursive_update_strict, parse_cmdline_arguments
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', help='Path to the training config file.', required=True)
    parser.add_argument('--wandb', action='store_true', help="Enable using Weights & Biases as the logger")
    parser.add_argument('--wandb_name', default='default', type=str)
    args, cfg_cmd = parser.parse_known_args()
    return args, cfg_cmd


def main():
    args, cfg_cmd = parse_args()
    cfg = Config(args.config)

    cfg_cmd = parse_cmdline_arguments(cfg_cmd)
    recursive_update_strict(cfg, cfg_cmd)
    
    trainer = Trainer(cfg)
    cfg.save_config(cfg.logdir)
    
    trainer.init_wandb(cfg,
                       project=args.wandb_name,
                       mode="disabled" if cfg.train.debug_from > -1 or not args.wandb else "online",
                       use_group=True)
    
    trainer.train()
    trainer.finalize()
    
    return


if __name__ == "__main__":
    main()
