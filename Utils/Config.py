import yaml
from easydict import EasyDict
import os
from Utils.Logger import print_log
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="Configurations/ModelNet40.yaml", help="configuration file path")
    parser.add_argument("--train", type=bool, default=True, help="train serial_mode")
    args = parser.parse_args()
    return args


def log_args_to_file(args, pre='args', logger=None):
    for key, val in args.__dict__.items():
        print_log(f'{pre}.{key}: {val}', logger=logger)


def log_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            print_log(f'{pre}.{key} = edict()', logger=logger)
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        print_log(f'{pre}.{key}: {val}', logger=logger)


def save_experiment_config(args, config, logger=None):
    config_path = os.path.join(args.experiment_path, 'config.yaml')
    os.system('cp %s %s' % (args.config, config_path))
    print_log(f'Copy the Config file from {args.config} to {config_path}', logger=logger)


def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(new_config['_base_'], 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config


def cfg_from_yaml_file(config, cfg_file):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
    merge_new_config(config=config, new_config=new_config)
    return config


def get_config(config_path, logger=None):
    cfgs = cfg_from_yaml_file(EasyDict(), config_path)
    return cfgs
