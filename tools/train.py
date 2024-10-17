import argparse
import copy
import os
import random
import time

import numpy as np
import torch
from mmcv import Config
from torchpack import distributed as dist
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs
from mmcv.runner import load_checkpoint

from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset, nuscenes_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval

import torch.nn as nn
import torch


def add_layer_channel_correction(
    model, output_channels=256, state_dict_path="pretrained/bevfusion-seg.pth"
):
    # Get the current layers of the downsample module
    current_layers = list(model.encoders.camera.vtransform.downsample.children())

    # Get the number of input channels from the last convolutional layer
    input_channels = current_layers[
        -3
    ].out_channels  # Assuming the last Conv2d is 3 positions from the end

    # Create the new layers you want to add
    new_conv = nn.Conv2d(
        input_channels,
        output_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        bias=False,
    )
    new_bn = nn.BatchNorm2d(output_channels)
    new_relu = nn.ReLU(inplace=True)
    det = torch.load("pretrained/bevfusion-det.pth")["state_dict"].keys()
    # Add the new conv layer at index 9 (after the last ReLU)
    current_layers.insert(9, new_conv)

    # Add the new ReLU at index 10
    current_layers.insert(10, new_bn)
    current_layers.insert(11, new_relu)
    # Create a new Sequential module with the updated layers
    new_downsample = nn.Sequential(*current_layers)

    model.encoders.camera.vtransform.downsample = new_downsample
    return model


def main():
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument(
        "--checkpoint", help="checkpoint file"
    )  # pretrained/bevfusion-seg.pth
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)
    cfg.run_dir = args.run_dir

    # dump config
    cfg.dump(os.path.join(cfg.run_dir, "configs.yaml"))

    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg.run_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)

    # log some basic info
    logger.info(f"Config:\n{cfg.pretty_text}")

    # set random seeds
    if cfg.seed is not None:
        logger.info(
            f"Set random seed to {cfg.seed}, "
            f"deterministic mode: {cfg.deterministic}"
        )
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    model = build_model(cfg.model)
    model.init_weights()
    # m = torch.load("pretrained/model.pt")
    # model.encoders.lidar = m.encoders.lidar
    #model = add_layer_channel_correction(model, 256)
    # state_dict = torch.load("runs/sbnet_3epochs/epoch_3.pth")["state_dict"]
    # model.load_state_dict(state_dict)

    # lidar = torch.load("pt/lidar.pt")
    # camera = torch.load("pt/camera.pt")
    # model.encoders.camera = lidar.module.encoders.camera
    # model.encoders.lidar = camera.module.encoders.lidar

    # for param in model.encoders.parameters():
    #    param.requires_grad = False
    # for param in model.encoders.camera.vtransform.downsample.parameters():
    #    param.requires_grad = True

    datasets = [build_dataset(cfg.data.train)]

    # datasets[0].dataset = nuscenes_dataset.MergedNuScenesDataset(datasets[0].dataset, datasets[0].dataset, ann_file=datasets[0].dataset.ann_file)

    # import pdb; pdb.set_trace()

    if cfg.get("sync_bn", None):
        if not isinstance(cfg["sync_bn"], dict):
            cfg["sync_bn"] = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])

    logger.info(f"Model:\n{model}")
    train_model(
        model,
        datasets,
        cfg,
        distributed=True,
        validate=True,
        timestamp=timestamp,
    )


if __name__ == "__main__":
    main()
