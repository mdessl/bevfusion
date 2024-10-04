import torch
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import (
    DistSamplerSeedHook,
    EpochBasedRunner,
    GradientCumulativeFp16OptimizerHook,
    Fp16OptimizerHook,
    OptimizerHook,
    build_optimizer,
    build_runner,
)
from mmdet3d.runner import CustomEpochBasedRunner

from mmdet3d.utils import get_root_logger
from mmdet.core import DistEvalHook
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor


import torch.nn as nn

def add_layer_channel_correction(model, output_channels=256, state_dict_path="pretrained/bevfusion-seg.pth"):
    # Get the current layers of the downsample module
    current_layers = list(model.encoders.camera.vtransform.downsample.children())
    
    # Get the number of input channels from the last convolutional layer
    input_channels = current_layers[-3].out_channels  # Assuming the last Conv2d is 3 positions from the end
    
    # Create the new layers you want to add
    new_conv = nn.Conv2d(input_channels, output_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    new_relu = nn.ReLU(inplace=True)
    
    # Add the new conv layer at index 9 (after the last ReLU)
    current_layers.insert(9, new_conv)
    
    # Add the new ReLU at index 10
    current_layers.insert(10, new_relu)
    
    # Create a new Sequential module with the updated layers
    new_downsample = nn.Sequential(*current_layers)
    
    model.encoders.camera.vtransform.downsample = new_downsample
    
    # Load the state dict if a path is provided
    if state_dict_path:
        state_dict = torch.load(state_dict_path)
        model.load_state_dict(state_dict, strict=False)
    
    # Initialize the new conv layer if it's not in the loaded state dict
    if 'encoders.camera.vtransform.downsample.9.weight' not in model.state_dict():
        model_state_dict = model.state_dict()
        model_state_dict['encoders.camera.vtransform.downsample.9.weight'] = new_conv.weight
        model.load_state_dict(model_state_dict)
    
    return model


def train_model(
    model,
    dataset,
    cfg,
    distributed=False,
    validate=False,
    timestamp=None,
):
    logger = get_root_logger()

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            None,
            dist=distributed,
            seed=cfg.seed,
        )
        for ds in dataset
    ]


    if cfg.get("freeze_sbnet", None):
        #for param in model.encoders.parameters():
        #    param.requires_grad = False
        
        state_dict_path = "pretrained/bevfusion-seg.pth"
        if len(list(model.encoders.camera.vtransform.downsample.children())) == 9:
            model = add_layer_channel_correction(model, 256, state_dict_path) # from 80 zo 25
        else:
            state_dict = torch.load(state_dict_path)
            model.load_state_dict(state_dict, strict=False)
        for param in model.encoders.camera.vtransform.downsample.parameters():
            param.requires_grad = True

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # put model on gpus
    #print("find_unused_parameters was set to False before in apis, train.py")
    find_unused_parameters = cfg.get("find_unused_parameters", True)
    # Sets the `find_unused_parameters` parameter in
    # torch.nn.parallel.DistributedDataParallel
    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
        find_unused_parameters=find_unused_parameters,
    )

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.run_dir,
            logger=logger,
            meta={},
        ),
    )
    
    if hasattr(runner, "set_dataset"):
        runner.set_dataset(dataset)

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        if "cumulative_iters" in cfg.optimizer_config:
            optimizer_config = GradientCumulativeFp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed
            )
        else:
            optimizer_config = Fp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed
            )
    elif distributed and "type" not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get("momentum_config", None),
    )
    if isinstance(runner, EpochBasedRunner):
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop("samples_per_gpu", 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
        )
        eval_cfg = cfg.get("evaluation", {})
        eval_cfg["by_epoch"] = cfg.runner["type"] != "IterBasedRunner"
        eval_hook = DistEvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)


    runner.run(data_loaders, [("train", 1)])
