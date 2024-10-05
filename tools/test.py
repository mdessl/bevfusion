import argparse
import copy
import os
import warnings
import json
import matplotlib.pyplot as plt
import numpy as np

import mmcv
import torch
from torchpack.utils.config import configs
from torchpack import distributed as dist
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmdet3d.apis import single_gpu_test, single_gpu_test_2_models, single_gpu_test_with_ratio, single_gpu_test_2_models_bbox
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.utils import recursive_eval
from tools.utils import get_all_scenes, add_layer_channel_correction

def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", default=None, help="checkpoint file")
    parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation. It is"
        "useful when you want to format the result to a specific format and "
        "submit it to the test server",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
    )
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument("--show-dir", help="directory where results will be saved")
    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results.",
    )
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu-collect is not specified",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function (deprecate), "
        "change to --eval-options instead.",
    )
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument(
        '--feature-type',
        type=str,
        choices=['camera', 'lidar'],
        default=None,
        help='Specify a single feature type to use (camera or lidar). If not specified, use all available features.'
    )

    parser.add_argument(
        "--empty-tensor",
        choices=["none", "points", "img"],
        default="none",
        help="Replace LiDAR data with zero tensors for testing",
    )

    parser.add_argument(
        '--zero-tensor-ratio',
        type=float,
        choices=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        default=1.0,
        help='Ratio of scenes to replace the specified feature type with zero tensors (0.0 to 1.0)'
    )
    parser.add_argument(
        '--run-experiment',
        action='store_true',
        help='Run the experiment with multiple zero-tensor-ratio values'
    )
    parser.add_argument(
        '--use-sbnet',
        action='store_true',
        help='Run the experiment with multiple zero-tensor-ratio values'
    )
    parser.add_argument(
        '--two-pretrained',
        action='store_true',
        help='Run the experiment with multiple zero-tensor-ratio values'
    )
    parser.add_argument(
        '--plot-output',
        type=str,
        default='evaluation_plot.png',
        help='Output file path for the evaluation plot'
    )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            "--options and --eval-options cannot be both specified, "
            "--options is deprecated in favor of --eval-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --eval-options")
        args.eval_options = args.options

    with open('custom_args.json', 'w') as f:
        json.dump({'empty_tensor':getattr(args, 'empty_tensor'), 'feature_type':getattr(args, 'feature_type')}, f)


    return args

def run_experiment(args):

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    assert args.out or args.eval or args.format_only or args.show or args.show_dir, (
        "Please specify at least one operation (save/eval/format/show the "
        'results / save the results) with the argument "--out", "--eval"'
        ', "--format-only", "--show" or "--show-dir"'
    )

    if args.eval and args.format_only:
        raise ValueError("--eval and --format_only cannot be both specified")

    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test]
        )
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
    
    #if args.feature_type is not None and args.zero_tensor_ratio:
    #    cfg.data.test.pipeline.insert(9,dict(type='AddMissingModality', zero_ratio=args.zero_tensor_ratio, zero_modality=args.feature_type))
    
    samples_per_gpu = 1
    #import pdb; pdb.set_trace()
    distributed = False

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    custom_args = {
        'empty_tensor': getattr(args, 'empty_tensor'),
        'feature_type': getattr(args, 'feature_type'),
        'use_sbnet': getattr(args, 'use_sbnet'),
        'zero_tensor_ratio': args.zero_tensor_ratio
    }
    
    # Always include all_scenes
    #all_scenes = np.unique([d["metas"].data["scene_token"] for d in dataset]).tolist()
    custom_args['all_scenes'] = get_all_scenes(cfg.data.val.ann_file)

    with open('custom_args.json', 'w') as f:
        json.dump(custom_args, f)
    print(custom_args)


    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.use_sbnet and False:
        model = add_layer_channel_correction(model, state_dict_path=args.checkpoint)
    if False:
        #checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
        model = torch.load(args.checkpoint)
        if "CLASSES" in checkpoint.get("meta", {}):
            model.CLASSES = checkpoint["meta"]["CLASSES"]
        else:
            model.CLASSES = dataset.CLASSES
    else:
        model.CLASSES = dataset.CLASSES
        #model = torch.load(args.checkpoint)

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if not distributed:
        if True: # test on pretrained single modality models
            if "bbox" in args.eval:
                #model, model_lidar = get_pretrained_single_modality_models_bbox(cfg)
                model = MMDataParallel(model, device_ids=[0])
                outputs = single_gpu_test_2_models_bbox(model, model, data_loader, (args.feature_type, args.zero_tensor_ratio))
            elif "map" in args.eval:
                model, model_lidar = get_pretrained_single_modality_models_seg(cfg)
                outputs = single_gpu_test_2_models(model, model_lidar, data_loader, (args.feature_type, args.zero_tensor_ratio))
        elif False:
            model = MMDataParallel(model, device_ids=[0])
            outputs = single_gpu_test_with_ratio(model, data_loader, (args.feature_type, args.zero_tensor_ratio))
        else:
            model = MMDataParallel(model, device_ids=[0])
            outputs = single_gpu_test(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f"\nwriting results to {args.out}")
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get("evaluation", {}).copy()
            for key in [
                "interval",
                "tmpdir",
                "start",
                "gpu_collect",
                "save_best",
                "rule",
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            result = dataset.evaluate(outputs, **eval_kwargs)
            return result

def get_pretrained_single_modality_models_seg(cfg):

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg")) # model is camera
    model_lidar = build_model(cfg.model_lidar, test_cfg=cfg.get("test_cfg"))
    wrap_fp16_model(model)
    wrap_fp16_model(model_lidar)
    checkpoint = load_checkpoint(model, "pretrained/camera-only-seg.pth", map_location="cpu")
    checkpoint = load_checkpoint(model_lidar, "pretrained/lidar-only-seg.pth", map_location="cpu")
    model = MMDataParallel(model, device_ids=[0])
    model_lidar = MMDataParallel(model_lidar, device_ids=[0])
    return model, model_lidar


def get_pretrained_single_modality_models_bbox(cfg): # the point here is the use the exact same model twice 

    configs.load("configs/nuscenes/det/centerhead/lssfpn/camera/256x704/resnet/default.yaml", recursive=True)
    cfg_camera = Config(recursive_eval(configs), filename="/root/bevfusion/configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml")
    configs.load("configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet.yaml", recursive=True)
    cfg_lidar = Config(recursive_eval(configs), filename="configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml")

    model = build_model(cfg_camera.model, test_cfg=cfg_camera.get("test_cfg")) # model is camera
    model_lidar = build_model(cfg_lidar.model, test_cfg=cfg_lidar.get("test_cfg"))
    wrap_fp16_model(model)
    wrap_fp16_model(model_lidar)

    checkpoint = load_checkpoint(model, "pretrained/camera-only-det.pth", map_location="cpu")
    checkpoint = load_checkpoint(model_lidar, "pretrained/lidar-only-det.pth", map_location="cpu")

    import pdb; pdb.set_trace()

    model = MMDataParallel(model, device_ids=[0])
    model_lidar = MMDataParallel(model_lidar, device_ids=[0])
    return model, model_lidar

def get_pretrained_single_modality_models_bbox(cfg):

    configs.load("configs/nuscenes/det/centerhead/lssfpn/camera/256x704/resnet/default.yaml", recursive=True)
    cfg_camera = Config(recursive_eval(configs), filename="/root/bevfusion/configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml")
    configs.load("configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet.yaml", recursive=True)
    cfg_lidar = Config(recursive_eval(configs), filename="configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml")

    model = build_model(cfg_camera.model, test_cfg=cfg_camera.get("test_cfg")) # model is camera
    model_lidar = build_model(cfg_lidar.model, test_cfg=cfg_lidar.get("test_cfg"))
    wrap_fp16_model(model)
    wrap_fp16_model(model_lidar)

    checkpoint = load_checkpoint(model, "pretrained/camera-only-det.pth", map_location="cpu")
    checkpoint = load_checkpoint(model_lidar, "pretrained/lidar-only-det.pth", map_location="cpu")

    import pdb; pdb.set_trace()

    model = MMDataParallel(model, device_ids=[0])
    model_lidar = MMDataParallel(model_lidar, device_ids=[0])
    return model, model_lidar

def plot_results(zero_tensor_ratios, results, task, modality, output_file):
    plt.figure(figsize=(10, 6))
    plt.plot(zero_tensor_ratios, results, marker='o')
    plt.xlabel('Zero Tensor Ratio (on scene level)')
    plt.ylabel(f'{task.upper()}')
    plt.title(f'Performance when {modality} modality is replaced by zero tensor with different ratios')
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()
    print(f"Evaluation plot saved to {output_file}")


def main():
    args = parse_args()
    dist.init()

    if args.run_experiment:
        zero_tensor_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0] # 
        tasks = {'map':"map/mean/iou@max"}
        modalities = ['lidar','camera']

        for name, metric in tasks.items():
            for modality in modalities:
                args.empty_tensor = 'img' if modality == 'camera' else 'points'
                args.feature_type = modality
                args.eval = [name]
                results = []
                results_dict = {}

                for ratio in zero_tensor_ratios:
                    args.zero_tensor_ratio = ratio
                    result = run_experiment(args)
                    print(result)
                    results.append(result[metric])
                    results_dict[ratio] = result[metric]
                print(results_dict)
                with open(f'results_dict_{args.empty_tensor}.json', 'w') as f:
                    json.dump(results_dict, f)
                print(f"Results for {name} task, when {modality} modality is not present 0-100% of the time: {results}")
                
                output_file = f'{args.plot_output}_{name}_{modality}.png'
                plot_results(zero_tensor_ratios, results, name, modality, output_file)
    else:
        result = run_experiment(args)
        print(f"Evaluation result for zero_tensor_ratio {args.zero_tensor_ratio}: {result}")

if __name__ == "__main__":
    main()