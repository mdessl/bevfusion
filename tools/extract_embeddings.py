import argparse
import torch
import mmcv
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from torchpack.utils.config import configs
from torchpack import distributed as dist
from torchpack.environ import auto_set_run_dir, set_run_dir
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.utils import get_root_logger, recursive_eval
import os

def main():
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--out-dir", required=True, help="directory to save embeddings")
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")
    args, opts = parser.parse_known_args()

    # Load config
    configs.load(args.config, recursive=True)
    configs.update(opts)
    cfg = Config(recursive_eval(configs), filename=args.config)

    # Set up environment
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    # Set up run directory
    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)
    cfg.run_dir = args.run_dir

    # Create output directory
    mmcv.mkdir_or_exist(os.path.abspath(args.out_dir))


    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')
    )

    # Enable embedding saving
    model.save_embeddings = True
    model.save_path = args.out_dir

    # Load checkpoint
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    # Set up model
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    # Build dataset and dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False
    )

    # Extract embeddings
    prog_bar = mmcv.ProgressBar(len(dataset))
    
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            model.forward(**data)  # Will save embeddings automatically
            prog_bar.update()


if __name__ == "__main__":
    main()