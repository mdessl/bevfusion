import mmcv
import torch
from mmcv.parallel import DataContainer as DC
from ensemble_boxes import *


def single_gpu_test(model, data_loader):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results

def single_gpu_test_with_ratio(model, data_loader, zero_tensor_ratio=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    modality, zero_tensor_ratio = zero_tensor_ratio
    num_zeros = int(len(data_loader) * zero_tensor_ratio)
    zero_indices = set(torch.randperm(len(data_loader))[:num_zeros].tolist())

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            if False:
                if i in zero_indices:
                    if modality == 'lidar':
                    #[torch.zeros_like(p) for p in data['points'].data[0]]
                        data['points'] = DC([torch.zeros_like(p) for p in data['points'].data[0][0]])
                    elif modality == 'camera':
                        data['img'] = DC(list(list(torch.zeros_like(data['img'].data[0][0]))))
            result = model(return_loss=False, rescale=True, **data)
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results




def single_gpu_test_2_models(model_lidar, model_camera, data_loader, zero_tensor_ratio=None):
    model_lidar.eval()
    model_camera.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    modality, zero_tensor_ratio = zero_tensor_ratio
    num_zeros = int(len(data_loader) * zero_tensor_ratio)
    
    # Randomly sample indices for zero tensors
    zero_indices = set(torch.randperm(len(data_loader))[:num_zeros].tolist())
    
    for i, data in enumerate(data_loader):
        with torch.no_grad():

            import pdb; pdb.set_trace()



            bevfusion.module.encoders.lidar = single_lidar.module.encoders.lidar 
            bevfusion.module.encoders.camera = single_camera.module.encoders.camera 


            res_lidar = model_lidar(return_loss=False, rescale=True, **data)
            res_cam = model_camera(return_loss=False, rescale=True, **data)

            if i in zero_indices:
                if modality == 'lidar':
                    result_tens = res_cam[0]['masks_bev']
                elif modality == 'camera':
                    result_tens = res_lidar[0]['masks_bev']
            else:
                result_tens = (res_lidar[0]['masks_bev'] + res_cam[0]['masks_bev']) / 2

            result = [{'masks_bev': result_tens, "gt_masks_bev":res_lidar[0]['gt_masks_bev']}]

        assert len(result) == 1 
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results

import numpy as np
import torch

def transform_to_list(data):
    """
    Transform each attribute of the input data to Python lists.
    
    Args:
    data (list): A list containing a dictionary with PyTorch tensors and other objects.
    
    Returns:
    list: A list containing a dictionary with Python lists.
    """
    def tensor_to_list(t):
        return t.cpu().tolist() if isinstance(t, torch.Tensor) else t

    result = []
    for item in data:
        list_item = {}
        for key, value in item.items():
            if key == 'boxes_3d':
                # Assuming 'boxes_3d' is the key for LiDARInstance3DBoxes
                list_item[key] = tensor_to_list(value.tensor)
            elif isinstance(value, torch.Tensor):
                list_item[key] = tensor_to_list(value)
            else:
                list_item[key] = value
        result.append(list_item)
    
    return result

# Usage

def single_gpu_test_2_models_bbox(model_lidar, model_camera, data_loader, zero_tensor_ratio=None):
    model_lidar.eval()
    model_camera.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    modality, zero_tensor_ratio = zero_tensor_ratio
    # Randomly sample indices for zero tensors
    num_zeros = int(len(data_loader) * zero_tensor_ratio)
    zero_indices = set(torch.randperm(len(data_loader))[:num_zeros].tolist())
    
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            

            if True: # we want to test the merging method. ideally, it would perform best than both single modality tests on bevfusion
                #import pdb; pdb.set_trace()
                data["points"].data[0][0] = torch.zeros_like(data["points"].data[0][0])
                res_lidar = model_lidar(return_loss=False, rescale=True, **data)
                res_lidar_np = transform_to_list(res_lidar)[0]
                boxes, scores, labels = weighted_boxes_fusion([res_lidar_np["boxes_3d"],res_lidar_np["boxes_3d"]], [res_lidar_np["scores_3d"],res_lidar_np["scores_3d"]], [res_lidar_np["labels_3d"],res_lidar_np["labels_3d"]])


                import pdb; pdb.set_trace()
                result_tens = res_lidar_np[0]['bboxes_3d']
                #res_cam = model_camera(return_loss=False, rescale=True, **data)
            elif i in zero_indices:
                res_lidar = model_lidar(return_loss=False, rescale=True, **data)
                res_cam = model_camera(return_loss=False, rescale=True, **data)
                if modality == 'lidar':
                    result_tens = res_cam[0]['bboxes_3d']
                elif modality == 'camera':
                    result_tens = res_lidar[0]['bboxes_3d']
            else:
                res_lidar = model_lidar(return_loss=False, rescale=True, **data)
                res_cam = model_camera(return_loss=False, rescale=True, **data)
                result_tens = (res_lidar[0]['masks_bev'] + res_cam[0]['masks_bev']) / 2

            result = [{'masks_bev': result_tens, "gt_masks_bev":res_lidar[0]['gt_masks_bev']}]

        assert len(result) == 1 
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results
