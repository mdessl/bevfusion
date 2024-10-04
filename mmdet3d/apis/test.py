import mmcv
import torch


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
