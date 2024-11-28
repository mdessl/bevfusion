from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
    build_channel_layer,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS

import os
from .base import Base3DFusionModel

__all__ = ["BEVFusion"]


@FUSIONMODELS.register_module()
class BEVFusion(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        save_embeddings: bool = False,
        save_path: str = None,
        precomputed: bool = False,
        embeddings_path: str = "embeddings/",
        **kwargs,
    ) -> None:
        super().__init__()

        self.save_embeddings = save_embeddings
        self.save_path = save_path
        if save_embeddings and save_path is None:
            raise ValueError("save_path must be specified when save_embeddings is True")

        self.precomputed = precomputed
        self.embeddings_path = embeddings_path

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
            if encoders["camera"].get("channel_layer") is not None:
                self.encoders["camera"]["channel_layer"] = build_channel_layer(encoders["camera"]["channel_layer"])
            
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        #for param in self.parameters():
        #    param.requires_grad = False

        # Unfreeze only the camera channel layer if it exists
        #if "camera" in self.encoders and "channel_layer" in self.encoders["camera"]:
        #    for param in self.encoders["camera"]["channel_layer"].parameters():
        #        param.requires_grad = True
            

    def init_weights(self) -> None:
        if not self.precomputed and "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def extract_camera_features(
        self,
        x,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        x = self.encoders["camera"]["vtransform"](
            x,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
        )
        if "channel_layer" in self.encoders["camera"]:
            x = self.encoders["camera"]["channel_layer"](x)
        return x

    def extract_lidar_features(self, x) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x)
        batch_size = coords[-1, 0] + 1
        x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders["lidar"]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        args = {
            "img": img,
            "points": points,
            "camera2ego": camera2ego,
            "lidar2ego": lidar2ego,
            "lidar2camera": lidar2camera,
            "lidar2image": lidar2image,
            "camera_intrinsics": camera_intrinsics,
            "camera2lidar": camera2lidar,
            "img_aug_matrix": img_aug_matrix,
            "lidar_aug_matrix": lidar_aug_matrix,
            "metas": metas,
            "gt_masks_bev": gt_masks_bev,
            "gt_bboxes_3d": gt_bboxes_3d,
            "gt_labels_3d": gt_labels_3d,
            **kwargs,
        }

        if isinstance(img, list):
            raise NotImplementedError
        if "channel_layer" in self.encoders["camera"] and self.training: #sbnet
            modality = args["metas"][0]["sbnet_modality"]
            return self.forward_sbnet(**args,modality=modality)  # camera or lidar

        elif "channel_layer" in self.encoders["camera"] and not self.training: #sbnet
            output_img = self.forward_sbnet(**args, modality="camera")
            output_lidar = self.forward_sbnet(**args, modality="lidar")
            return self.sbnet_forward_inference(output_img, output_lidar, args)
        else:
            outputs = self.forward_single(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        print("forward working")
        if self.save_embeddings:
            token = metas[0]['token']
            if "camera" in self.encoders:
                camera_feat = self.extract_camera_features(
                    img,
                    points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                )
                camera_path = os.path.join(self.save_path, f"{token}_camera.pth")
                torch.save(camera_feat, camera_path)

            if "lidar" in self.encoders:
                lidar_feat = self.extract_lidar_features(points)
                lidar_path = os.path.join(self.save_path, f"{token}_lidar.pth")
                torch.save(lidar_feat, lidar_path)
            if gt_masks_bev is not None:
                gt_masks_path = os.path.join(self.save_path, f"{token}_gt_masks.pth")
                torch.save(gt_masks_bev, gt_masks_path)
            
            return {}

        if self.precomputed:
            features = []
            token = metas[0]['token']
            camera_feat, lidar_feat, gt_masks_bev = self.load_embeddings(token)
            if camera_feat is not None:
                features.append(camera_feat)
            if lidar_feat is not None:
                features.append(lidar_feat)
        else:
            features = []
            for sensor in (
                self.encoders if self.training else list(self.encoders.keys())[::-1]
            ):
                print(sensor)
                if sensor == "camera":
                    feature = self.extract_camera_features(
                        img,
                        points,
                        camera2ego,
                        lidar2ego,
                        lidar2camera,
                        lidar2image,
                        camera_intrinsics,
                        camera2lidar,
                        img_aug_matrix,
                        lidar_aug_matrix,
                        metas,
                    )
                elif sensor == "lidar":
                    feature = self.extract_lidar_features(points)
                else:
                    raise ValueError(f"unsupported sensor: {sensor}")
                features.append(feature)
        if not self.training:
            features = features[::-1]
        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        batch_size = x.shape[0]

        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)

        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs

    def extract_features(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
    ) -> tuple:
        """Extract features using existing extraction methods.
        
        Returns:
            tuple: (camera_features, lidar_features)
        """

        camera_features = self.extract_camera_features(
            img,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            metas,
        ) if "camera" in self.encoders else None

        lidar_features = self.extract_lidar_features(points) if "lidar" in self.encoders else None

        return camera_features, lidar_features

    def load_embeddings(self, token):
        """Load precomputed embeddings for a sample."""
        camera_path = os.path.join(self.embeddings_path, f"{token}_camera.pth")
        lidar_path = os.path.join(self.embeddings_path, f"{token}_lidar.pth")
        gt_path = os.path.join(self.embeddings_path, f"{token}_gt_mask.pth")
        
        camera_feat = torch.load(camera_path) if os.path.exists(camera_path) else None
        lidar_feat = torch.load(lidar_path) if os.path.exists(lidar_path) else None
        gt_mask = torch.load(gt_path) if os.path.exists(gt_path) else None
        
        return camera_feat, lidar_feat, gt_mask

    def forward_sbnet(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        depths=None,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        modality="camera",  # New parameter to specify the modality
        **kwargs,
    ):
        features = []
        auxiliary_losses = {}
        if modality == "camera":
            feature = self.extract_camera_features(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas
            )

        elif modality == "lidar":
            feature = self.extract_lidar_features(points)

        else:
            raise ValueError(f"unsupported modality: {modality}")

        x = feature
        batch_size = x.shape[0]

        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)

        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            print(outputs)
            return outputs

    def sbnet_forward_inference(self, output_img, output_lidar, args):
        batch_size = len(output_img)
        combined_outputs = [{} for _ in range(batch_size)]

        for i in range(batch_size):

            img_is_zero = torch.all(args["img"][i] == 0)
            lidar_is_zero = torch.all(args["points"][i] == 0)

            if img_is_zero:
                # Only camera modality is present
                combined_outputs[i] = output_lidar[i]
            elif lidar_is_zero:
                # Only LiDAR modality is present
                combined_outputs[i] = output_img[i]
            elif not img_is_zero and not lidar_is_zero:
                # Both modalities are present, combine results
                for head_type in self.heads:
                    if head_type == "object":
                        boxes_3d = torch.cat(
                            [output_img[i]["boxes_3d"], output_lidar[i]["boxes_3d"]]
                        )
                        scores_3d = torch.cat(
                            [output_img[i]["scores_3d"], output_lidar[i]["scores_3d"]]
                        )
                        labels_3d = torch.cat(
                            [output_img[i]["labels_3d"], output_lidar[i]["labels_3d"]]
                        )
                        raise NotImplementedError
                        # Perform NMS
                        keep = self.nms_3d(boxes_3d, scores_3d, iou_threshold=0.5)

                        combined_outputs[i].update(
                            {
                                "boxes_3d": boxes_3d[keep],
                                "scores_3d": scores_3d[keep],
                                "labels_3d": labels_3d[keep],
                            }
                        )
                    elif head_type == "map":
                        combined_outputs[i] = output_img[i]
                        combined_outputs[i]["masks_bev"] = (
                            output_img[i]["masks_bev"] + output_lidar[i]["masks_bev"]
                        ) / 2

            else:
                raise ValueError(
                    f"Invalid modality combination: {img_modality}, {lidar_modality}"
                )

        return combined_outputs