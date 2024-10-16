from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F
import random
import os, json

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS


from .base import Base3DFusionModel

__all__ = ["BEVFusion", "SBNet"]


@FUSIONMODELS.register_module()
class BEVFusion(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        if os.path.exists("custom_args.json"):
            with open("custom_args.json", "r") as f:
                self.costum_args = json.load(f)
        else:
            self.costum_args = {}

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
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

        if encoders.get("radar") is not None:
            if encoders["radar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["radar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["radar"]["voxelize"])
            self.encoders["radar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["radar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["radar"].get("voxelize_reduce", True)

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

        # If the camera's vtransform is a BEVDepth version, then we're using depth loss.
        self.use_depth_loss = (
            (encoders.get("camera", {}) or {}).get("vtransform", {}) or {}
        ).get("type", "") in [
            "BEVDepth",
            "AwareBEVDepth",
            "DBEVDepth",
            "AwareDBEVDepth",
        ]


    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def extract_camera_features(
        self,
        x,
        points,
        radar_points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        gt_depths=None,
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
            radar_points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
            depth_loss=self.use_depth_loss,
            gt_depths=gt_depths,
        )
        return x

    def extract_features(self, x, sensor) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x, sensor)
        batch_size = coords[-1, 0] + 1
        x = self.encoders[sensor]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points, sensor):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders[sensor]["voxelize"](res)
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
        depths,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        if self.training:
            pass
        else:
            pass
            # img, points = self.create_zero_tensors(img, points, metas)
            # print("setting zero tensors")

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
            "depths": depths,
            "radar": radar,
            "gt_masks_bev": gt_masks_bev,
            "gt_bboxes_3d": gt_bboxes_3d,
            "gt_labels_3d": gt_labels_3d,
            **kwargs,
        }

        if isinstance(img, list):
            raise NotImplementedError

        # self.use_sbnet = True
        if False:  # inference with sbnet
            modality = args["metas"][0]["sbnet_modality"]
            return self.forward_sbnet(**args, modality=modality)  # camera or lidar
        elif True:
            lidar = self.forward_single_with_logits(**args)
            cam = kwargs["model_cam"](**args)
            
            import pdb; pdb.set_trace()
            #return 

        elif False:
            modality = args["metas"][0]["sbnet_modality"]
            img_temp = args["img"].clone()
            args["img"] = torch.zeros_like(img)
            lid = self.forward_single_with_logits(**args)
            args["img"] = img_temp
            args["points"] = [torch.zeros_like(p) for p in args["points"]]
            cam = self.forward_single_with_logits(**args)
            import pdb

            pdb.set_trace()
        elif False:  # self.use_sbnet and not self.training:  # inference with sbnet
            output_img = self.forward_sbnet(**args, modality="camera")
            output_lidar = self.forward_sbnet(**args, modality="lidar")
            return self.sbnet_forward_inference(output_img, output_lidar, args)
        else:
            # äimport pdb; pdb.set_trace()
            outputs = self.forward_single(**args)
            return outputs

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
                radar,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                gt_depths=depths,
            )
            if self.use_depth_loss:
                feature, auxiliary_losses["depth"] = feature[0], feature[-1]
        elif modality == "lidar":
            feature = self.extract_features(points, modality)
        elif modality == "radar":
            feature = self.extract_features(radar, modality)
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
            if self.use_depth_loss and modality == "camera":
                if "depth" in auxiliary_losses:
                    outputs["loss/depth"] = auxiliary_losses["depth"]
                else:
                    raise ValueError("Use depth loss is true, but depth loss not found")
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
        depths=None,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        features = []
        auxiliary_losses = {}
        feature_type = getattr(self, "feature_type", None)
        sbnet_modality = getattr(self, "sbnet_modality", None)

        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):

            if sensor == "camera":
                feature = self.extract_camera_features(
                    img,
                    points,
                    radar,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                    gt_depths=depths,
                )
                if self.use_depth_loss:
                    feature, auxiliary_losses["depth"] = feature[0], feature[-1]
            elif sensor == "lidar":
                feature = self.extract_features(points, sensor)
            elif sensor == "radar":
                feature = self.extract_features(radar, sensor)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")

            features.append(feature)

        if not self.training:
            # avoid OOM
            features = features[::-1]

        # Remove fusion step if only one feature type is used
        if len(features) == 1 or sbnet_modality:
            assert len(features) == 1, features
            x = features[0]
            print("wrong")
        elif self.fuser:
            x = self.fuser(features)
        else:
            raise ("error")
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
            if self.use_depth_loss:
                if "depth" in auxiliary_losses:
                    outputs["loss/depth"] = auxiliary_losses["depth"]
                else:
                    raise ValueError("Use depth loss is true, but depth loss not found")
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

    @auto_fp16(apply_to=("img", "points"))
    def forward_single_with_logits(
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
        feature_type = getattr(self, "feature_type", None)
        sbnet_modality = getattr(self, "sbnet_modality", None)

        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            # Skip processing if it's not the specified feature type
            if feature_type and sensor != feature_type:
                # import pdb; pdb.set_trace()
                continue

            if sensor == "camera":
                feature = self.extract_camera_features(
                    img,
                    points,
                    radar,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                    gt_depths=depths,
                )
                if self.use_depth_loss:
                    feature, auxiliary_losses["depth"] = feature[0], feature[-1]
            elif sensor == "lidar":
                feature = self.extract_features(points, sensor)
            elif sensor == "radar":
                feature = self.extract_features(radar, sensor)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")

            features.append(feature)

        if not self.training:
            # avoid OOM
            features = features[::-1]

        # Remove fusion step if only one feature type is used
        if len(features) == 1 or sbnet_modality:
            assert len(features) == 1, features
            x = features[0]
            print("wrong")
        elif self.fuser:
            x = self.fuser(features)
        else:
            raise ("error")
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
            if self.use_depth_loss and modality == "camera":
                if "depth" in auxiliary_losses:
                    outputs["loss/depth"] = auxiliary_losses["depth"]
                else:
                    raise ValueError("Use depth loss is true, but depth loss not found")
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
                                "x": x,
                                "metas": metas,
                                "head": head,
                                "pred_dict": pred_dict,
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


@FUSIONMODELS.register_module()
class SBNet(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
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

        if encoders.get("radar") is not None:
            if encoders["radar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["radar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["radar"]["voxelize"])
            self.encoders["radar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["radar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["radar"].get("voxelize_reduce", True)

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

        # If the camera's vtransform is a BEVDepth version, then we're using depth loss.
        self.use_depth_loss = (
            (encoders.get("camera", {}) or {}).get("vtransform", {}) or {}
        ).get("type", "") in [
            "BEVDepth",
            "AwareBEVDepth",
            "DBEVDepth",
            "AwareDBEVDepth",
        ]

        self.init_weights()

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def extract_camera_features(
        self,
        x,
        points,
        radar_points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        gt_depths=None,
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
            radar_points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
            depth_loss=self.use_depth_loss,
            gt_depths=gt_depths,
        )
        return x

    def extract_features(self, x, sensor) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x, sensor)
        batch_size = coords[-1, 0] + 1
        x = self.encoders[sensor]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points, sensor):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders[sensor]["voxelize"](res)
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
        depths,
        radar=None,
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
            "depths": depths,
            "radar": radar,
            "gt_masks_bev": gt_masks_bev,
            "gt_bboxes_3d": gt_bboxes_3d,
            "gt_labels_3d": gt_labels_3d,
            **kwargs,
        }

        if isinstance(img, list):
            raise NotImplementedError

        if self.training:  # inference with sbnet
            output_img = self.forward_single(**args, modality="camera")
            output_lidar = self.forward_single(**args, modality="lidar")
            return self.sbnet_forward_inference(output_img, output_lidar, args)
        else:
            modality = args["metas"][0]["sbnet_modality"]
            return self.forward_single(**args, modality=modality)  # camera or lidar

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
                radar,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                gt_depths=depths,
            )
            if self.use_depth_loss:
                feature, auxiliary_losses["depth"] = feature[0], feature[-1]
        elif modality == "lidar":
            feature = self.extract_features(points, modality)
        elif modality == "radar":
            feature = self.extract_features(radar, modality)
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
            if self.use_depth_loss and modality == "camera":
                if "depth" in auxiliary_losses:
                    outputs["loss/depth"] = auxiliary_losses["depth"]
                else:
                    raise ValueError("Use depth loss is true, but depth loss not found")
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
