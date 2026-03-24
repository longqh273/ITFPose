
# Copyright (c) OpenMMLab. All rights reserved.
#from mmpose.registry import DATASETS
#from ..base import BaseCocoStyleDataset

import os
import warnings
from collections import OrderedDict, defaultdict

import json_tricks as json
import numpy as np
from mmcv import Config
from xtcocotools.cocoeval import COCOeval

from ....core.post_processing import oks_nms, soft_oks_nms
from ...builder import DATASETS
from ..base import Kpt2dSviewRgbImgTopDownDataset

@DATASETS.register_module()
class AnimalKingdomDataset(Kpt2dSviewRgbImgTopDownDataset):
    """AnimalKingdomDataset for animal pose estimation (top-down).

    "[CVPR2022] Animal Kingdom: A Large and Diverse Dataset for Animal Behavior Understanding"

    Animal Kingdom keypoint indexes::

        0: 'Head_Mid_Top',
        1: 'Eye_Left',
        2: 'Eye_Right',
        3: 'Mouth_Front_Top',
        4: 'Mouth_Back_Left',
        5: 'Mouth_Back_Right',
        6: 'Mouth_Front_Bottom',
        7: 'Shoulder_Left',
        8: 'Shoulder_Right',
        9: 'Elbow_Left',
        10: 'Elbow_Right',
        11: 'Wrist_Left',
        12: 'Wrist_Right',
        13: 'Torso_Mid_Back',
        14: 'Hip_Left',
        15: 'Hip_Right',
        16: 'Knee_Left',
        17: 'Knee_Right',
        18: 'Ankle_Left',
        19: 'Ankle_Right',
        20: 'Tail_Top_Back',
        21: 'Tail_Mid_Back',
        22: 'Tail_End_Back'

    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):

        if dataset_info is None:
            warnings.warn(
                'dataset_info is missing. '
                'Load default dataset_info from config.', DeprecationWarning)
            cfg = Config.fromfile('configs/_base_/datasets/ak.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode
        )

        self.db = self._get_db()
        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    def _get_db(self):
        """Load dataset and parse keypoints."""
        gt_db = []
        bbox_id = 0
        num_joints = self.ann_info['num_joints']

        for img_id in self.img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            objs = self.coco.loadAnns(ann_ids)

            for obj in objs:
                if max(obj['keypoints']) == 0:
                    continue
                joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
                joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

                keypoints = np.array(obj['keypoints']).reshape(-1, 3)
                joints_3d[:, :2] = keypoints[:, :2]
                joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

                # Compute center/scale for the bbox
                x, y, w, h = obj['bbox']
                center, scale = self._xywh2cs(x, y, w, h, 1.25)

                image_file = os.path.join(self.img_prefix, self.id2name[img_id])

                gt_db.append({
                    'image_file': image_file,
                    'center': center,
                    'scale': scale,
                    'rotation': 0,
                    'joints_3d': joints_3d,
                    'joints_3d_visible': joints_3d_visible,
                    'dataset': self.dataset_name,
                    'bbox': obj['bbox'],
                    'bbox_score': 1,
                    'bbox_id': bbox_id
                })
                bbox_id += 1

        gt_db = sorted(gt_db, key=lambda x: x['bbox_id'])
        return gt_db

    def evaluate(self, outputs, res_folder, metric='PCK', **kwargs):
        """Evaluate keypoint results. Similar to AnimalLocustDataset."""
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['PCK', 'AUC', 'EPE']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        res_file = os.path.join(res_folder, 'result_keypoints.json')
        kpts = []
        for output in outputs:
            preds = output['preds']
            boxes = output['boxes']
            image_paths = output['image_paths']
            bbox_ids = output['bbox_ids']
            batch_size = len(image_paths)
            for i in range(batch_size):
                image_id = self.name2id[image_paths[i][len(self.img_prefix):]]
                kpts.append({
                    'keypoints': preds[i].tolist(),
                    'center': boxes[i][0:2].tolist(),
                    'scale': boxes[i][2:4].tolist(),
                    'area': float(boxes[i][4]),
                    'score': float(boxes[i][5]),
                    'image_id': image_id,
                    'bbox_id': bbox_ids[i]
                })

        kpts = self._sort_and_unique_bboxes(kpts)
        self._write_keypoint_results(kpts, res_file)
        info_str = self._report_metric(res_file, metrics)
        return OrderedDict(info_str)
