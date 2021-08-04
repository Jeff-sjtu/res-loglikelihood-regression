# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

"""MPII Human keypoint dataset."""
import copy
import os
import pickle as pk

import numpy as np
import scipy.misc
import torch.utils.data as data
from pycocotools.coco import COCO

from rlepose.utils.bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy
from rlepose.utils.presets import SimpleTransform3D


class Mpii(data.Dataset):
    """ MPII Human Pose Dataset.
    Parameters
    ----------
    root: str, default './data/mpii'
        Path to the mpii dataset.
    train: bool, default is True
        If true, will set as training mode.
    dpg: bool, default is False
        If true, will activate `dpg` for data augmentation.
    """

    CLASSES = ['person']
    num_joints = 16
    bbox_3d_shape = (2000, 2000, 2000)
    joints_name = ('right_ankle', 'right_knee', 'right_hip',  # 2
                   'left_hip', 'left_knee', 'left_ankle',   # 5
                   'pelv', 'thrx', 'neck', 'head',  # 9
                   'right_wrist', 'right_elbow', 'right_shoulder',  # 12
                   'left_shoulder', 'left_elbow', 'left_wrist')  # 15
    skeleton = ((3, 6), (4, 3), (5, 4),
                (2, 6), (1, 2), (0, 1),
                (7, 6), (8, 7), (9, 8),
                (13, 7), (14, 13), (15, 14),
                (12, 7), (11, 12), (10, 11))
    mean_bone_len = None

    def __init__(self,
                 cfg,
                 ann_file,
                 root='./data/mpii',
                 train=True,
                 skip_empty=True,
                 dpg=False,
                 lazy_import=False):

        self._cfg = cfg
        self._preset_cfg = cfg['PRESET']
        self._ann_file = os.path.join(root, 'annotations', ann_file)
        self._lazy_import = lazy_import
        self._root = root
        self._skip_empty = skip_empty
        self._train = train
        self._dpg = dpg

        self._scale_factor = self._preset_cfg.SCALE_FACTOR
        self._color_factor = self._preset_cfg.COLOR_FACTOR
        self._rot = self._preset_cfg.ROT_FACTOR
        self._input_size = self._preset_cfg.IMAGE_SIZE
        self._output_size = self._preset_cfg.HEATMAP_SIZE

        self._occlusion = self._preset_cfg.OCCLUSION
        # self._occlusion = False

        self._sigma = self._preset_cfg.SIGMA
        self._img_prefix = 'images'

        self._check_centers = False

        self.num_class = len(self.CLASSES)

        self.num_joints_half_body = self._preset_cfg.NUM_JOINTS_HALF_BODY
        self.prob_half_body = self._preset_cfg.PROB_HALF_BODY

        self._loss_type = cfg['heatmap2coord']

        self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13, 14, 15)
        self.lower_body_ids = (0, 1, 2, 3, 4, 5, 6)

        self.transformation = SimpleTransform3D(
            self, scale_factor=self._scale_factor,
            color_factor=self._color_factor,
            occlusion=self._occlusion,
            input_size=self._input_size,
            output_size=self._output_size,
            bbox_3d_shape=self.bbox_3d_shape,
            rot=self._rot, sigma=self._sigma,
            train=self._train, add_dpg=self._dpg,
            loss_type=self._loss_type)

        self._items, self._labels = self._lazy_load_json()

    def __getitem__(self, idx):
        # get image id
        img_path = self._items[idx]
        img_id = int(os.path.splitext(os.path.basename(img_path))[0])

        # load ground truth, including bbox, keypoints, image size
        label = copy.deepcopy(self._labels[idx])
        img = scipy.misc.imread(img_path, mode='RGB')
        # transform ground truth into training label and apply data augmentation
        target = self.transformation(img, label)

        img = target.pop('image')
        bbox = target.pop('bbox')
        return img, target, img_id, bbox

    def __len__(self):
        return len(self._items)

    def _lazy_load_ann_file(self):
        if os.path.exists(self._ann_file + '.pkl') and self._lazy_import:
            print('Lazy load json...')
            with open(self._ann_file + '.pkl', 'rb') as fid:
                return pk.load(fid)
        else:
            _coco = COCO(self._ann_file)
            try:
                with open(self._ann_file + '.pkl', 'wb') as fid:
                    pk.dump(_coco, fid, pk.HIGHEST_PROTOCOL)
            except Exception as e:
                print(e)
                print('Skip writing to .pkl file.')
            return _coco

    def _lazy_load_json(self):
        if os.path.exists(self._ann_file + '_annot_keypoint.pkl') and self._lazy_import:
            print('Lazy load annot...')
            with open(self._ann_file + '_annot_keypoint.pkl', 'rb') as fid:
                items, labels = pk.load(fid)
        else:
            items, labels = self._load_jsons()
            try:
                with open(self._ann_file + '_annot_keypoint.pkl', 'wb') as fid:
                    pk.dump((items, labels), fid, pk.HIGHEST_PROTOCOL)
            except Exception as e:
                print(e)
                print('Skip writing to .pkl file.')

        return items, labels

    def _load_jsons(self):
        """Load all image paths and labels from annotation files into buffer."""
        items = []
        labels = []

        _mpii = self._lazy_load_ann_file()
        classes = [c['name'] for c in _mpii.loadCats(_mpii.getCatIds())]
        assert classes == self.CLASSES, "Incompatible category names with MPII. "

        # iterate through the annotations
        image_ids = sorted(_mpii.getImgIds())
        for entry in _mpii.loadImgs(image_ids):
            filename = entry['file_name']
            abs_path = os.path.join(self._root, self._img_prefix, filename)
            if False and not os.path.exists(abs_path):
                raise IOError('Image: {} not exists.'.format(abs_path))
            label = self._check_load_keypoints(_mpii, entry)
            if not label:
                continue

            # num of items are relative to person, not image
            for obj in label:
                items.append(abs_path)
                labels.append(obj)

        return items, labels

    def _check_load_keypoints(self, _mpii, entry):
        """Check and load ground-truth keypoints"""
        ann_ids = _mpii.getAnnIds(imgIds=entry['id'], iscrowd=False)
        objs = _mpii.loadAnns(ann_ids)
        # check valid bboxes
        valid_objs = []
        width = entry['width']
        height = entry['height']

        for obj in objs:
            if max(obj['keypoints']) == 0:
                continue
            # convert from (x, y, w, h) to (xmin, ymin, xmax, ymax) and clip bound
            xmin, ymin, xmax, ymax = bbox_clip_xyxy(
                bbox_xywh_to_xyxy(obj['bbox']), width, height)
            # require non-zero box area
            if xmax <= xmin or ymax <= ymin:
                continue
            if obj['num_keypoints'] == 0:
                continue
            # joints 3d: (num_joints, 3, 2); 3 is for x, y, z; 2 is for position, visibility
            joints_3d = np.zeros((self.num_joints, 3, 2), dtype=np.float32)
            for i in range(self.num_joints):
                joints_3d[i, 0, 0] = obj['keypoints'][i * 3 + 0]
                joints_3d[i, 1, 0] = obj['keypoints'][i * 3 + 1]
                # joints_3d[i, 2, 0] = 0
                visible = min(1, obj['keypoints'][i * 3 + 2])
                joints_3d[i, :2, 1] = visible
                # joints_3d[i, 2, 1] = 0

            if np.sum(joints_3d[:, 0, 1]) < 1:
                # no visible keypoint
                continue

            if self._check_centers and self._train:
                bbox_center, bbox_area = self._get_box_center_area(
                    (xmin, ymin, xmax, ymax))
                kp_center, num_vis = self._get_keypoints_center_count(
                    joints_3d)
                ks = np.exp(-2 * np.sum(np.square(bbox_center -
                                                  kp_center)) / bbox_area)
                if (num_vis / 80.0 + 47 / 80.0) > ks:
                    continue

            joint_img = np.zeros((self.num_joints, 3))
            joint_vis = np.ones((self.num_joints, 3))
            joint_img[:, 0] = joints_3d[:, 0, 0]
            joint_img[:, 1] = joints_3d[:, 1, 0]
            joint_vis[:, 2] = 0

            valid_objs.append({
                'bbox': (xmin, ymin, xmax, ymax),
                'width': width,
                'height': height,
                'joint_img': joint_img,
                'joint_vis': joint_vis,
            })

        if not valid_objs:
            if not self._skip_empty:
                # dummy invalid labels if no valid objects are found
                valid_objs.append({
                    'bbox': np.array([-1, -1, 0, 0]),
                    'width': width,
                    'height': height,
                    'joints_3d': np.zeros((self.num_joints, 2, 2), dtype=np.float32)
                })
        return valid_objs

    @property
    def joint_pairs(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return [[0, 5], [1, 4], [2, 3],
                [10, 15], [11, 14], [12, 13]]

    def _get_box_center_area(self, bbox):
        """Get bbox center"""
        c = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
        area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        return c, area

    def _get_keypoints_center_count(self, keypoints):
        """Get geometric center of all keypoints"""
        keypoint_x = np.sum(keypoints[:, 0, 0] * (keypoints[:, 0, 1] > 0))
        keypoint_y = np.sum(keypoints[:, 1, 0] * (keypoints[:, 1, 1] > 0))
        num = float(np.sum(keypoints[:, 0, 1]))
        return np.array([keypoint_x / num, keypoint_y / num]), num
