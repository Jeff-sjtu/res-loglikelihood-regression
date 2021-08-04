import random

import torch
import torch.utils.data as data
from rlepose.models.builder import DATASET
from easydict import EasyDict

from .h36m import H36m
from .mpii import Mpii

s_mpii_2_hm36_jt = [6, 2, 1, 0, 3, 4, 5, -1, 8,
                    -1, 9, 13, 14, 15, 12, 11, 10, 7]
s_36_jt_num = 18


@DATASET.register_module
class H36mMpii(data.Dataset):
    CLASSES = ['person']
    EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    num_joints = 18
    num_bones = 17
    bbox_3d_shape = (2000, 2000, 2000)
    joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck',
                   'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'Thorax')
    action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases',
                   'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']
    skeleton = ((1, 0), (2, 0),
                (3, 1), (4, 2), (5, 3), (6, 4),
                (7, 0),
                (8, 0), (9, 0),
                (10, 8), (11, 9), (12, 10), (13, 11)
                )
    data_domain = set([
        'type',
        'target_uvd',
        'target_uvd_weight'
    ])

    def __init__(self,
                 train=True,
                 skip_empty=True,
                 dpg=False,
                 lazy_import=False,
                 **cfg):
        self._train = train
        self._preset_cfg = cfg['PRESET']

        cfg = EasyDict(cfg)
        if train:
            self.db0 = H36m(
                cfg=cfg,
                ann_file=cfg.SET_LIST[0].TRAIN_SET,
                train=train)
            self.db1 = Mpii(
                cfg=cfg,
                ann_file=f'{cfg.SET_LIST[1].TRAIN_SET}.json',
                train=True)

            self._subsets = [self.db0, self.db1]
        else:
            self.db0 = H36m(
                cfg=cfg,
                ann_file=cfg.TEST_SET,
                train=train)

            self._subsets = [self.db0]

        self._subset_size = [len(item) for item in self._subsets]
        self.cumulative_sizes = self.cumsum(self._subset_size)
        self._db0_size = len(self.db0)

        if train:
            self._db1_size = len(self.db1)
            self.tot_size = 2 * self._db0_size
        else:
            self.tot_size = self._db0_size
        self.joint_pairs = self.db0.joint_pairs
        self.evaluate = self.db0.evaluate

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            r.append(e + s)
            s += e
        return r

    def __len__(self):
        return self.tot_size

    def __getitem__(self, idx):
        assert idx >= 0
        if idx < self._db0_size:
            dataset_idx = 0
            sample_idx = idx
        else:
            assert self._train
            dataset_idx = 1

            _rand = random.uniform(0, 1)
            sample_idx = int(_rand * self._db1_size)

        sample = self._subsets[dataset_idx][sample_idx]
        img, target, img_id, bbox = sample

        if dataset_idx == 1:
            # Mpii
            target_uvd_origin = target.pop('target_uvd')
            target_uvd_weight_origin = target.pop('target_uvd_weight')

            target_uvd = torch.zeros(self.num_joints, 3)
            target_uvd_weight = torch.zeros(self.num_joints, 3)

            assert target_uvd_origin.dim() == 1 and target_uvd_origin.shape[0] == 16 * 3, target_uvd_origin.shape
            target_uvd_origin = target_uvd_origin.reshape(16, 3)
            target_uvd_weight_origin = target_uvd_weight_origin.reshape(16, 3)
            for i in range(s_36_jt_num):
                id1 = i
                id2 = s_mpii_2_hm36_jt[i]
                if id2 >= 0:
                    target_uvd[id1, :2] = target_uvd_origin[id2, :2].clone()
                    target_uvd_weight[id1, :2] = target_uvd_weight_origin[id2, :2].clone()

            target['target_uvd'] = target_uvd.reshape(-1)
            target['target_uvd_weight'] = target_uvd_weight.reshape(-1)

        assert set(target.keys()).issubset(self.data_domain), (set(target.keys()), self.data_domain)
        target.pop('type')

        return img, target, img_id, bbox
