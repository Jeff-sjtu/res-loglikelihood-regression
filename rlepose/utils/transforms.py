import math

import cv2
import numpy as np
import torch


def rnd(x):
    return max(-2 * x, min(2 * x, np.random.randn(1)[0] * x))


def box_transform(bbox, sf, imgwidth, imght, train):
    """Random scaling."""
    width = bbox[2] - bbox[0]
    ht = bbox[3] - bbox[1]
    if train:
        scaleRate = 0.25 * np.clip(np.random.randn() * sf, - sf, sf)

        bbox[0] = max(0, bbox[0] - width * scaleRate / 2)
        bbox[1] = max(0, bbox[1] - ht * scaleRate / 2)
        bbox[2] = min(imgwidth, bbox[2] + width * scaleRate / 2)
        bbox[3] = min(imght, bbox[3] + ht * scaleRate / 2)
    else:
        scaleRate = 0.25

        bbox[0] = max(0, bbox[0] - width * scaleRate / 2)
        bbox[1] = max(0, bbox[1] - ht * scaleRate / 2)
        bbox[2] = min(imgwidth, max(bbox[2] + width * scaleRate / 2, bbox[0] + 5))
        bbox[3] = min(imght, max(bbox[3] + ht * scaleRate / 2, bbox[1] + 5))

    return bbox


def im_to_torch(img):
    """Transform ndarray image to torch tensor.

    Parameters
    ----------
    img: numpy.ndarray
        An ndarray with shape: `(H, W, 3)`.

    Returns
    -------
    torch.Tensor
        A tensor with shape: `(3, H, W)`.

    """
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img


def torch_to_im(img):
    """Transform torch tensor to ndarray image.

    Parameters
    ----------
    img: torch.Tensor
        A tensor with shape: `(3, H, W)`.

    Returns
    -------
    numpy.ndarray
        An ndarray with shape: `(H, W, 3)`.

    """
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0))  # C*H*W
    return img


def load_image(img_path):
    # H x W x C => C x H x W
    return im_to_torch(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))  # scipy.misc.imread(img_path, mode='RGB'))


def to_numpy(tensor):
    # torch.Tensor => numpy.ndarray
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    # numpy.ndarray => torch.Tensor
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def get_3rd_point(a, b):
    """Return vector c that perpendicular to (a - b)."""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    """Rotate the point by `rot_rad` degree."""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def count_visible(bbox, joints_3d):
    """Count number of visible joints given bound box."""
    vis = np.logical_and.reduce((
        joints_3d[:, 0, 0] > 0,
        joints_3d[:, 0, 0] > bbox[0],
        joints_3d[:, 0, 0] < bbox[2],
        joints_3d[:, 1, 0] > 0,
        joints_3d[:, 1, 0] > bbox[1],
        joints_3d[:, 1, 0] < bbox[3],
        joints_3d[:, 0, 1] > 0,
        joints_3d[:, 1, 1] > 0
    ))
    return np.sum(vis), vis


def flip(x):
    assert (x.dim() == 3 or x.dim() == 4)
    dim = x.dim() - 1

    return x.flip(dims=(dim,))


def flip_output(output, joint_pairs, width_dim, shift=False):
    if 'heatmap' in output.keys():
        flipped_hm = flip_heatmap(output.heatmap, joint_pairs, shift)
        output.heatmap = flipped_hm
    if 'pred_jts' in output.keys():
        flipped_jts, flipped_maxvals = flip_coord((output.pred_jts, output.maxvals), joint_pairs, width_dim, shift, flatten=False)
        output.pred_jts = flipped_jts
        output.maxvals = flipped_maxvals
    return output


def flip_heatmap(heatmap, joint_pairs, shift=False):
    """Flip pose heatmap according to joint pairs.

    Parameters
    ----------
    heatmap : numpy.ndarray
        Heatmap of joints.
    joint_pairs : list
        List of joint pairs.
    shift : bool
        Whether to shift the output.

    Returns
    -------
    numpy.ndarray
        Flipped heatmap.

    """
    assert (heatmap.dim() == 3 or heatmap.dim() == 4)
    out = flip(heatmap)

    for pair in joint_pairs:
        dim0, dim1 = pair
        idx = torch.Tensor((dim0, dim1)).long()
        inv_idx = torch.Tensor((dim1, dim0)).long()
        if out.dim() == 4:
            out[:, idx] = out[:, inv_idx]
        else:
            out[idx] = out[inv_idx]

    if shift:
        if out.dim() == 3:
            out[:, :, 1:] = out[:, :, 0:-1]
        else:
            out[:, :, :, 1:] = out[:, :, :, 0:-1]
    return out


def flip_joints_3d(joints_3d, width, joint_pairs):
    """Flip 3d joints.

    Parameters
    ----------
    joints_3d : numpy.ndarray
        Joints in shape (num_joints, 3, 2)
    width : int
        Image width.
    joint_pairs : list
        List of joint pairs.

    Returns
    -------
    numpy.ndarray
        Flipped 3d joints with shape (num_joints, 3, 2)

    """
    joints = joints_3d.copy()
    # flip horizontally
    joints[:, 0, 0] = width - joints[:, 0, 0] - 1
    # change left-right parts
    for pair in joint_pairs:
        joints[pair[0], :, 0], joints[pair[1], :, 0] = \
            joints[pair[1], :, 0], joints[pair[0], :, 0].copy()
        joints[pair[0], :, 1], joints[pair[1], :, 1] = \
            joints[pair[1], :, 1], joints[pair[0], :, 1].copy()

    joints[:, :, 0] *= joints[:, :, 1]
    return joints


def flip_coord(preds, joint_pairs, width_dim, shift=False, flatten=True):
    """Flip pose heatmap according to joint pairs.

    Parameters
    ----------
    heatmap : numpy.ndarray
        Heatmap of joints.
    joint_pairs : list
        List of joint pairs.
    shift : bool
        Whether to shift the output.

    Returns
    -------
    numpy.ndarray
        Flipped heatmap.

    """
    pred_jts, pred_scores = preds
    if flatten:
        assert pred_jts.dim() == 2 and pred_scores.dim() == 3
        num_batches = pred_jts.shape[0]
        num_joints = pred_jts.shape[1] // 3
        pred_jts = pred_jts.reshape(num_batches, num_joints, 3)
    else:
        assert pred_jts.dim() == 3 and pred_scores.dim() == 3
        num_batches = pred_jts.shape[0]
        num_joints = pred_jts.shape[1]

    # flip
    if shift:
        pred_jts[:, :, 0] = - pred_jts[:, :, 0] - 1 / (width_dim * 4)
    else:
        pred_jts[:, :, 0] = -1 / width_dim - pred_jts[:, :, 0]

    for pair in joint_pairs:
        dim0, dim1 = pair
        idx = torch.Tensor((dim0, dim1)).long()
        inv_idx = torch.Tensor((dim1, dim0)).long()
        pred_jts[:, idx] = pred_jts[:, inv_idx]
        pred_scores[:, idx] = pred_scores[:, inv_idx]

    return pred_jts, pred_scores


def heatmap_to_coord_simple(hms, bbox, **kwargs):
    if not isinstance(hms, np.ndarray):
        hms = hms.cpu().data.numpy()
    coords, maxvals = get_max_pred(hms)

    hm_h = hms.shape[1]
    hm_w = hms.shape[2]

    # post-processing
    for p in range(coords.shape[0]):
        hm = hms[p]
        px = int(round(float(coords[p][0])))
        py = int(round(float(coords[p][1])))
        if 1 < px < hm_w - 1 and 1 < py < hm_h - 1:
            diff = np.array((hm[py][px + 1] - hm[py][px - 1],
                             hm[py + 1][px] - hm[py - 1][px]))
            coords[p] += np.sign(diff) * .25

    preds = np.zeros_like(coords)

    # transform bbox to scale
    xmin, ymin, xmax, ymax = bbox
    w = xmax - xmin
    h = ymax - ymin
    center = np.array([xmin + w * 0.5, ymin + h * 0.5])
    scale = np.array([w, h])
    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center, scale,
                                   [hm_w, hm_h])

    return preds[None, :, :], maxvals[None, :, :]


def heatmap_to_coord(pred_jts, pred_scores, hm_shape, bbox, output_3d=False):
    hm_height, hm_width = hm_shape
    hm_height = hm_height * 4
    hm_width = hm_width * 4

    ndims = pred_jts.dim()
    assert ndims in [2, 3], "Dimensions of input heatmap should be 2 or 3"
    if ndims == 2:
        pred_jts = pred_jts.unsqueeze(0)
        pred_scores = pred_scores.unsqueeze(0)

    coords = pred_jts.cpu().numpy()
    coords = coords.astype(float)
    pred_scores = pred_scores.cpu().numpy()
    pred_scores = pred_scores.astype(float)

    coords[:, :, 0] = (coords[:, :, 0] + 0.5) * hm_width
    coords[:, :, 1] = (coords[:, :, 1] + 0.5) * hm_height

    preds = np.zeros_like(coords)
    # transform bbox to scale
    xmin, ymin, xmax, ymax = bbox
    w = xmax - xmin
    h = ymax - ymin
    center = np.array([xmin + w * 0.5, ymin + h * 0.5])
    scale = np.array([w, h])
    # Transform back
    for i in range(coords.shape[0]):
        for j in range(coords.shape[1]):
            preds[i, j, 0:2] = transform_preds(coords[i, j, 0:2], center, scale,
                                               [hm_width, hm_height])
            if output_3d:
                preds[i, j, 2] = coords[i, j, 2]

    return preds, pred_scores


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    target_coords[0:2] = affine_transform(coords[0:2], trans)
    return target_coords


def get_max_pred(heatmaps):
    num_joints = heatmaps.shape[0]
    width = heatmaps.shape[2]
    heatmaps_reshaped = heatmaps.reshape((num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 1)
    maxvals = np.max(heatmaps_reshaped, 1)

    maxvals = maxvals.reshape((num_joints, 1))
    idx = idx.reshape((num_joints, 1))

    preds = np.tile(idx, (1, 2)).astype(np.float32)

    preds[:, 0] = (preds[:, 0]) % width
    preds[:, 1] = np.floor((preds[:, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_max_pred_batch(batch_heatmaps):
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.max(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0,
                         align=False):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_warpmatrix(theta, size_input, size_dst, size_target, pixel_std):
    size_target = size_target * pixel_std
    theta = theta / 180.0 * np.pi

    matrix = np.zeros((2, 3), dtype=np.float32)

    scale_x = size_target[0] / size_dst[0]
    scale_y = size_target[1] / size_dst[1]
    matrix[0, 0] = math.cos(theta) * scale_x
    matrix[0, 1] = math.sin(theta) * scale_y
    matrix[0, 2] = -0.5 * size_target[0] * math.cos(theta) - 0.5 * size_target[1] * math.sin(theta) + 0.5 * size_input[0]
    matrix[1, 0] = -math.sin(theta) * scale_x
    matrix[1, 1] = math.cos(theta) * scale_y
    matrix[1, 2] = 0.5 * size_target[0] * math.sin(theta) - 0.5 * size_target[1] * math.cos(theta) + 0.5 * size_input[1]

    return matrix


def get_warpmatrix_inverse(theta, size_input, size_dst, size_target):
    """
    :param theta: angle x y
    :param size_input:[w,h]
    :param size_dst: [w,h] i
    :param size_target: [w,h] b
    :return:
    """
    size_target = size_target * 200.0
    theta = theta / 180.0 * math.pi
    matrix = np.zeros((2, 3), dtype=np.float32)
    scale_x = size_dst[0] / size_target[0]
    scale_y = size_dst[1] / size_target[1]
    matrix[0, 0] = math.cos(theta) * scale_x
    matrix[0, 1] = -math.sin(theta) * scale_x
    matrix[0, 2] = scale_x * (-0.5 * size_input[0] * math.cos(theta) + 0.5 * size_input[1] * math.sin(theta) + 0.5 * size_target[0])
    matrix[1, 0] = math.sin(theta) * scale_y
    matrix[1, 1] = math.cos(theta) * scale_y
    matrix[1, 2] = scale_y * (-0.5 * size_input[0] * math.sin(theta) - 0.5 * size_input[1] * math.cos(theta) + 0.5 * size_target[1])
    return matrix


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_func_heatmap_to_coord(cfg):
    if cfg.TEST.get('HEATMAP2COORD') == 'coord':
        return heatmap_to_coord
    elif cfg.TEST.get('HEATMAP2COORD') == 'heatmap':
        return heatmap_to_coord_simple
    else:
        raise NotImplementedError


class get_coord(object):
    def __init__(self, cfg, norm_size, output_3d=False):
        self.type = cfg.TEST.get('HEATMAP2COORD')
        self.input_size = cfg.DATA_PRESET.IMAGE_SIZE
        self.norm_size = norm_size
        self.output_3d = output_3d

    def __call__(self, output, bbox, idx):
        if self.type == 'coord':
            pred_jts = output.pred_jts[idx]
            pred_scores = output.maxvals[idx]
            return heatmap_to_coord(pred_jts, pred_scores, self.norm_size, bbox, self.output_3d)
        elif self.type == 'heatmap':
            pred_hms = output.heatmap[idx]
            return heatmap_to_coord_simple(pred_hms, bbox)
        else:
            raise NotImplementedError
