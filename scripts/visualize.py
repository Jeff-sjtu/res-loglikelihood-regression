import json
import os
import random
import time

import numpy as np
import torch
import torch.utils.data
from rlepose.models import builder
from rlepose.opt import cfg
import cv2


def to_torch(ndarray):
    # numpy.ndarray => torch.Tensor
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


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


def _box_to_center_scale(x, y, w, h, aspect_ratio=1.0, scale_mult=1.25):
    """Convert box coordinates to center and scale.
    adapted from https://github.com/Microsoft/human-pose-estimation.pytorch
    """
    pixel_std = 1
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    if center[0] != -1:
        scale = scale * scale_mult
    return center, scale


def _center_scale_to_box(center, scale):
    pixel_std = 1.0
    w = scale[0] * pixel_std
    h = scale[1] * pixel_std
    xmin = center[0] - w * 0.5
    ymin = center[1] - h * 0.5
    xmax = xmin + w
    ymax = ymin + h
    bbox = [xmin, ymin, xmax, ymax]
    return bbox


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


def test_transform(src, bbox, input_size=(1080, 1920)):
    aspect_ratio = float(input_size[1]) / input_size[0]  # w / h

    xmin, ymin, xmax, ymax = bbox
    center, scale = _box_to_center_scale(
        xmin, ymin, xmax - xmin, ymax - ymin, aspect_ratio)
    scale = scale * 1.0

    input_size = input_size
    inp_h, inp_w = input_size

    trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
    img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
    bbox = _center_scale_to_box(center, scale)

    img = im_to_torch(img)
    img[0].add_(-0.406)
    img[1].add_(-0.457)
    img[2].add_(-0.480)

    return img, bbox


def preset_model(cfg):
    model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    if cfg.VISUALIZE.CHECKPOINT:
        print(f'Loading model from ... {cfg.VISUALIZE.CHECKPOINT}')
        model.load_state_dict(torch.load(cfg.VISUALIZE.CHECKPOINT))

    return model


def get_adjacent_keypoints(pose_coords, pred_scores, link, min_confidence=0.3):
    adjacent_keypoints = []
    results = []
    for left, right in link:
        if pred_scores[0][left][0] < min_confidence or pred_scores[0][right][0] < min_confidence:
            continue
        results.append(
            np.array([pose_coords[0][left][::],
                      pose_coords[0][right][::]]).astype(np.int32),
        )
    adjacent_keypoints.extend(results)
    return adjacent_keypoints


def get_Keypoints(pose_coords, pose_scores, min_confidence):
    cv_keypoints = []
    for kc in range(0, 17):
        if pose_scores[0][kc][0] < min_confidence:
            continue
        cv_keypoints.append(cv2.KeyPoint(int(pose_coords[0][kc][0]), int(pose_coords[0][kc][1]), 15))
    return cv_keypoints


def transformation(src, bbox):
    bbox1 = list(bbox)
    # print('original bbox', bbox1)

    input_size = [256, 192]
    aspect_ratio = float(input_size[1]) / input_size[0]  # w / h

    xmin, ymin, xmax, ymax = bbox1
    center, scale = _box_to_center_scale(
        xmin, ymin, xmax - xmin, ymax - ymin, aspect_ratio)

    scale = scale * 1.0
    r = 0

    inp_h, inp_w = input_size
    trans = get_affine_transform(center, scale, r, [inp_w, inp_h])
    img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)

    bbox = _center_scale_to_box(center, scale)

    img = im_to_torch(img)
    img[0].add_(-0.406)
    img[1].add_(-0.457)
    img[2].add_(-0.480)

    return img, torch.Tensor(bbox)


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def heatmap_to_coord_2D(pred_jts, pred_scores, hm_shape, bbox):
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

    return preds, pred_scores


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    target_coords[0:2] = affine_transform(coords[0:2], trans)
    return target_coords


if __name__ == "__main__":
    link = [(0, 1), (1, 3), (0, 2), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10), (11, 13),
            (13, 15), (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12)]

    video_path = cfg.VISUALIZE.VIDEO_PATH
    min_confidence = cfg.VISUALIZE.CONFIDENCE_THRESHOLD
    use_gpu = cfg.VISUALIZE.USE_GPU

    model = preset_model(cfg)
    if use_gpu:
        model = model.cuda()
    model.eval()

    video_capture = cv2.VideoCapture(video_path)
    start_time = time.time()
    frame_count = 0

    # flag of Record inference video
    if cfg.VISUALIZE.FLAG_WRITE_VIDEO:
        flag_write_video = True
        print('flag_write_video', flag_write_video)
        flag_video_start = False
        video_writer = None

    while True:
        ret, image = video_capture.read()
        if ret:
            if flag_video_start is False and flag_write_video:
                loc_time = time.localtime()
                str_time = time.strftime("%Y-%m-%d_%H-%M-%S", loc_time)
                video_writer = cv2.VideoWriter("./demo/demo_{}.mp4".format(str_time), cv2.VideoWriter_fourcc(*"mp4v"),
                                               fps=24, frameSize=(int(image.shape[1]), int(image.shape[0])))
                flag_video_start = True
            # print(im0.shape)
            # image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

            bbox = [0, 0, cfg.VISUALIZE.VIDEO_WIDTH, cfg.VISUALIZE.VIDEO_HEIGHT]
            input_image, bbox = transformation(image, bbox)
            input_image = input_image.unsqueeze(0)
            if use_gpu:
                input_image = input_image.cuda()

            with torch.no_grad():
                output = model(input_image)

            # get pose_coords, pose_scores
            pred_jts = output.pred_jts[0]
            pred_scores = output.maxvals[0]
            pose_coords, pose_scores = heatmap_to_coord_2D(pred_jts, pred_scores, cfg.DATA_PRESET.HEATMAP_SIZE, bbox)

            # get drawing keypoints and lines
            cv_keypoints = get_Keypoints(pose_coords, pose_scores, min_confidence)
            adjacent_keypoints = get_adjacent_keypoints(pose_coords, pose_scores, link, min_confidence)
            cv2.drawKeypoints(
                image, cv_keypoints, image, color=(0, 0, 255),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.polylines(image, adjacent_keypoints, isClosed=False, color=(255, 255, 0), thickness=5)

            # display
            cv2.namedWindow("movenet", 0)
            cv2.resizeWindow("movenet", 1280, 720)
            cv2.imshow('movenet', image)

            # save video
            if flag_write_video and flag_video_start:
                video_writer.write(image)

            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    print('Average FPS:', frame_count / (time.time() - start_time))
    cv2.destroyAllWindows()