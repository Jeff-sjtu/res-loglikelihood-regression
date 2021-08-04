import cv2
import matplotlib.pyplot as plt
import numpy as np


def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1]), (S1.shape, S2.shape)

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale * (R.dot(mu1))

    # 7. Error:
    S1_hat = scale * R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    if S1.ndim == 2:
        S1_hat = compute_similarity_transform(S1.copy(), S2.copy())
    else:
        S1_hat = np.zeros_like(S1)
        for i in range(S1.shape[0]):
            S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat


def reconstruction_error(S1, S2):
    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform_batch(S1, S2)
    return S1_hat


def world2cam(world_coord, R, T):
    cam_coord = np.dot(R, world_coord - T)
    return cam_coord


def weak_cam2pixel(cam_coord, root_z, f, c):
    x = cam_coord[:, 0] / (root_z + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (root_z + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)

    avg_f = (f[0] + f[1]) / 2
    cam_param = np.array([avg_f / (root_z + 1e-8), c[0], c[1]])
    return img_coord, cam_param


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
    return img_coord


def cam2pixel_matrix(cam_coord, intrinsic_param):
    cam_coord = cam_coord.transpose(1, 0)
    cam_homogeneous_coord = np.concatenate((cam_coord, np.ones((1, cam_coord.shape[1]), dtype=np.float32)), axis=0)
    img_coord = np.dot(intrinsic_param, cam_homogeneous_coord) / (cam_coord[2, :] + 1e-8)
    img_coord = np.concatenate((img_coord[:2, :], cam_coord[2:3, :]), axis=0)
    return img_coord.transpose(1, 0)


def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
    return cam_coord


def pixel2cam_matrix(pixel_coord, intrinsic_param):

    x = (pixel_coord[:, 0] - intrinsic_param[0][2]) / intrinsic_param[0][0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - intrinsic_param[1][2]) / intrinsic_param[1][1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
    return cam_coord


def rigid_transform_3D(A, B):
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B)
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))
    t = -np.dot(R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return R, t


def rigid_align(A, B):
    R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(R, np.transpose(A))) + t
    return A2


def get_intrinsic_metrix(f, c, inv=False):
    intrinsic_metrix = np.zeros((3, 3)).astype(np.float32)
    intrinsic_metrix[0, 0] = f[0]
    intrinsic_metrix[0, 2] = c[0]
    intrinsic_metrix[1, 1] = f[1]
    intrinsic_metrix[1, 2] = c[1]
    intrinsic_metrix[2, 2] = 1

    if inv:
        intrinsic_metrix = np.linalg.inv(intrinsic_metrix).astype(np.float32)
    return intrinsic_metrix


def vis_keypoints(img, kps, kps_lines, kp_thresh=0.4, alpha=1):

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for link in range(len(kps_lines)):
        i1 = kps_lines[link][0]
        i2 = kps_lines[link][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[link], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[link], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[link], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)
