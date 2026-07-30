import torch
from time import time
import torch.nn.functional as F
import taichi as ti
import ipdb

@torch.no_grad()
def downsample_with_kmeans_gpu(points_array: torch.Tensor, num_points: int):

    # from kmeans_gpu import KMeans
    from kmeans_pytorch import kmeans

    # kmeans = KMeans(
    #     n_clusters=num_points,
    #     max_iter=100,
    #     tolerance=1e-4,
    #     distance="euclidean",
    #     sub_sampling=None,
    #     max_neighbors=15,
    # )
    # breakpoint()
    features = torch.ones(1, 1, points_array.shape[0], device=points_array.device)
    # points_array = points_array.unsqueeze(0)
    # Forward

    print(
        "=> staring downsample with kmeans from ",
        points_array.shape[0],
        " points to ",
        num_points,
        " points",
    )
    s_time = time()
    # centroids, features = kmeans(points_array, features)
    cluster_ids_x, centroids, iters = kmeans(
        X=points_array, num_clusters=num_points, distance='euclidean', device=points_array.device
        )

    ret_points = centroids.squeeze(0)
    e_time = time()
    print("=> downsample with kmeans takes ", e_time - s_time, " seconds")

    # [np_subsample, 3]
    return ret_points


def get_rigid_transform(A, B):
    """
    Estimate the rigid body transformation between two sets of 3D points.
    A and B are Nx3 matrices where each row is a 3D point.
    Returns a rotation matrix R and translation vector t.
    Args:
        A, B: [batch, N, 3] matrix of 3D points
    Outputs:
        R, t: [batch, 3, 3/1]
        target = R @ source (source shape [3, 1]) + t
    """
    assert A.shape == B.shape, "Input matrices must have the same shape"
    assert A.shape[-1] == 3, "Input matrices must have 3 columns (x, y, z coordinates)"

    # Compute centroids. [..., 1, 3]
    centroid_A = torch.mean(A, dim=-2, keepdim=True)
    centroid_B = torch.mean(B, dim=-2, keepdim=True)

    # Center the point sets
    A_centered = A - centroid_A
    B_centered = B - centroid_B

    # Compute the cross-covariance matrix. [..., 3, 3]
    H = A_centered.transpose(-2, -1) @ B_centered

    # Compute the Singular Value Decomposition. Along last two dimensions
    U, S, Vt = torch.linalg.svd(H)

    # Compute the rotation matrix
    R = Vt.transpose(-2, -1) @ U.transpose(-2, -1)

    # Ensure a right-handed coordinate system
    flip_mask = (torch.det(R) < 0) * -2.0 + 1.0
    # Vt[:, 2, :] *= flip_mask[..., None]

    # [N] => [N, 3]
    pad_flip_mask = torch.stack(
        [torch.ones_like(flip_mask), torch.ones_like(flip_mask), flip_mask], dim=-1
    )
    Vt = Vt * pad_flip_mask[..., None]

    # Compute the rotation matrix
    R = Vt.transpose(-2, -1) @ U.transpose(-2, -1)

    # print(R.shape, centroid_A.shape, centroid_B.shape, flip_mask.shape)
    # Compute the translation
    t = centroid_B - (R @ centroid_A.transpose(-2, -1)).transpose(-2, -1)
    t = t.transpose(-2, -1)
    return R, t


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    From pytorch3d
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    ret = torch.stack((ow, ox, oy, oz), -1)
    ret = standardize_quaternion(ret)
    return ret

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    from Pytorch3d
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    from pytorch3d. Based on trace_method like: https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L205
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def select_particle_x_inside(tensor_x, boundary):
    ti_x = ti.Vector.field(n=3, dtype=float, shape=tensor_x.shape[0])
    ti_x.from_torch(tensor_x.reshape(-1, 3))
    index_p = ti.field(int, shape=tensor_x.shape[0])
    index_p.fill(0)
    # print('boundary', boundary)
    select_particle_x_inside_kernel(
        ti_x,
        index_p,
        boundary[0],
        boundary[1],
        boundary[2],
        boundary[3],
        boundary[4],
        boundary[5]
    )
    index_p_np = index_p.to_numpy()
    index_p_np = index_p_np.astype('bool')
    # breakpoint()
    return index_p_np.tolist()


@ti.kernel
def select_particle_x_inside_kernel(
    ti_x: ti.template(),
    index_p: ti.template(),
    boundary_xmin: float,
    boundary_xmax: float,
    boundary_ymin: float,
    boundary_ymax: float,
    boundary_zmin: float,
    boundary_zmax: float
):
    # print(boundary_zmin, boundary_zmax)
    for pi in range(ti_x.shape[0]):
        px = ti_x[pi]
        if (boundary_xmin - px[0]) < 1e-15 and (px[0] - boundary_xmax) < 1e-15 and (boundary_ymin - px[1]) < 1e-15 and (px[1] - boundary_ymax) < 1e-15 and (boundary_zmin - px[2]) < 1e-15 and (px[2] - boundary_zmax) < 1e-15:
            index_p[pi] = 1
