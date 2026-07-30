import warp as wp
import warp.torch
import torch
import numpy as np


@wp.struct
class MPMModelStruct:
    ####### essential #######
    grid_lim: float
    n_particles: int
    n_grid: int
    dx: float
    inv_dx: float
    grid_dim_x: int
    grid_dim_y: int
    grid_dim_z: int
    object_nums: int
    mu: wp.array(dtype=float)
    lam: wp.array(dtype=float)
    E: wp.array(dtype=float)
    nu: wp.array(dtype=float)
    Ef: wp.array(dtype=float)
    material: wp.array(dtype=int)

    ######## for plasticity ####
    yield_stress: wp.array(dtype=float)

    alpha: float
    gravitational_accelaration: wp.vec3
    hardening: int
    permeability: float
    porosity: float
    diff_coe: wp.array(dtype=float)  
    xi: wp.array(dtype=float)
    plastic_viscosity: float    #0.0
    gamma: float
    # energy: float
    # softening: wp.array(dtype=float)      #0.1
    # sand_s: wp.array(dtype=float, ndim=2)
    # vc_s: wp.array(dtype=float)
    # alpha_s: wp.array(dtype=float)
    # q_s: wp.array(dtype=float)
    
    ########### for snow ###########
    compression: float
    stretch: float
    hardening_coe: float

    ####### for damping
    rpic_damping: wp.array(dtype=float)      # 0.0
    grid_v_damping_scale: float  #1.1

    ####### for PhysGaussian: covariance
    update_cov_with_F: int
    shs_mix: int
    
    


@wp.struct
class MPMStateStruct:
    ###### essential #####
    # particle
    # energy: wp.array(dtype=float)
    particle_x: wp.array(dtype=wp.vec3)  # current position
    particle_v: wp.array(dtype=wp.vec3)  # particle velocity
    particle_F: wp.array(dtype=wp.mat33)  # particle elastic deformation gradient
    particle_init_cov: wp.array(dtype=float)  # initial covariance matrix
    # particle_cov: wp.array(dtype=float)  # current covariance matrix
    # particle_F_trial: wp.array(
    #     dtype=wp.mat33
    # )  # apply return mapping on this to obtain elastic def grad
    # particle_R: wp.array(dtype=wp.mat33)  # rotation matrix
    particle_stress: wp.array(dtype=wp.mat33)  # Kirchoff stress, elastic stress
    particle_C: wp.array(dtype=wp.mat33)
    particle_vol: wp.array(dtype=float)  # current volume
    particle_mass: wp.array(dtype=float)  # mass
    particle_density: wp.array(dtype=float)  # density
    particle_Jp: wp.array(dtype=float)
    particle_Jf: wp.array(dtype=float)
    particle_init_shs: wp.array(dtype=wp.vec3)

    # particle_selection: wp.array(
        # dtype=int
    # )  # only particle_selection[p] = 0 will be simulated

    # grid
    grid_m: wp.array(dtype=float, ndim=4)
    # grid_v_in: wp.array(dtype=wp.vec3, ndim=2)  # grid node momentum/velocity
    grid_v: wp.array(
        dtype=wp.vec3, ndim=4
    )  # grid node momentum/velocity, after grid update
    grid_v_in: wp.array(dtype=wp.vec3, ndim=4)
    grid_vol: wp.array(dtype=float, ndim=4)
    particle_object_link: wp.array(dtype=int)
    # particle_grid_link: wp.array(dtype=int)
    grid_data: wp.array(dtype=wp.vec3, ndim=4)
    grid_shs: wp.array(dtype=wp.vec3, ndim=4)
    # static: wp.array(dtype = int)
    
# @wp.struct
# class MPMTempStruct:
#     particle_cov: wp.array(dtype=float)


# for various boundary conditions
@wp.struct
class Dirichlet_collider:
    point: wp.vec3
    normal: wp.vec3
    direction: wp.vec3

    start_time: float
    end_time: float

    friction: float
    surface_type: int

    velocity: wp.vec3

    threshold: float
    reset: int
    index: int

    x_unit: wp.vec3
    y_unit: wp.vec3
    radius: float
    v_scale: float
    width: float
    height: float
    length: float
    R: float

    size: wp.vec3

    horizontal_axis_1: wp.vec3
    horizontal_axis_2: wp.vec3
    half_height_and_radius: wp.vec2
    
    x_plane: wp.vec2
    y_plane: wp.vec2
    z_plane: wp.vec2
    
@wp.struct
class Static_object:
    object_index: int
    start_time: float
    end_time: float
    velocity: wp.vec3
    reset: int
    static_inf: bool


@wp.struct
class Impulse_modifier:
    # this needs to be changed for each different BC!
    point: wp.vec3
    normal: wp.vec3
    start_time: wp.array(dtype = float)
    end_time: wp.array(dtype = float)
    force: wp.vec3
    forceTimesDt: wp.vec3
    numsteps: int

    point: wp.vec3
    size: wp.vec3
    mask: wp.array(dtype=int)
    interval: float


@wp.struct
class MPMtailoredStruct:
    # this needs to be changed for each different BC!
    point: wp.vec3
    normal: wp.vec3
    start_time: float
    end_time: float
    friction: float
    surface_type: int
    velocity: wp.vec3
    threshold: float
    reset: int

    point_rotate: wp.vec3
    normal_rotate: wp.vec3
    x_unit: wp.vec3
    y_unit: wp.vec3
    radius: float
    v_scale: float
    width: float
    point_plane: wp.vec3
    normal_plane: wp.vec3
    velocity_plane: wp.vec3
    threshold_plane: float


@wp.struct
class MaterialParamsModifier:
    point: wp.vec3
    size: wp.vec3
    E: float
    nu: float
    density: float


@wp.struct
class ParticleVelocityModifier:
    point: wp.vec3
    normal: wp.vec3
    half_height_and_radius: wp.vec2
    rotation_scale: float
    translation_scale: float

    size: wp.vec3

    horizontal_axis_1: wp.vec3
    horizontal_axis_2: wp.vec3

    start_time: float

    end_time: float

    velocity: wp.vec3
    
    object_index: int

    mask: wp.array(dtype=int)


@wp.kernel
def set_vec3_to_zero(target_array: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    target_array[tid] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def set_mat33_to_identity(target_array: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    target_array[tid] = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


@wp.kernel
def add_identity_to_mat33(target_array: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    target_array[tid] = wp.add(
        target_array[tid], wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    )


@wp.kernel
def subtract_identity_to_mat33(target_array: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    target_array[tid] = wp.sub(
        target_array[tid], wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    )


@wp.kernel
def add_vec3_to_vec3(
    first_array: wp.array(dtype=wp.vec3), second_array: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    first_array[tid] = wp.add(first_array[tid], second_array[tid])


@wp.kernel
def set_value_to_float_array(target_array: wp.array(dtype=float), value: float):
    tid = wp.tid()
    target_array[tid] = value

@wp.kernel
def set_value_to_float_subarray(target_array: wp.array(dtype=float, ndim=2), value: float, i: int):
    tid = wp.tid()
    target_array[i, tid] = value

# @wp.kernel
# def set_array_to_float_array_range(target_array: wp.array(dtype=float), array: np.ndarray, n_particle: int):
#     tid = wp.tid()
#     target_array[n_particle + tid] = array[tid]
    
@wp.kernel
def set_value_to_float_array_range(target_array: wp.array(dtype=float), value: float, n_particle: int):
    tid = wp.tid()
    target_array[n_particle + tid] = value
    
@wp.kernel
def set_value_to_float_array_arange(target_array: wp.array(dtype=float), value: float, min_index: int, max_index: int):
    tid = wp.tid()
    if tid >= min_index and tid < max_index:
        target_array[tid] = value

@wp.kernel
def set_value_to_float_subarray_arange(target_array: wp.array(dtype=float, ndim=2), value: float, min_index: int, max_index: int, i: int):
    tid = wp.tid()
    if tid >= min_index and tid < max_index:
        target_array[i, tid] = value

@wp.kernel
def copy_subarray_to_float_array(target_array: wp.array(dtype=float, ndim=2), subarray: wp.array(dtype=float), n_particle: int, i: int):
    tid = wp.tid()
    target_array[i, n_particle + tid] = subarray[tid]

@wp.kernel
def copy_subarray_to_vec3_array(target_array: wp.array(dtype=wp.vec3, ndim=2), subarray: wp.array(dtype=wp.vec3), n_particle: int, i: int):
    tid = wp.tid()
    target_array[i, n_particle + tid] = subarray[tid]
        

@wp.kernel
def get_float_array_product(
    arrayA: wp.array(dtype=float),
    arrayB: wp.array(dtype=float),
    arrayC: wp.array(dtype=float),
):
    tid = wp.tid()
    arrayC[tid] = arrayA[tid] * arrayB[tid]


def torch2warp_quat(t, copy=False, dtype=warp.types.float32, dvc="cuda:0"):
    assert t.is_contiguous()
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    assert t.shape[1] == 4
    a = warp.types.array(
        ptr=t.data_ptr(),
        dtype=wp.quat,
        shape=t.shape[0],
        copy=False,
        owner=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a


def torch2warp_float(t, copy=False, dtype=warp.types.float32, dvc="cuda:0"):
    assert t.is_contiguous()
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    a = warp.types.array(
        ptr=t.data_ptr(),
        dtype=warp.types.float32,
        shape=t.shape[0],
        copy=False,
        owner=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a

def torch2warp_int(t, copy=False, dtype=warp.types.int32, dvc="cuda:0"):
    assert t.is_contiguous()
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    a = warp.types.array(
        ptr=t.data_ptr(),
        dtype=warp.types.int32,
        shape=t.shape[0],
        copy=False,
        owner=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a

def torch2warp_vec3(t, copy=False, dtype=warp.types.float32, dvc="cuda:0"):
    assert t.is_contiguous()
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    assert t.shape[1] == 3
    a = warp.types.array(
        ptr=t.data_ptr(),
        dtype=wp.vec3,
        shape=t.shape[0],
        copy=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a


def torch2warp_mat33(t, copy=False, dtype=warp.types.float32, dvc="cuda:0"):
    assert t.is_contiguous()
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    assert t.shape[1] == 3
    a = warp.types.array(
        ptr=t.data_ptr(),
        dtype=wp.mat33,
        shape=t.shape[0],
        copy=False,
        owner=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a
