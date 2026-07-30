import sys

sys.path.append("gaussian-splatting")

import argparse
import math
import cv2
import torch
import os
import numpy as np
import json
from tqdm import tqdm

# Gaussian splatting dependencies
from utils.sh_utils import eval_sh
from scene.gaussian_model import GaussianModel
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from scene.cameras import Camera as GSCamera
from gaussian_renderer import render, GaussianModel
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import focal2fov
from utils.sh_utils import *

# MPM dependencies
from mpm_solver_warp.engine_utils import *
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
import warp as wp

# Particle filling dependencies
from particle_filling.filling import *

# Utils
from utils.decode_param import *
from utils.transformation_utils import *
from utils.camera_view_utils import *
from utils.render_utils import *
from utils.local_utils import *
import shutil
import ipdb
import time
import gc


ti.init(arch=ti.cuda, device_memory_GB=5.0, random_seed = 3407)
wp.init()
# wp.config.verify_cuda = True
# wp.set_mempool_release_threshold("cuda:0", 1024**3)
# wp.set_peer_access_enabled("cuda:0", "cuda:1", True)
# wp.set_mempool_access_enabled("cuda:0", "cuda:1", True)


class PipelineParamsNoparse:
    """Same as PipelineParams but without argument parser."""

    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def load_checkpoint(model_path, sh_degree=0, iteration=-1, is_ply=False):
    # Find checkpoint
    if is_ply:
        checkpt_path = model_path
    else:
        checkpt_dir = os.path.join(model_path, "point_cloud")
        if iteration == -1:
            iteration = searchForMaxIteration(checkpt_dir)
        checkpt_path = os.path.join(
            checkpt_dir, f"iteration_{iteration}", "point_cloud.ply"
        )

    # Load guassians
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path)
    return gaussians


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_ply", action="store_true")
    parser.add_argument("--output_h5", action="store_true")
    parser.add_argument("--render_img", action="store_true")
    parser.add_argument("--compile_video", action="store_true")
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--shs_mix", action="store_true")
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--only_water", action="store_true")
    parser.add_argument("--no_water", action="store_true")
    parser.add_argument("--background_img", type=str, default = "")
    args = parser.parse_args()

    # if not os.path.exists(args.model_path):
    #     AssertionError("Model path does not exist!")
    if not os.path.exists(args.config):
        AssertionError("Scene config does not exist!")
    if args.output_path is not None and not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    shutil.copyfile(args.config, os.path.join(args.output_path, 'config.json'))

    # load scene config
    print("Loading scene config...")
    (
        grid_params,
        materials_params,
        bc_params,
        time_params,
        preprocessings_params,
        camera_params,
        add_particle_params,
        modify_material_params
    ) = decode_param_json(args.config)

    device = "cuda"
    mpm_init_pos = torch.empty(0, device=device)
    gs_nums = []
    fill_gs_nums = [0]
    unselected_gs_nums = [0]
    mpm_init_shs = torch.empty(0, device = device)
    mpm_init_cov = torch.empty(0, device = device)
    mpm_init_opacity = torch.empty(0, device = device)
    mpm_init_vol = torch.empty(0, device = device)
    # mpm_init_material = torch.empty(0)
    scales = torch.empty(0, device = device)
    original_mean_poss = torch.empty(0, device = device)
    center_points = torch.empty(0, device = device)
    rotation_matricess = []
    init_screen_points_scene = torch.empty(0, device = device)
    unselected_pos_scene = torch.empty(0, device = device)
    unselected_opacity_scene = torch.empty(0, device = device)
    unselected_cov_scene = torch.empty(0, device = device)
    unselected_shs_scene = torch.empty(0, device = device)
    mpm_particle_material = torch.empty(0, device = device)    # 粒子与材料的匹配
    # grid_center = torch.tensor([grid_params["grid_lim"]/2, grid_params["grid_lim"]/2, grid_params["grid_lim"]/2]).to(device=device)
    grid_center = torch.tensor(camera_params["mpm_space_viewpoint_center"]).to(device=device)
    object_material_index = []
    is_ply = False
    exist_floor = False

    for mi, material_params in enumerate(materials_params):
        preprocessing_params = preprocessings_params[mi]
        # load gaussians
        print(f"Loading gaussians{mi}...")
        model_path = preprocessing_params["model_path"]
        if not os.path.exists(model_path):
            print(model_path)
            AssertionError("Model path does not exist!")
        is_ply = (model_path.find('.ply')!=-1)
        gaussians = load_checkpoint(model_path, sh_degree=args.sh_degree, is_ply=is_ply)
        pipeline = PipelineParamsNoparse()
        pipeline.compute_cov3D_python = True
        background = (
            torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
            if args.white_bg
            else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        )

        # init the scene
        print(f"Initializing object{mi} and pre-processing...")
        params = load_params_from_gs(gaussians, pipeline)
        # breakpoint()

        init_pos = params["pos"]
        init_cov = params["cov3D_precomp"]
        # init_screen_points = params["screen_points"]
        init_opacity = params["opacity"]
        init_shs = params["shs"]
        init_rot = params["rotations"]
        # breakpoint()

        # throw away low opacity kernels
        mask = init_opacity[:, 0] > preprocessing_params["opacity_threshold"]
        init_pos = init_pos[mask, :]
        init_cov = init_cov[mask, :]
        init_opacity = init_opacity[mask, :]
        # init_screen_points = init_screen_points[mask, :]
        init_shs = init_shs[mask, :]
        # breakpoint()
        # breakpoint()
        # init_opacity_1 = init_opacity
        # if preprocessing_params["opacity_scale"] < 1.0 and material_params["material"] == "fluid":
        #     init_opacity = init_opacity * preprocessing_params["opacity_scale"]
        #     init_opacity = torch.mean(init_opacity).repeat(init_opacity.shape)
        # init_opacity = torch.mean(init_opacity).repeat(init_opacity.shape)
        # rorate and translate object
        if args.debug:
            if not os.path.exists("./log"):
                os.makedirs("./log")
            particle_position_tensor_to_ply(
                init_pos,
                f"./log/init_particles_{mi}.ply",
            )
        rotation_matrices = generate_rotation_matrices(
            torch.tensor(preprocessing_params["rotation_degree"]),
            preprocessing_params["rotation_axis"],
        )
        rotated_pos = apply_rotations(init_pos, rotation_matrices)

        if args.debug:
            particle_position_tensor_to_ply(rotated_pos, f"./log/rotated_particles_{mi}.ply")

        # select a sim area and save params of unslected particles
        unselected_pos, unselected_cov, unselected_opacity, unselected_shs = (
            None,
            None,
            None,
            None,
        )
        # import ipdb; ipdb.set_trace()
        # breakpoint()

        transformed_pos, scale_origin, original_mean_pos = transform2origin(rotated_pos)
        if preprocessing_params["model_path"] == "model/cat_cup.ply":
            transformed_pos[:, 1] = -transformed_pos[:, 1]
        if material_params["type"] == "floor":
            transformed_pos[:, 0] = transformed_pos[:, 0] * preprocessing_params['scale']
            transformed_pos[:, 1] = transformed_pos[:, 1] * preprocessing_params['scale']
        elif material_params["type"] == "table":
            if preprocessing_params['scale'] < 5.0:
                transformed_pos = transformed_pos * preprocessing_params['scale']
            else:
                transformed_pos = transformed_pos * 5.0
                transformed_pos[:, 0] = transformed_pos[:, 0] * (preprocessing_params['scale'] / 5.0)
                transformed_pos[:, 1] = transformed_pos[:, 1] * (preprocessing_params['scale'] / 5.0)
        else:
            transformed_pos = transformed_pos * preprocessing_params['scale']
        center_point = torch.tensor(preprocessing_params["center_point"]).to(device=device)
        transformed_pos, original_mean_pos = transform2center(transformed_pos, original_mean_pos, scale_origin, center_point)

        # init_cov = apply_cov_rotations(init_cov, rotation_matrices)
        init_cov = scale_origin * scale_origin * init_cov
        # init_cov[:, 0] = scale_origin * scale_origin * init_cov[:, 0]
        # init_cov[:, 3] = scale_origin * scale_origin * init_cov[:, 3]
        # init_cov[:, 5] = scale_origin * scale_origin * init_cov[:, 5]
        if preprocessing_params["scale"] > 1.0:
            init_cov[:, 0] = init_cov[:, 0] * preprocessing_params["scale"] * preprocessing_params["scale"]
            init_cov[:, 3] = init_cov[:, 3] * preprocessing_params["scale"] * preprocessing_params["scale"]
            init_cov[:, 5] = init_cov[:, 5] * preprocessing_params["scale"] * preprocessing_params["scale"]
        else:
            init_cov = init_cov * preprocessing_params["scale"] * preprocessing_params["scale"]

        if preprocessing_params["sim_area"] is not None:
            boundary = preprocessing_params["sim_area"]
            assert len(boundary) == 6
            mask = torch.ones(transformed_pos.shape[0], dtype=torch.bool).to(device="cuda")
            for i in range(3):
                mask = torch.logical_and(mask, transformed_pos[:, i] > boundary[2 * i])
                mask = torch.logical_and(mask, transformed_pos[:, i] < boundary[2 * i + 1])
        
            unselected_pos = transformed_pos[~mask, :]
            unselected_cov = init_cov[~mask, :]
            unselected_opacity = init_opacity[~mask, :]
            unselected_shs = init_shs[~mask, :]

            transformed_pos = transformed_pos[mask, :]
            init_cov = init_cov[mask, :]
            init_opacity = init_opacity[mask, :]
            init_shs = init_shs[mask, :]
            
            
        if preprocessing_params["sim_selected_area"] is not None:
            boundary = preprocessing_params["sim_selected_area"]
            assert len(boundary) == 6
            mask = torch.ones(transformed_pos.shape[0], dtype=torch.bool).to(device="cuda")
            for i in range(3):
                mask = torch.logical_and(mask, transformed_pos[:, i] > boundary[2 * i])
                mask = torch.logical_and(mask, transformed_pos[:, i] < boundary[2 * i + 1])
        
            transformed_pos = transformed_pos[mask, :]
            init_cov = init_cov[mask, :]
            init_opacity = init_opacity[mask, :]
            init_shs = init_shs[mask, :]
        # import ipdb; ipdb.set_trace()
        
        # transformed_pos = shift2center111(transformed_pos)
        # scale_origin = scale_origin * preprocessing_params["scale"]

        # modify covariance matrix accordingly

        if args.debug:
            particle_position_tensor_to_ply(
                transformed_pos,
                f"./log/transformed_particles_{mi}.ply",
            )

        # fill particles if needed
        # r_index = torch.randint(0, init_pos.shape[0], size = [int(init_pos.shape[0] * 0.005)])
        # transformed_pos = torch.index_select(transformed_pos, 0, torch.tensor(r_index, device=device))
        # init_cov = torch.index_select(init_cov, 0, torch.tensor(r_index, device=device))
        # init_opacity = torch.index_select(init_opacity, 0, torch.tensor(r_index, device=device))
        # init_shs = torch.index_select(init_shs, 0, torch.tensor(r_index, device=device))
        # if preprocessing_params['scale'] < 1.0:
        #     transformed_pos = reduce_particles(
        #         pos = transformed_pos,
        #         grid_n=filling_params["n_grid"],
        #         max_particles_per_cell=filling_params["max_partciels_per_cell"],
        #         boundary=filling_params["boundary"]
        #     )
        # cov_scale = preprocessing_params["scale"] if camera_params["init_radius"] >= 6 else preprocessing_params["scale"] * (6 - camera_params["init_radius"])
        if preprocessing_params["scale"] < 0.5 and material_params["material"] != "fluid":
            
            # r_index = torch.randint(0, init_pos.shape[0], size = [int(init_pos.shape[0] * 0.005)])
            # transformed_pos = torch.index_select(transformed_pos, 0, torch.tensor(r_index, device=device))
            # init_cov = torch.index_select(init_cov, 0, torch.tensor(r_index, device=device))
            # init_opacity = torch.index_select(init_opacity, 0, torch.tensor(r_index, device=device))
            # init_shs = torch.index_select(init_shs, 0, torch.tensor(r_index, device=device))
        #     r_index = torch.randint(0, init_pos.shape[0], size = [int(init_pos.shape[0] * preprocessing_params['scale'])])
        #     transformed_pos = torch.index_select(transformed_pos, 0, torch.tensor(r_index, device=device))
        #     init_cov = torch.index_select(init_cov, 0, torch.tensor(r_index, device=device))
        #     init_opacity = torch.index_select(init_opacity, 0, torch.tensor(r_index, device=device))
        #     init_shs = torch.index_select(init_shs, 0, torch.tensor(r_index, device=device))
            # if init_rot is not None:
            #     init_rot = torch.index_select(init_rot, 0, torch.tensor(r_index, device=device))
            transformed_pos, init_shs, init_opacity, init_cov = reduce_particles(transformed_pos, init_opacity, init_cov, init_shs, preprocessing_params["scale"])
        gs_num = transformed_pos.shape[0]
        print("object_nums: ", gs_num)
        device = "cuda"
        filling_params = preprocessing_params["particle_filling"]
        if filling_params is not None:
            print("Filling internal particles...")
            # breakpoint()
            mpm_init_pos_object = fill_particles(
                pos=transformed_pos,
                opacity=init_opacity,
                cov=init_cov,
                grid_n=filling_params["n_grid"],
                max_samples=int(filling_params["max_particles_num"]),
                grid_dx=grid_params["grid_lim"] / filling_params["n_grid"],
                density_thres=filling_params["density_threshold"],
                search_thres=filling_params["search_threshold"],
                max_particles_per_cell=filling_params["max_partciels_per_cell"],
                search_exclude_dir=filling_params["search_exclude_direction"],
                ray_cast_dir=filling_params["ray_cast_direction"],
                boundary=filling_params["boundary"],
                smooth=filling_params["smooth"],
            ).to(device=device)

            if args.debug:
                particle_position_tensor_to_ply(mpm_init_pos_object, f"./log/filled_particles_{mi}.ply")
        else:
            mpm_init_pos_object = transformed_pos.to(device=device)
        # breakpoint()
        if filling_params is not None and filling_params["visualize"] == True and mpm_init_pos_object.shape[0] != gs_num:
            # init_opacity[:gs_num] = init_opacity_1
            shs_object, opacity_object, mpm_init_cov_object = init_filled_particles(
                mpm_init_pos_object[:gs_num],
                init_shs,
                init_cov,
                init_opacity,
                mpm_init_pos_object[gs_num:],
            )
            if (material_params["material"] == "fluid" or material_params["material"] == "jam") and filling_params["surface"]:
                shs_object = shs_object[gs_num:]
                opacity_object = opacity_object[gs_num:]
                mpm_init_cov_object = mpm_init_cov_object[gs_num:]
                mpm_init_pos_object = mpm_init_pos_object[gs_num:]
            gs_num = mpm_init_pos_object.shape[0]
        else:
            mpm_init_cov_object = torch.zeros((mpm_init_pos_object.shape[0], 6), device=device)
            mpm_init_cov_object[:gs_num] = init_cov
            # breakpoint()
            shs_object = torch.zeros((mpm_init_pos_object.shape[0], 1, 3), device=device)
            shs_object[:gs_num] = init_shs
            opacity_object = init_opacity
            
        if preprocessing_params["object_area"] is not None:
            # breakpoint()
            boundary = preprocessing_params["object_area"]
            assert len(boundary) == 6
            mask = torch.ones(mpm_init_pos_object.shape[0], dtype=torch.bool).to(device="cuda")
            for i in range(3):
                mask = torch.logical_and(mask, mpm_init_pos_object[:, i] > boundary[2 * i])
                mask = torch.logical_and(mask, mpm_init_pos_object[:, i] < boundary[2 * i + 1])
            mpm_init_pos_object = mpm_init_pos_object[mask, :]
            mpm_init_cov_object = mpm_init_cov_object[mask, :]
            opacity_object = opacity_object[mask, :]
            shs_object = shs_object[mask, :]
            gs_num = mpm_init_pos_object.shape[0]
            

        # import ipdb; ipdb.set_trace()
            
        # mpm_init_material_object = torch.empty(0)
        # if material_params["material"] == "jelly":
        #     mpm_init_material_object = torch.full([mpm_init_pos_object.shape[0]], 0).to(device = device)
        #     object_material_index.append(0)
        # elif material_params["material"] == "metal":
        #     mpm_init_material_object = torch.full([mpm_init_pos_object.shape[0]], 1).to(device = device)
        #     object_material_index.append(1)
        # elif material_params["material"] == "sand":
        #     mpm_init_material_object = torch.full([mpm_init_pos_object.shape[0]], 2).to(device = device)
        #     object_material_index.append(2)
        # elif material_params["material"] == "foam":
        #     mpm_init_material_object = torch.full([mpm_init_pos_object.shape[0]], 3).to(device = device)
        #     object_material_index.append(3)
        # elif material_params["material"] == "snow":
        #     mpm_init_material_object = torch.full([mpm_init_pos_object.shape[0]], 4).to(device = device)
        #     object_material_index.append(4)
        # elif material_params["material"] == "plasticine":
        #     mpm_init_material_object = torch.full([mpm_init_pos_object.shape[0]], 5).to(device = device)
        #     object_material_index.append(5)
        # elif material_params["material"] == "fluid":
        #     mpm_init_material_object = torch.full([mpm_init_pos_object.shape[0]], 6).to(device = device)
        #     object_material_index.append(6)
        # else:
        #     raise TypeError("Undefined material type")
        
        
        # if material_params["type"] == "floor":
        #     object_index = mi - 1
        #     exist_floor = True
        # else:
        #     object_index = mi
        object_index = mi
        mpm_particle_material_object = torch.full([mpm_init_pos_object.shape[0]], object_index, dtype=torch.int32).to(device = device)

        if mpm_init_pos_object.shape[0] != 0:
            mpm_init_vol_object = get_particle_volume(
                mpm_init_pos_object,
                grid_params["n_grid"],
                grid_params["grid_lim"] / grid_params["n_grid"],
                unifrom=(material_params["material"] == "sand" or material_params["material"] == "fluid" or material_params["material"] == "jam"),
                # unifrom=(material_params["material"] == "sand"),
            ).to(device=device)
        else:
            mpm_init_vol_object = torch.zeros(size = [0], device = device)
            
        # breakpoint()
            
        
        # mpm_init_vol_object = mpm_init_vol_object

        if preprocessing_params["opacity_scale"] < 1.0 and (material_params["material"] == "fluid" or material_params["material"] == "jam"):
            opacity_object = opacity_object * preprocessing_params["opacity_scale"]
            opacity_object = torch.mean(opacity_object).repeat(opacity_object.shape)

        # ipdb.set_trace()
        # mpm_init_vol_object = torch.ones(mpm_init_vol_object.shape, device = device)*1e-7
        # mpm_init_vol_object = (mpm_init_vol_object * 0.01)

        
        if material_params["material"] == "fluid" or material_params["material"] == "jam":
            # cov_object = torch.mean(mpm_init_cov_object)
            cov_object = ((mpm_init_vol_object) / (3.14 * 4 / 3)) ** (2 / 3) * 3
            # breakpoint()
        #     vol_cov_mean = torch.mean(mpm_init_vol_object) ** (1/3) / (12 * 3.14)
        #     # ipdb.set_trace()
        #     cov_mean = torch.min(vol_cov_mean ** 2, cov_mean)
            mpm_init_cov_object[:, 0] = mpm_init_cov_object[:, 3] = mpm_init_cov_object[:, 5] = cov_object
            
            # # breakpoint()
            mpm_init_cov_object[:, 1] = mpm_init_cov_object[:, 2] = mpm_init_cov_object[:, 4] = torch.tensor([0]).repeat([cov_object.shape[0]])
            # mpm_init_cov_object = cov_object.repeat(mpm_init_cov_object.shape)
            # breakpoint()
            # mpm_init_cov_object = cov_object.unsqueeze(1).expand([cov_object.shape[0], 6])
            if preprocessing_params["color"] != None:
                from utils.sh_utils import RGB2SH
                SH_color = RGB2SH(torch.tensor(preprocessing_params["color"], device=device) / 255)
                shs_object = SH_color.repeat([shs_object.shape[0], 1, 1])
                # breakpoint()
            else:
                shs_mean = torch.mean(shs_object, axis = 0)
                shs_object = shs_mean.repeat([shs_object.shape[0], 1, 1])
        
        # if preprocessing_params["color"] != None:
        #     from utils.sh_utils import RGB2SH
        #     SH_color = RGB2SH(torch.tensor(preprocessing_params["color"], device=device) / 255)
        #     shs_object = SH_color.repeat([shs_object.shape[0], 1, 1])
                
        # breakpoint()  # -1.77245 -1.77245, 1.50936
        # shs_object = torch.tensor([[[-1.77245, -1.77245, 1.50936]]], device=device).expand([shs_object.shape[0], 1, 3])
        # # breakpoint()
        # mpm_init_cov_object[:, 0] = mpm_init_cov_object[:, 3] = mpm_init_cov_object[:, 5] = torch.min(mpm_init_cov_object)
        # mpm_init_cov_object[:, 1] = mpm_init_cov_object[:, 2] = mpm_init_cov_object[:, 4] = torch.tensor([0]).repeat([mpm_init_vol_object.shape[0]])
        # opacity_object = torch.tensor([1.0], device=device).expand([shs_object.shape[0], 1])
        # breakpoint()
        # mpm_init_vol_object = torch.index_select(mpm_init_vol_object, 0, torch.tensor(r_index, device=device))
        # breakpoint()
        if len(add_particle_params) != 0 and add_particle_params["object_material_index"] == mi:
            # breakpoint()
            tensor_x = mpm_init_pos_object
            # index_p = select_particle_x_inside(tensor_x, add_particle_params["boundary"])
            boundary = add_particle_params["boundary"]
            assert len(boundary) == 6
            index_p = torch.ones(mpm_init_pos_object.shape[0], dtype=torch.bool).to(device="cuda")
            for i in range(3):
                index_p = torch.logical_and(index_p, mpm_init_pos_object[:, i] >= boundary[2 * i])
                index_p = torch.logical_and(index_p, mpm_init_pos_object[:, i] <= boundary[2 * i + 1])
            mpm_add_pos = tensor_x[index_p]
            # ipdb.set_trace()
            mpm_add_cov = mpm_init_cov_object[index_p]
            mpm_add_vol = mpm_init_vol_object[index_p]
            # mpm_add_material = mpm_init_material_object[index_p]
            mpm_add_particle_material = mpm_particle_material_object[index_p]
            index_p_nf = index_p[:gs_num]
            mpm_add_shs = shs_object[index_p_nf]
            mpm_add_opacity = opacity_object[index_p_nf]
            mpm_add_pos = mpm_add_pos + torch.tensor(add_particle_params["move_pos"], device='cuda')
            index_p = torch.tensor(index_p).to(device = 'cuda')
            index_p_nf = torch.tensor(index_p_nf).to(device = 'cuda')
            # breakpoint()
            # ipdb.set_trace()

            if add_particle_params["additional"]:
                mpm_init_pos_object = torch.empty(0, device = device)
                mpm_init_cov_object = torch.empty(0, device = device)
                mpm_init_vol_object = torch.empty(0, device = device)
                # mpm_init_material_object = mpm_init_material_object[~index_p]
                mpm_particle_material_object = torch.empty(0, device = device)
                shs_object = torch.empty(0, device = device)
                opacity_object = torch.empty(0, device = device)
                gs_num = 0

        # breakpoint()
        # ipdb.set_trace()
        scale_origin.unsqueeze_(0)
        original_mean_pos.unsqueeze_(0)
        center_point.unsqueeze_(0)
        # breakpoint()
        if mi == 0:
            mpm_init_pos = mpm_init_pos_object
            mpm_init_shs = shs_object
            mpm_init_opacity = opacity_object
            mpm_init_cov = mpm_init_cov_object
            mpm_init_vol = mpm_init_vol_object
            # mpm_init_material = mpm_init_material_object
            scales = scale_origin
            original_mean_poss = original_mean_pos
            center_points = center_point
            # init_screen_points_scene = init_screen_points
            if preprocessing_params["sim_area"] is not None:
                unselected_cov_scene = unselected_cov
                unselected_opacity_scene = unselected_opacity
                unselected_pos_scene = unselected_pos
                unselected_shs_scene = unselected_shs
            mpm_particle_material = mpm_particle_material_object
        else:
            mpm_init_pos = torch.cat([mpm_init_pos, mpm_init_pos_object])
            mpm_init_shs = torch.cat([mpm_init_shs, shs_object])
            mpm_init_cov = torch.cat([mpm_init_cov, mpm_init_cov_object])
            mpm_init_opacity = torch.cat([mpm_init_opacity, opacity_object])
            mpm_init_vol = torch.cat([mpm_init_vol, mpm_init_vol_object])
            # mpm_init_material = torch.cat([mpm_init_material, mpm_init_material_object])
            scales = torch.cat([scales, scale_origin])
            original_mean_poss = torch.cat([original_mean_poss, original_mean_pos])
            center_points = torch.cat([center_points, center_point])
            # init_screen_points_scene = torch.cat([init_screen_points_scene, init_screen_points])

            if preprocessing_params["sim_area"] is not None:
                unselected_cov_scene = torch.cat([unselected_cov_scene, unselected_cov])
                unselected_opacity_scene = torch.cat([unselected_opacity_scene, unselected_opacity])
                unselected_pos_scene = torch.cat([unselected_pos_scene, unselected_pos])
                unselected_shs_scene = torch.cat([unselected_shs_scene, unselected_shs])
            mpm_particle_material = torch.cat([mpm_particle_material, mpm_particle_material_object])
        fill_gs_nums.append(fill_gs_nums[mi] + mpm_init_pos_object.shape[0])
        gs_nums.append(fill_gs_nums[mi] + gs_num)
        unselected_gs_nums.append(unselected_pos_scene.shape[0])
        rotation_matricess.append(rotation_matrices)
        # breakpoint()
        # args.sh_degree = 3
        
    # breakpoint()
    # import ipdb; ipdb.set_trace()
    object_num = len(materials_params)
    if args.debug:
        print("check *.ply files to see if it's ready for simulation")

    # init the mpm solver
    print("Initializing MPM solver and setting up boundary conditions...")
    # set up the mpm solver
    mpm_solver = MPM_Simulator_WARP(10)
    # breakpoint()
    if len(add_particle_params) != 0:
        # import ipdb; ipdb.set_trace()
        
        # if torch.numel(unselected_pos_scene) != 0:
        #     unselected_index_p = select_particle_x_inside(unselected_pos_scene, add_particle_params["boundary"])
        #     unselected_mpm_add_pos = unselected_pos_scene[unselected_index_p]
        #     torch.cat([mpm_add_pos, unselected_mpm_add_pos])
        #     torch.cat([mpm_add_cov, unselected_cov_scene[unselected_index_p]])

        #     unselected_mpm_add_vol = get_particle_volume(
        #         unselected_mpm_add_pos,
        #         grid_params["n_grid"],
        #         grid_params["grid_lim"] / grid_params["n_grid"],
        #         unifrom=materials_params[material_index]["material"] == "sand",
        #     ).to(device=device)
        #     torch.cat([mpm_add_vol, unselected_mpm_add_vol])
            
        #     unselected_mpm_add_material = torch.full([len(unselected_index_p)], object_material_index[material_index])
        #     torch.cat([mpm_add_material, unselected_mpm_add_material])
            
        #     unselected_mpm_add_particle_material = torch.full([len(unselected_index_p)], material_index)
        #     torch.cat([mpm_add_particle_material, unselected_mpm_add_particle_material])
            
        #     unselected_mpm_add_shs = unselected_shs_scene[unselected_index_p]
        #     unsele
            
        # mpm_add_screen_points = init_screen_points_scene[index_p_nf]
        # breakpoint()
        add_material_params = {}
        material_index = add_particle_params["object_material_index"]
        if "E" in materials_params[material_index].keys():
            add_material_params["E"] = materials_params[material_index]["E"]
        if "nu" in materials_params[material_index].keys():
            add_material_params["nu"] = materials_params[material_index]["nu"]
        if "yield_stress" in materials_params[material_index].keys():
            add_material_params["yield_stress"] = materials_params[material_index]["yield_stress"]
        if "bulk_modulus" in materials_params[material_index].keys():
            add_material_params["bulk_modulus"] = materials_params[material_index]["bulk_modulus"]
        if "density" in materials_params[material_index].keys():
            add_material_params["density"] = materials_params[material_index]["density"]
        add_frames = min(time_params["frame_num"], round((add_particle_params["end_time"] - add_particle_params["start_time"])/ time_params["frame_dt"]))
        all_n_particles = add_frames * mpm_add_pos.shape[0] + mpm_init_pos.shape[0]
    else:
        all_n_particles = mpm_init_pos.shape[0]
    
    # ob_num = object_num - 1 if exist_floor else object_num
    # breakpoint()
    mpm_solver.load_initial_data_from_torch(
        mpm_init_pos,
        mpm_init_vol,
        # mpm_init_material,
        all_n_particles,
        mpm_init_cov,
        mpm_init_shs,
        tensor_link=mpm_particle_material,
        n_grid=grid_params["n_grid"],
        grid_lim=grid_params["grid_lim"],
        object_nums=object_num,
        shs_mix=args.shs_mix
    )
    print(all_n_particles)
    # material_params["n_grid"] = grid_params["n_grid"]
    # material_params["grid_lim"] = grid_params["grid_lim"]
    mpm_solver.set_parameters_dict(grid_params, materials_params, fill_gs_nums)
    
    # ipdb.set_trace()

    # Note: boundary conditions may depend on mass, so the order cannot be changed!
    set_boundary_conditions(mpm_solver, bc_params, time_params)

    mpm_solver.finalize_mu_lam()

    # camera setting
    mpm_space_viewpoint_center = (
        torch.tensor(camera_params["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
    )
    mpm_space_vertical_upward_axis = (
        torch.tensor(camera_params["mpm_space_vertical_upward_axis"])
        .reshape((1, 3))
        .cuda()
    )
    
    scale_mean = scales.mean()
    scene_mean_pos = torch.mean(original_mean_poss, 0)
    scene_mean_pos += grid_center / scale_mean
    # scene_mean_pos = torch.tensor([1.0046, 1.0056, 1.0107], device='cuda')
    camera_rotation_matrices = [torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).to(device = device)]
    (
        viewpoint_center_worldspace,
        observant_coordinates,
    ) = get_center_view_worldspace_and_observant_coordinate(
        mpm_space_viewpoint_center,
        mpm_space_vertical_upward_axis,
        camera_rotation_matrices,
        scale_mean,
        scene_mean_pos,
        grid_center
    )
    # breakpoint()
    for mi in range(object_num):
        if unselected_gs_nums[mi] != unselected_gs_nums[mi+1]:
            unselected_pos_mi = unselected_pos_scene[unselected_gs_nums[mi]: unselected_gs_nums[mi+1]]
            unselected_pos_mi = undotransform2origin(
                undotransform2center(unselected_pos_mi, grid_center), scale_mean, scene_mean_pos
            )
            unselected_pos_scene[unselected_gs_nums[mi]: unselected_gs_nums[mi+1]] = unselected_pos_mi
            
            unselected_cov_mi = unselected_cov_scene[unselected_gs_nums[mi]: unselected_gs_nums[mi+1]]
                        
            unselected_cov_mi = unselected_cov_mi / (scale_mean * scale_mean)
            # unselected_cov_mi[:, 0] = unselected_cov_mi[:, 0] / (scale_mean * scale_mean)
            # unselected_cov_mi[:, 3] = unselected_cov_mi[:, 3] / (scale_mean * scale_mean)
            # unselected_cov_mi[:, 5] = unselected_cov_mi[:, 5] / (scale_mean * scale_mean)
            # breakpoint()
            unselected_cov_mi = apply_inverse_cov_rotations(unselected_cov_mi, rotation_matricess[mi])
            unselected_cov_scene[unselected_gs_nums[mi]: unselected_gs_nums[mi+1]] = unselected_cov_mi
            
    # run the simulation
    if args.output_ply or args.output_h5:
        directory_to_save = os.path.join(args.output_path, "simulation_ply")
        if not os.path.exists(directory_to_save):
            os.makedirs(directory_to_save)

        save_data_at_frame(
            mpm_solver,
            directory_to_save,
            0,
            save_to_ply=args.output_ply,
            save_to_h5=args.output_h5,
        )

    substep_dt = time_params["substep_dt"]
    frame_dt = time_params["frame_dt"]
    frame_num = time_params["frame_num"]
    step_per_frame = int(frame_dt / substep_dt)
    opacity_render = mpm_init_opacity
    # shs_render = mpm_init_shs
    height = None
    width = None
    sim_time_avg = 0
    render_time_avg = 0
    first_modify = True
    
    if is_ply:
        camera_path = preprocessings_params[-1]["camera_path"]
    else:
        camera_path = model_path

    current_camera = get_camera_view(
        camera_path,
        default_camera_index=camera_params["default_camera_index"],
        center_view_world_space=viewpoint_center_worldspace,
        observant_coordinates=observant_coordinates,
        show_hint=camera_params["show_hint"],
        init_azimuthm=camera_params["init_azimuthm"],
        init_elevation=camera_params["init_elevation"],
        init_radius=camera_params["init_radius"],
        move_camera=camera_params["move_camera"],
        first_move_frame = camera_params["first_move_frame"],
        current_frame=0,
        delta_a=camera_params["delta_a"],
        delta_e=camera_params["delta_e"],
        delta_r=camera_params["delta_r"],
    )
    rasterize = initialize_resterize(
        current_camera, gaussians, pipeline, background
    )

    rasterizes = []
    if camera_params["multi_camera"]:
        for c in range(1, camera_params["camera_nums"]):
            current_camera_add = get_camera_view(
                camera_path,
                default_camera_index=camera_params["default_camera_index"],
                center_view_world_space=viewpoint_center_worldspace,
                observant_coordinates=observant_coordinates,
                show_hint=camera_params["show_hint"],
                init_azimuthm=camera_params["init_azimuthm"] + camera_params["camera_delta_a"] * c,
                init_elevation=camera_params["init_elevation"] + camera_params["camera_delta_e"] * c,
                init_radius=camera_params["init_radius"],
                move_camera=camera_params["move_camera"],
                first_move_frame = camera_params["first_move_frame"],
                current_frame=0,
                delta_a=camera_params["delta_a"],
                delta_e=camera_params["delta_e"],
                delta_r=camera_params["delta_r"],
            )
            rasterize_addi = initialize_resterize(
                current_camera_add, gaussians, pipeline, background
            )
            rasterizes.append(rasterize_addi)

    with open('energy.txt', 'a+') as f:

        if len(args.background_img) != 0:
            background = cv2.imread(args.background_img)
            # background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
            # 归一化到[0, 1]
            background = background.astype(np.float32) / 255.0
    
        for frame in tqdm(range(int(frame_num))):
            
            if len(add_particle_params) != 0:
                frame_t = frame_dt * (frame + 1)
                if frame_t > add_particle_params["start_time"] and frame_t < add_particle_params["end_time"]:
                    # import ipdb; ipdb.set_trace()
                    mpm_solver.add_particle(mpm_add_pos, mpm_add_vol, 
                    # mpm_add_material,
                        mpm_add_cov, mpm_add_shs, mpm_add_particle_material, add_material_params)
                    # import ipdb; ipdb.set_trace()
                    # breakpoint()
                    if mpm_init_shs.shape[0] == mpm_init_pos.shape[0]:
                        add_particle_num = mpm_add_pos.shape[0]
                    else:
                        add_particle_num = mpm_add_shs.shape[0]
                    if object_num == len(gs_nums):
                        gs_nums.append(fill_gs_nums[-1] + add_particle_num)
                        # breakpoint()
                        center_points = torch.cat([center_points, center_points[add_particle_params["object_material_index"], None]])
                        original_mean_poss = torch.cat([original_mean_poss, original_mean_poss[add_particle_params["object_material_index"], None]])
                        scales = torch.cat([scales, scales[add_particle_params["object_material_index"], None]])
                        rotation_matricess.append(rotation_matricess[add_particle_params["object_material_index"]])
                        # original_mean_poss = torch.cat()
                    else:
                        gs_nums[-1] = gs_nums[-1] + add_particle_num
                    opacity_render = torch.cat([opacity_render, mpm_add_opacity])
            
            # start = time.time()
            for step in range(step_per_frame):
                mpm_solver.p2g2p(frame, substep_dt, device=device)
            
            
            # energy = mpm_solver.export_energy_to_torch()
            # print("energy", energy, torch.sum(energy))
            # f.write(f"{torch.sum(energy)}\n")

            # mpm_solver.export_energy_to_torch()
            # print("energy", mpm_solver.mpm_model.energy)
            # end = time.time()
            # print("模拟时间: %.2f秒" % (end - start))
            # sim_time_avg = sim_time_avg + (end - start)

            if args.output_ply or args.output_h5:
                save_data_at_frame(
                    mpm_solver,
                    directory_to_save,
                    frame + 1,
                    save_to_ply=args.output_ply,
                    save_to_h5=args.output_h5,
                )
                    # shs_render = torch.cat([shs_render, mpm_add_shs])
                # breakpoint()
            
            if len(modify_material_params) != 0 and first_modify:
                frame_t = frame_dt * (frame + 1)
                # breakpoint()
                if frame_t > modify_material_params["start_time"]:
                    mpm_solver.modify_particle_material(modify_material_params)
                    if modify_material_params["material"] == "fluid":
                        mi = modify_material_params["object_material_index"]
                        shs_object = mpm_init_shs[fill_gs_nums[mi]: fill_gs_nums[mi + 1]]
                        cov_object = mpm_init_cov[fill_gs_nums[mi]: fill_gs_nums[mi + 1]]
                        vol_object = mpm_init_vol[fill_gs_nums[mi]: fill_gs_nums[mi + 1]]
                        p_color = preprocessings_params[mi]["color"]
                        if p_color != None:
                            from utils.sh_utils import RGB2SH
                            SH_color = RGB2SH(torch.tensor(p_color, device=device) / 255)
                            shs_object = SH_color.repeat([shs_object.shape[0], 1, 1])
                            # breakpoint()
                        else:
                            shs_mean = torch.mean(shs_object, axis = 0)
                            shs_object = shs_mean.repeat([shs_object.shape[0], 1, 1])
                        vol_object = torch.mean(vol_object).repeat(vol_object.shape[0])
                        cov_object_mean = ((vol_object) / (3.14 * 4 / 3)) ** (2 / 3) * 3
                        cov_object[:, 0] = cov_object[:, 3] = cov_object[:, 5] = cov_object_mean
                        cov_object[:, 1] = cov_object[:, 2] = cov_object[:, 4] = torch.tensor([0]).repeat([cov_object_mean.shape[0]])
                        # cov_object[:, 1] = cov_object[:, 2] = cov_object[:, 4] = cov_object_mean
                        mpm_solver.modify_cov_shs(shs_object, cov_object, fill_gs_nums[mi])
                        opacity_scale = preprocessings_params[mi]["opacity_scale"]
                        if opacity_scale < 1.0:
                            opacity_object = opacity_render[fill_gs_nums[mi]: fill_gs_nums[mi + 1]]
                            opacity_object = opacity_object * opacity_scale
                            opacity_object = torch.mean(opacity_object).repeat(opacity_object.shape)
                            opacity_render[fill_gs_nums[mi]: fill_gs_nums[mi + 1]] = opacity_object
                    first_modify = False
                
            with torch.no_grad():
                if args.render_img:
                    # start = time.time()
                    # pos = mpm_solver.export_particle_x_to_torch()[:gs_num].to(device)
                    cov3D = mpm_solver.export_particle_cov_to_torch().clone()
                    # cov3D = mpm_init_cov.clone()
                    # rot = mpm_solver.export_particle_R_to_torch().clone()
                    pos = mpm_solver.export_particle_x_to_torch().clone().to(device)
                    shs = mpm_solver.export_particle_shs_to_torch().clone()
                    shs = shs.view(-1, 1, 3).to(device)
                    # breakpoint()
                    cov3D = cov3D.view(-1, 6).to(device)

                    # rot = rot.view(-1, 3, 3).to(device)
                    # if frame > 120:
                    #     breakpoint()
                    # breakpoint()
                    # # breakpoint()

                    # pos = apply_inverse_rotations(undotransform2origin(
                    #         undoshift2center111(pos), scale_origin, original_mean_pos), rotation_matrices,
                    # )
                        
                    # cov3D = cov3D / (scale_origin * scale_origin)
                    # cov3D = apply_inverse_cov_rotations(cov3D, rotation_matrices)
                    # opacity = opacity_render
                    # shs = shs_render
                    # if preprocessing_params["sim_area"] is not None:
                    #     pos = torch.cat([pos, unselected_pos], dim=0)
                    #     cov3D = torch.cat([cov3D, unselected_cov], dim=0)
                    #     opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
                    #     shs = torch.cat([shs_render, unselected_shs], dim=0)
                    
                    
                    pos = torch.cat([pos[fill_gs_nums[mi]: gs_nums[mi]] for mi in range(len(gs_nums))])
                    cov3D = torch.cat([cov3D[fill_gs_nums[mi]: gs_nums[mi]] for mi in range(len(gs_nums))])
                    # rot = torch.cat([rot[fill_gs_nums[mi]: gs_nums[mi]] for mi in range(len(gs_nums))])

                    shs = torch.cat([shs[fill_gs_nums[mi]: gs_nums[mi]] for mi in range(len(gs_nums))])
                    
                    gs_nums_1 = [0]
                    for mi in range(len(gs_nums)):
                        gs_nums_1 = gs_nums_1 + [gs_nums_1[mi] + gs_nums[mi] - fill_gs_nums[mi]]
                    # breakpoint()
                    # if frame == 0:
                    #     past_pos = pos[gs_nums_1[1]:gs_nums_1[2]]
                    # pos[gs_nums_1[1]:gs_nums_1[2]] = past_pos
                    
                    for mi in range(len(gs_nums_1) - 1):
                        # center_point = torch.tensor(preprocessings_params[mi]["center_point"]).to(device=device)
                        # center_point = center_points[mi]
                        pos_mi = pos[gs_nums_1[mi]:gs_nums_1[mi+1]]
                        pos_mi = undotransform2origin(
                                undotransform2center(pos_mi, grid_center), scale_mean, scene_mean_pos
                            )
                        # import ipdb; ipdb.set_trace()
                        pos[gs_nums_1[mi]:gs_nums_1[mi+1]] = pos_mi
                        cov3D_mi = cov3D[gs_nums_1[mi]:gs_nums_1[mi+1]]
                        
                        # cov3D_mi[:, 0] = cov3D_mi[:, 0] / (scale_mean * scale_mean)
                        # cov3D_mi[:, 3] = cov3D_mi[:, 3] / (scale_mean * scale_mean)
                        # cov3D_mi[:, 5] = cov3D_mi[:, 5] / (scale_mean * scale_mean)
                        cov3D_mi = cov3D_mi / (scale_mean * scale_mean)
                        # breakpoint()
                        cov3D_mi = apply_inverse_cov_rotations(cov3D_mi, rotation_matricess[mi])
                        cov3D[gs_nums_1[mi]:gs_nums_1[mi+1]] = cov3D_mi


                    # pos = pos[gs_nums[0]:]
                    # cov3D = cov3D[gs_nums[0]:]
                    # rot = rot[gs_nums[0]:]
                    # center_point = torch.tensor(materials_params[0]["center_point"]).to(device=device)
                    # pos = apply_inverse_rotations(
                    #         undotransform2origin(
                    #             undotransform2center(pos, center_point), scales[0], original_mean_poss[0]
                    #         ),
                    #         rotation_matricess[0],
                    #     )
                    # cov3D = cov3D / (scales[0] * scales[0])
                    # cov3D = apply_inverse_cov_rotations(cov3D, rotation_matricess[0])
                    
                    opacity = opacity_render
                    # shs = shs_render
                    
                    # breakpoint()
                    if torch.numel(unselected_pos_scene) != 0:
                        pos = torch.cat([pos, unselected_pos_scene], dim=0)
                        cov3D = torch.cat([cov3D, unselected_cov_scene], dim=0)
                        opacity = torch.cat([opacity_render, unselected_opacity_scene], dim=0)
                        shs = torch.cat([shs, unselected_shs_scene], dim=0)

                    
                    # breakpoint()
                    # screenspace_points = (
                    #     torch.zeros_like(
                    #         pos,
                    #         dtype=pos.dtype,
                    #         requires_grad=False,
                    #         device="cuda",
                    #     )
                    #     + 0
                    # )
                    # import ipdb; ipdb.set_trace()
                    screen_points_scene = torch.zeros_like(pos, dtype=pos.dtype, requires_grad=True, device="cuda") + 0
                    # rot = torch.zeros(shape=[pos.shape[0], 3, 3], dtype=pos.dtype, requires_grad=True, device="cuda")
                    colors_precomp = convert_SH(shs, current_camera, gaussians, pos)
                    # end = time.time()
                    # print("提取时间: %.2f秒" % (end - start))

                    # start = time.time()
                    rendering, raddi, depth_alpha  = rasterize(
                        means3D=pos,
                        means2D=screen_points_scene,
                        shs=None,
                        colors_precomp=colors_precomp,
                        opacities=opacity,
                        scales=None,
                        rotations=None,
                        cov3D_precomp=cov3D,
                    )
                    # end = time.time()
                    # print("渲染时间: %.2f秒" % (end - start))
                    # render_time_avg = render_time_avg + (end - start)
                    depth, alpha = torch.chunk(depth_alpha, 2)
                    cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
                    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                    if height is None or width is None:
                        height = cv2_img.shape[0] // 2 * 2
                        width = cv2_img.shape[1] // 2 * 2
                    assert args.output_path is not None
                    # if frame > 200 and frame % 100 == 0:
                    #     cv2.imwrite(
                    #         os.path.join(args.output_path, f"{frame}.png".rjust(8, "0")),
                    #         255 * cv2_img,
                    #     )
                    # elif frame <= 200:
                    
                    if len(args.background_img) != 0:
                        alpha = alpha.permute(1, 2, 0).detach().cpu().numpy()
                        # alpha = np.squeeze(alpha)
                        # rgba = np.dstack(cv2_img, alpha)
                        # alpha_expanded = np.expand_dims(alpha, axis=2)
                        alpha_expanded = np.repeat(alpha, 3, axis=2)
                        background = cv2.resize(background, (height, width))
                        # breakpoint()
                        composite = cv2_img * (1 - alpha_expanded) + background * alpha_expanded
                        cv2.imwrite(
                            os.path.join(args.output_path, f"{frame}.png".rjust(8, "0")),
                            255 * composite,
                        )
                    else:
                        cv2.imwrite(
                            os.path.join(args.output_path, f"{frame}.png".rjust(8, "0")),
                            255 * cv2_img,
                        )
                    
                    if args.only_water:
                        pos_water = pos[gs_nums_1[0]:gs_nums_1[1]]
                        cov3D_water = cov3D[gs_nums_1[0]:gs_nums_1[1]]
                        shs_water = shs[gs_nums_1[0]:gs_nums_1[1]]
                        opacity_water = opacity[gs_nums_1[0]:gs_nums_1[1]]
                        screen_points_scene_water = torch.zeros_like(pos_water, dtype=pos_water.dtype, requires_grad=True, device="cuda") + 0
                        # rot = torch.zeros(shape=[pos.shape[0], 3, 3], dtype=pos.dtype, requires_grad=True, device="cuda")
                        colors_precomp_water = convert_SH(shs_water, current_camera, gaussians, pos_water)
                        # end = time.time()
                        # print("提取时间: %.2f秒" % (end - start))

                        # start = time.time()
                        rendering, raddi, depth_alpha = rasterize(
                            means3D=pos_water,
                            means2D=screen_points_scene_water,
                            shs=None,
                            colors_precomp=colors_precomp_water,
                            opacities=opacity_water,
                            scales=None,
                            rotations=None,
                            cov3D_precomp=cov3D_water,
                        )
                        
                        
                        # end = time.time()
                        # print("渲染时间: %.2f秒" % (end - start))
                        # render_time_avg = render_time_avg + (end - start)
                        cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
                        if height is None or width is None:
                            height = cv2_img.shape[0] // 2 * 2
                            width = cv2_img.shape[1] // 2 * 2
                        assert args.output_path is not None
                        # if frame > 200 and frame % 100 == 0:
                        #     cv2.imwrite(
                        #         os.path.join(args.output_path, f"{frame}.png".rjust(8, "0")),
                        #         255 * cv2_img,
                        #     )
                        # elif frame <= 200:
                        out_path_water = os.path.join(args.output_path, f"only_water")
                        os.makedirs(out_path_water, exist_ok=True)
                        cv2.imwrite(
                            os.path.join(out_path_water, f"{frame}.png".rjust(8, "0")),
                            255 * cv2_img,
                        )
                    
                    if args.no_water:
                        pos_no_water = pos[gs_nums_1[1]:]
                        cov3D_no_water = cov3D[gs_nums_1[1]:]
                        shs_no_water = shs[gs_nums_1[1]:]
                        opacity_no_water = opacity[gs_nums_1[1]:]
                        screen_points_scene_no_water = torch.zeros_like(pos_no_water, dtype=pos_no_water.dtype, requires_grad=True, device="cuda") + 0
                        # rot = torch.zeros(shape=[pos.shape[0], 3, 3], dtype=pos.dtype, requires_grad=True, device="cuda")
                        colors_precomp_no_water = convert_SH(shs_no_water, current_camera, gaussians, pos_no_water)
                        # end = time.time()
                        # print("提取时间: %.2f秒" % (end - start))

                        # start = time.time()
                        rendering, raddi, depth_alpha = rasterize(
                            means3D=pos_no_water,
                            means2D=screen_points_scene_no_water,
                            shs=None,
                            colors_precomp=colors_precomp_no_water,
                            opacities=opacity_no_water,
                            scales=None,
                            rotations=None,
                            cov3D_precomp=cov3D_no_water,
                        )
                        # end = time.time()
                        # print("渲染时间: %.2f秒" % (end - start))
                        # render_time_avg = render_time_avg + (end - start)
                        cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
                        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                        if height is None or width is None:
                            height = cv2_img.shape[0] // 2 * 2
                            width = cv2_img.shape[1] // 2 * 2
                        assert args.output_path is not None
                        # if frame > 200 and frame % 100 == 0:
                        #     cv2.imwrite(
                        #         os.path.join(args.output_path, f"{frame}.png".rjust(8, "0")),
                        #         255 * cv2_img,
                        #     )
                        # elif frame <= 200:
                        out_path_no_water = os.path.join(args.output_path, f"no_water")
                        os.makedirs(out_path_no_water, exist_ok=True)
                        cv2.imwrite(
                            os.path.join(out_path_no_water, f"{frame}.png".rjust(8, "0")),
                            255 * cv2_img,
                        )
                        
                    # rendering, raddi = rasterize(
                    #     means3D=pos,
                    #     means2D=screen_points_scene,
                    #     shs=None,
                    #     colors_precomp=colors_precomp,
                    #     opacities=opacity / 0.2,
                    #     scales=None,
                    #     rotations=None,
                    #     cov3D_precomp=cov3D,
                    # )
                    # cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
                    # cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                    # if height is None or width is None:
                    #     height = cv2_img.shape[0] // 2 * 2
                    #     width = cv2_img.shape[1] // 2 * 2
                    # assert args.output_path is not None
                    # cv2.imwrite(
                    #     os.path.join(args.output_path, f"{frame}_1.png".rjust(8, "0")),
                    #     255 * cv2_img,
                    # )
                    if camera_params["multi_camera"]:
                        for ri, rasterize_addi in enumerate(rasterizes):
                            rendering, raddi = rasterize_addi(
                                means3D=pos,
                                means2D=screen_points_scene,
                                shs=None,
                                colors_precomp=colors_precomp,
                                opacities=opacity,
                                scales=None,
                                rotations=None,
                                cov3D_precomp=cov3D,
                            )
                            cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
                            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                            if height is None or width is None:
                                height = cv2_img.shape[0] // 2 * 2
                                width = cv2_img.shape[1] // 2 * 2
                            
                            out_path_addi = os.path.join(args.output_path, f"camera_{ri+1}")
                            os.makedirs(out_path_addi, exist_ok=True)
                            cv2.imwrite(
                                os.path.join(out_path_addi, f"{frame}.png".rjust(8, "0")),
                                255 * cv2_img,
                            )
        
        if frame % 10 == 0:
            gc.collect()
    # breakpoint()
    # print(sim_time_avg / frame_num)
    # print(render_time_avg / frame_num)
    if args.render_img and args.compile_video:
        fps = int(1.0 / time_params["frame_dt"])
        os.system(
            f"ffmpeg -framerate {fps} -i {args.output_path}/%04d.png -c:v libx264 -s {width}x{height} -y -pix_fmt yuv420p {args.output_path}/output.mp4"
        )
        if args.only_water:
            os.system(
                f"ffmpeg -framerate {fps} -i {args.output_path}/only_water/%04d.png -c:v libx264 -s {width}x{height} -y -pix_fmt yuv420p {args.output_path}/only_water/output.mp4"
            )
            
        if args.no_water:
            os.system(
                f"ffmpeg -framerate {fps} -i {args.output_path}/no_water/%04d.png -c:v libx264 -s {width}x{height} -y -pix_fmt yuv420p {args.output_path}/no_water/output.mp4"
            )
        if camera_params["multi_camera"]:
            for ci in range(1, camera_params["camera_nums"]):
                os.system(
                    f"ffmpeg -framerate {fps} -i {args.output_path}/camera_{ci}/%04d.png -c:v libx264 -s {width}x{height} -y -pix_fmt yuv420p {args.output_path}/camera_{ci}/output.mp4"
                )

