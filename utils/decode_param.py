import json
import warp as wp
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
from mpm_solver_warp.engine_utils import *


def decode_param_json(json_file):
    f = open(json_file)
    sim_params = json.load(f)
    grid_params = {}
    
    if "grid_lim" in sim_params.keys():
        grid_params["grid_lim"] = sim_params["grid_lim"]
    else:
        grid_params["grid_lim"] = 4.0

    if "n_grid" in sim_params.keys():
        grid_params["n_grid"] = sim_params["n_grid"]
    else:
        grid_params["n_grid"] = 200
        
    if "g" in sim_params.keys():
        grid_params["g"] = sim_params["g"]
    else:
        grid_params["g"] = [0, 0, -9.8]

    if "grid_v_damping_scale" in sim_params.keys():
        grid_params["grid_v_damping_scale"] = sim_params["grid_v_damping_scale"]

    materials_params = []
    preprocessings_params = []
    object_num = 1
    
    if "object_num" in sim_params.keys():
        object_num = sim_params["object_num"]
    for i in range(object_num):
        # material parameters
        material_params = {}
        if "material" in sim_params["material_params"][i].keys():
            material_params["material"] = sim_params["material_params"][i]["material"]
        else:
            material_params["material"] = "jelly"
            
        if "static" in sim_params["material_params"][i].keys():
            material_params["static"] = sim_params["material_params"][i]["static"]
        
            if "static_inf" in sim_params["material_params"][i].keys():
                material_params["static_inf"] = sim_params["material_params"][i]["static_inf"]
            else:
                material_params["static_inf"] = False
            
        if "type" in sim_params["material_params"][i].keys():
            material_params["type"] = sim_params["material_params"][i]["type"]
        else:
            material_params["type"] = None

        if "nu" in sim_params["material_params"][i].keys():
            material_params["nu"] = sim_params["material_params"][i]["nu"]
            if material_params["nu"] > 0.5 or material_params["nu"] < 0.0:
                raise ValueError("Poisson's ratio should be less than 0.5")
        else:
            material_params["nu"] = 0.4
            
        if "gamma" in sim_params["material_params"][i].keys():
            material_params["gamma"] = sim_params["material_params"][i]["gamma"]
        else:
            material_params["gamma"] = 3.0

        if "E" in sim_params["material_params"][i].keys():
            material_params["E"] = sim_params["material_params"][i]["E"]
        else:
            material_params["E"] = 1e5

        if "yield_stress" in sim_params["material_params"][i].keys():
            material_params["yield_stress"] = sim_params["material_params"][i]["yield_stress"]

        if "hardening" in sim_params["material_params"][i].keys():
            material_params["hardening"] = sim_params["material_params"][i]["hardening"]

        if "xi" in sim_params["material_params"][i].keys():
            material_params["xi"] = sim_params["material_params"][i]["xi"]

        if "permeability" in sim_params["material_params"][i].keys():
            material_params["permeability"] = sim_params["material_params"][i]["permeability"]

        if "porosity" in sim_params["material_params"][i].keys():
            material_params["porosity"] = sim_params["material_params"][i]["porosity"]
            
        if "compression" in sim_params["material_params"][i].keys():
            material_params["compression"] = sim_params["material_params"][i]["compression"]
            
        if "stretch" in sim_params["material_params"][i].keys():
            material_params["stretch"] = sim_params["material_params"][i]["stretch"]
            
        if "hardening_coe" in sim_params["material_params"][i].keys():
            material_params["hardening_coe"] = sim_params["material_params"][i]["hardening_coe"]

        if "friction_angle" in sim_params["material_params"][i].keys():
            material_params["friction_angle"] = sim_params["material_params"][i]["friction_angle"]

        if "plastic_viscosity" in sim_params["material_params"][i].keys():
            material_params["plastic_viscosity"] = sim_params["material_params"][i]["plastic_viscosity"]

        if "density" in sim_params["material_params"][i].keys():
            material_params["density"] = sim_params["material_params"][i]["density"]
        else:
            material_params["density"] = 200.0

        if "bulk_modulus" in sim_params["material_params"][i].keys():
            material_params["bulk_modulus"] = sim_params["material_params"][i]["bulk_modulus"]

        if "diff_coe" in sim_params["material_params"][i].keys():
            material_params["diff_coe"] = sim_params["material_params"][i]["diff_coe"]
        else:
            material_params["diff_coe"] = 0.0

        if "rpic_damping" in sim_params["material_params"][i].keys():
            material_params["rpic_damping"] = sim_params["material_params"][i]["rpic_damping"]

        if "pic_damping" in sim_params["material_params"][i].keys():
            material_params["pic_damping"] = sim_params["material_params"][i]["pic_damping"]

        if "softening" in sim_params["material_params"][i].keys():
            material_params["softening"] = sim_params["material_params"][i]["softening"]

        if "additional_material_params" in sim_params["material_params"][i].keys():
            additional_params = sim_params["material_params"][i]["additional_material_params"]
            for j in range(len(additional_params)):
                if not "point" in additional_params[j].keys():
                    raise TypeError("point is not defined")

                if not "size" in additional_params[j].keys():
                    raise TypeError("size is not defined")

                if not "E" in additional_params[j].keys():
                    raise TypeError("E is not defined")

                if not "nu" in additional_params[j].keys():
                    raise TypeError("nu is not defined")

                if not "density" in additional_params[j].keys():
                    additional_params[j]["density"] = material_params["density"]

            material_params["additional_material_params"] = additional_params
            
        materials_params.append(material_params)
            
    # preprocessing_params
    for i in range(object_num):
        preprocessing_params = {}
        
        if "model_path" in sim_params["preprocess_params"][i].keys():
            preprocessing_params["model_path"] = sim_params["preprocess_params"][i]["model_path"]
        else:
            print("model_path should not be valid.")
            assert False
        
        if "camera_path" in sim_params["preprocess_params"][i].keys():
            preprocessing_params["camera_path"] = sim_params["preprocess_params"][i]["camera_path"]
        
        if "center_point" in sim_params["preprocess_params"][i].keys():
            preprocessing_params["center_point"] = sim_params["preprocess_params"][i]["center_point"]
        else:
            preprocessing_params["center_point"] = [1.0, 1.0, 1.0]
        
        if "scale" in sim_params["preprocess_params"][i].keys():
            preprocessing_params["scale"] = sim_params["preprocess_params"][i]["scale"]
        else:
            preprocessing_params["scale"] = 1.0
        
        if "opacity_threshold" in sim_params["preprocess_params"][i].keys():
            preprocessing_params["opacity_threshold"] = sim_params["preprocess_params"][i]["opacity_threshold"]
        else:
            preprocessing_params["opacity_threshold"] = 0.02

        if "opacity_scale" in sim_params["preprocess_params"][i].keys():
            preprocessing_params["opacity_scale"] = sim_params["preprocess_params"][i]["opacity_scale"]
        else:
            preprocessing_params["opacity_scale"] = 1.0

        if "rotation_degree" in sim_params["preprocess_params"][i].keys():
            preprocessing_params["rotation_degree"] = sim_params["preprocess_params"][i]["rotation_degree"]
        else:
            preprocessing_params["rotation_degree"] = []

        if "rotation_axis" in sim_params["preprocess_params"][i].keys():
            preprocessing_params["rotation_axis"] = sim_params["preprocess_params"][i]["rotation_axis"]
        else:
            preprocessing_params["rotation_axis"] = []

        if "sim_area" in sim_params["preprocess_params"][i].keys():
            preprocessing_params["sim_area"] = sim_params["preprocess_params"][i]["sim_area"]
        else:
            preprocessing_params["sim_area"] = None

        if "object_area" in sim_params["preprocess_params"][i].keys():
            preprocessing_params["object_area"] = sim_params["preprocess_params"][i]["object_area"]
        else:
            preprocessing_params["object_area"] = None
            
        if "sim_selected_area" in sim_params["preprocess_params"][i].keys():
            preprocessing_params["sim_selected_area"] = sim_params["preprocess_params"][i]["sim_selected_area"]
        else:
            preprocessing_params["sim_selected_area"] = None
            
        if "color" in sim_params["preprocess_params"][i].keys():
            preprocessing_params["color"] = sim_params["preprocess_params"][i]["color"]
        else:
            preprocessing_params["color"] = None

        if "particle_filling" in sim_params["preprocess_params"][i].keys():
            preprocessing_params["particle_filling"] = sim_params["preprocess_params"][i]["particle_filling"]
            filling_params = preprocessing_params["particle_filling"]
            if not "n_grid" in filling_params.keys():
                filling_params["n_grid"] = material_params["n_grid"] * 4

            if not "density_threshold" in filling_params.keys():
                filling_params["density_threshold"] = 5.0

            if not "search_threshold" in filling_params.keys():
                filling_params["search_threshold"] = 3.0

            if not "max_particles_num" in filling_params.keys():
                filling_params["max_particles_num"] = 2000000

            if not "max_partciels_per_cell" in filling_params.keys():
                filling_params["max_partciels_per_cell"] = 1

            if not "search_exclude_direction" in filling_params.keys():
                filling_params["search_exclude_direction"] = 5

            if not "ray_cast_direction" in filling_params.keys():
                filling_params["ray_cast_direction"] = 4

            if not "boundary" in filling_params.keys():
                filling_params["boundary"] = None

            if not "smooth" in filling_params.keys():
                filling_params["smooth"] = False
            
            if not "visualize" in filling_params.keys():
                filling_params["visualize"] = False
            
            if not "surface" in filling_params.keys():
                filling_params["surface"] = True
        else:
            preprocessing_params["particle_filling"] = None
            
        preprocessings_params.append(preprocessing_params)
        
    # boundary conditions
    bc_params = {}
    if "boundary_conditions" in sim_params.keys():
        bc_params = sim_params["boundary_conditions"]

    # time step
    time_params = {}
    if "substep_dt" in sim_params.keys():
        time_params["substep_dt"] = sim_params["substep_dt"]
    else:
        time_params["substep_dt"] = 1e-4

    if "frame_dt" in sim_params.keys():
        time_params["frame_dt"] = sim_params["frame_dt"]
    else:
        time_params["frame_dt"] = 1e-2

    if "frame_num" in sim_params.keys():
        time_params["frame_num"] = sim_params["frame_num"]
    else:
        time_params["frame_num"] = 100
    
    add_particle_params = {} 
    if "add_particle" in sim_params.keys():
        add_particle_params = sim_params["add_particle"]
        if not "start_time" in add_particle_params.keys():
            add_particle_params["start_time"] = 0
        if not "end_time" in add_particle_params.keys():
            add_particle_params["end_time"] = 1e3
        if not "move_pos" in add_particle_params.keys():
            add_particle_params["move_pos"] = [0, 0, 0]
        if not "object_material_index" in add_particle_params.keys():
            add_particle_params["object_material_index"] = 0
        if not "additional" in add_particle_params.keys():
            add_particle_params["additional"] = False
            
    modify_material_params = {}
    if "modify_material" in sim_params.keys():
        modify_material_params = sim_params["modify_material"]
        if not "start_time" in modify_material_params.keys():
            modify_material_params["start_time"] = 0
        if not "object_material_index" in modify_material_params.keys():
            modify_material_params["object_material_index"] = 0
        

    # camera params
    camera_params = {}
    if "mpm_space_viewpoint_center" in sim_params.keys():
        camera_params["mpm_space_viewpoint_center"] = sim_params[
            "mpm_space_viewpoint_center"
        ]
    else:
        camera_params["mpm_space_viewpoint_center"] = [1.0, 1.0, 1.0]
    if "mpm_space_vertical_upward_axis" in sim_params.keys():
        camera_params["mpm_space_vertical_upward_axis"] = sim_params[
            "mpm_space_vertical_upward_axis"
        ]
    else:
        camera_params["mpm_space_vertical_upward_axis"] = [0, 0, 1]
    if "default_camera_index" in sim_params.keys():
        camera_params["default_camera_index"] = sim_params["default_camera_index"]
    else:
        camera_params["default_camera_index"] = 0
    if "show_hint" in sim_params.keys():
        camera_params["show_hint"] = sim_params["show_hint"]
    else:
        camera_params["show_hint"] = False
    if "init_azimuthm" in sim_params.keys():
        camera_params["init_azimuthm"] = sim_params["init_azimuthm"]
    else:
        camera_params["init_azimuthm"] = None
    if "init_elevation" in sim_params.keys():
        camera_params["init_elevation"] = sim_params["init_elevation"]
    else:
        camera_params["init_elevation"] = None
    if "init_radius" in sim_params.keys():
        camera_params["init_radius"] = sim_params["init_radius"]
    else:
        camera_params["init_radius"] = None
    if "delta_a" in sim_params.keys():
        camera_params["delta_a"] = sim_params["delta_a"]
    else:
        camera_params["delta_a"] = None
    if "delta_e" in sim_params.keys():
        camera_params["delta_e"] = sim_params["delta_e"]
    else:
        camera_params["delta_e"] = None
    if "delta_r" in sim_params.keys():
        camera_params["delta_r"] = sim_params["delta_r"]
    else:
        camera_params["delta_r"] = None
    if "move_camera" in sim_params.keys():
        camera_params["move_camera"] = sim_params["move_camera"]
    else:
        camera_params["move_camera"] = False
        
    if "first_move_frame" in sim_params.keys():
        camera_params["first_move_frame"] = sim_params["first_move_frame"]
    else:
        camera_params["first_move_frame"] = 0

    if "multi_camera" in sim_params.keys():
        camera_params["multi_camera"] = sim_params["multi_camera"]
    else:
        camera_params["multi_camera"] = False
    
    if "camera_delta_a" in sim_params.keys():
        camera_params["camera_delta_a"] = sim_params["camera_delta_a"]
    else:
        camera_params["camera_delta_a"] = 0
    
    if "camera_delta_e" in sim_params.keys():
        camera_params["camera_delta_e"] = sim_params["camera_delta_e"]
    else:
        camera_params["camera_delta_e"] = 0

    if "camera_nums" in sim_params.keys():
        camera_params["camera_nums"] = sim_params["camera_nums"]
    else:
        camera_params["camera_nums"] = 1

    return grid_params, materials_params, bc_params, time_params, preprocessings_params, camera_params, add_particle_params, modify_material_params


def set_boundary_conditions(
    mpm_solver: MPM_Simulator_WARP, bc_params: dict, time_params: dict
):
    for bc in bc_params:
        if bc["type"] == "cuboid":
            assert (
                "point" in bc.keys() and "size" in bc.keys() and "velocity" in bc.keys()
            )
            start_time = 0.0
            end_time = 1e3
            reset = 0
            if "start_time" in bc.keys():
                start_time = bc["start_time"]
            if "end_time" in bc.keys():
                end_time = bc["end_time"]
            if "reset" in bc.keys():
                reset = bc["reset"]
            mpm_solver.set_velocity_on_cuboid(
                point=bc["point"],
                size=bc["size"],
                velocity=bc["velocity"],
                start_time=start_time,
                end_time=end_time,
                reset=reset,
            )

        elif bc["type"] == "particle_impulse":
            assert "force" in bc.keys()

            start_time = 0.0
            if "start_time" in bc.keys():
                start_time = bc["start_time"]
            num_dt = 1
            if "num_dt" in bc.keys():
                num_dt = bc["num_dt"]
            point = [1, 1, 1]
            if "point" in bc.keys():
                point = bc["point"]
            size = [1, 1, 1]
            if "size" in bc.keys():
                size = bc["size"]
            end_time = 0.0
            interval=1000.0
            if "interval" in bc.keys():
                interval = bc["interval"]
            if "end_time" in bc.keys():
                end_time = bc["end_time"]
            # breakpoint()
            mpm_solver.add_impulse_on_particles(
                force=bc["force"],
                dt=time_params["substep_dt"],
                point=point,
                size=size,
                num_dt=num_dt,
                start_time=start_time,
                end_time=end_time,
                interval= interval,
            )
        elif bc["type"] == "bounding_box":
            mpm_solver.add_bounding_box()

        elif bc["type"] == "enforce_particle_translation":
            assert "point" in bc.keys()
            assert "size" in bc.keys()
            assert "velocity" in bc.keys()
            assert "start_time" in bc.keys()
            assert "end_time" in bc.keys()

            mpm_solver.enforce_particle_velocity_translation(
                point=bc["point"],
                size=bc["size"],
                velocity=bc["velocity"],
                start_time=bc["start_time"],
                end_time=bc["end_time"],
                object_index = bc["object_index"]
            )
        elif bc["type"] == "surface_collider":
            assert "point" in bc.keys()
            assert "normal" in bc.keys()
            assert "surface" in bc.keys()
            assert "friction" in bc.keys()
            assert "start_time" in bc.keys()
            assert "end_time" in bc.keys()

            mpm_solver.add_surface_collider(
                point=bc["point"],
                normal=bc["normal"],
                surface=bc["surface"],
                friction=bc["friction"],
                start_time=bc["start_time"],
                end_time=bc["end_time"],
            )
        elif bc["type"] == "release_particles_sequentially":
            assert "normal" in bc.keys()
            assert "start_position" in bc.keys()
            assert "end_position" in bc.keys()
            assert "num_layers" in bc.keys()
            assert "start_time" in bc.keys()
            assert "end_time" in bc.keys()
            if "point" not in bc.keys():
                bc["point"] = [0, 0, 0]

            mpm_solver.release_particles_sequentially(
                normal=bc["normal"],
                start_position=bc["start_position"],
                end_position=bc["end_position"],
                num_layers=bc["num_layers"],
                start_time=bc["start_time"],
                end_time=bc["end_time"],
                point=bc["point"]
            )
        elif bc["type"] == "enforce_particle_velocity_rotation":
            assert "normal" in bc.keys()
            assert "point" in bc.keys()
            assert "start_time" in bc.keys()
            assert "end_time" in bc.keys()
            assert "half_height_and_radius" in bc.keys()
            assert "rotation_scale" in bc.keys()
            assert "translation_scale" in bc.keys()
            assert "object_index" in bc.keys()

            mpm_solver.enforce_particle_velocity_rotation(
                point=bc["point"],
                normal=bc["normal"],
                half_height_and_radius=bc["half_height_and_radius"],
                rotation_scale=bc["rotation_scale"],
                translation_scale=bc["translation_scale"],
                object_index=bc["object_index"],
                start_time=bc["start_time"],
                end_time=bc["end_time"],
            )

        elif bc["type"] == "add_plane":
            assert "x_plane" in bc.keys()
            assert "y_plane" in bc.keys()
            assert "z_plane" in bc.keys()
            
            mpm_solver.add_plane(
                x_plane = bc["x_plane"],
                y_plane = bc["y_plane"],
                z_plane = bc["z_plane"]
            )
        else:
            raise TypeError("Undefined BC type")
