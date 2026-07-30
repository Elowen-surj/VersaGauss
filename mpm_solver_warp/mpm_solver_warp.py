import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from engine_utils import *
from warp_utils import *
from mpm_utils import *
import warp as wp


class MPM_Simulator_WARP:
    def __init__(self, n_particles, n_grid=100, grid_lim=1.0, device="cuda:0"):
        self.initialize(n_particles, n_grid, grid_lim, device=device)
        self.time_profile = {}

    def initialize(self, n_particles, n_grid=100, grid_lim=1.0, object_nums=1, device="cuda:0", shs_mix = False):
        # self.n_particles = n_particles
        self.device = device

        self.mpm_model = MPMModelStruct()
        self.mpm_model.object_nums = object_nums
        # domain will be [0,grid_lim]*[0,grid_lim]*[0,grid_lim] !!!
        # domain will be [0,grid_lim]*[0,grid_lim]*[0,grid_lim] !!!
        # domain will be [0,grid_lim]*[0,grid_lim]*[0,grid_lim] !!!
        self.mpm_model.grid_lim = grid_lim
        self.mpm_model.n_grid = n_grid
        self.mpm_model.grid_dim_x = self.mpm_model.n_grid
        self.mpm_model.grid_dim_y = self.mpm_model.n_grid
        self.mpm_model.grid_dim_z = self.mpm_model.n_grid
        (
            self.mpm_model.dx,
            self.mpm_model.inv_dx,
        ) = self.mpm_model.grid_lim / self.mpm_model.n_grid, float(
            self.mpm_model.n_grid / self.mpm_model.grid_lim
        )
        
        # print("************************")
        # print(self.mpm_model.dx)
        # print(self.mpm_model.n_grid)
        # print(self.mpm_model.grid_lim)
        # print("**************************")

        self.mpm_model.E = wp.zeros(shape=self.mpm_model.object_nums, dtype=float, device=device)
        self.mpm_model.nu = wp.zeros(shape=self.mpm_model.object_nums, dtype=float, device=device)
        self.mpm_model.mu = wp.zeros(shape=self.mpm_model.object_nums, dtype=float, device=device)
        self.mpm_model.lam = wp.zeros(shape=self.mpm_model.object_nums, dtype=float, device=device)
        
        self.mpm_model.Ef = wp.zeros(shape=self.mpm_model.object_nums, dtype=float, device=device)

        self.mpm_model.update_cov_with_F = False

        self.mpm_model.shs_mix = shs_mix

        # material is used to switch between different elastoplastic models. 0 is jelly
        self.mpm_model.material = wp.zeros(shape=self.mpm_model.object_nums, dtype=int, device=device)
        

        self.mpm_model.plastic_viscosity = 0.0
        # self.mpm_model.yield_stress = wp.zeros(
        #     shape=n_particles, dtype=float, device=device
        # )
        # self.mpm_model.friction_angle = 25.0
        # sin_phi = wp.sin(self.mpm_model.friction_angle / 180.0 * 3.14159265)
        # self.mpm_model.alpha = wp.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)

        # self.mpm_model.gravitational_accelaration = wp.vec3(0.0, 0.0, 0.0)

        # self.mpm_model.rpic_damping = 0.0  # 0.0 if no damping (apic). -1 if pic

        self.mpm_model.grid_v_damping_scale = 1.1  # globally applied
        # self.mpm_model.plastic_viscosity = wp.zeros(shape=self.mpm_model.object_nums, dtype=float, device=device)
        # self.mpm_model.plastic_viscosity = 0.0
        self.mpm_model.porosity = 0.0
        self.mpm_model.permeability = 0.0
        
        # self.mpm_model.softening = wp.zeros(shape=self.mpm_model.object_nums, dtype=float, device=device)
        # wp.launch(
        #     kernel=set_value_to_float_array,
        #     dim=self.mpm_model.object_nums,
        #     inputs=[self.mpm_model.softening, 0.1],
        #     device=device,
        # )
        
        self.mpm_model.xi = wp.zeros(shape=self.mpm_model.object_nums, dtype=float, device=device)

        self.mpm_model.diff_coe = wp.zeros(shape=self.mpm_model.object_nums, dtype=float, device=device)
        
        self.mpm_model.yield_stress = wp.zeros(shape=self.mpm_model.object_nums, dtype=float, device=device)
        
        self.mpm_model.compression = 2.5e-2
        self.mpm_model.stretch = 7.5e-3
        self.mpm_model.hardening_coe = 10.0

        # self.mpm_model.sand_s = wp.zeros(shape=(4, n_particles), dtype=float, device=device) # 0: phi_s, 1: vc_s, 2: alpha_s, 3: q_s

        # self.mpm_model.vc_s = wp.zeros(shape=n_particles, dtype=float, device=device)

        # self.mpm_model.alpha_s = wp.zeros(shape=n_particles, dtype=float, device=device)

        # self.mpm_model.q_s = wp.zeros(shape=n_particles, dtype=float, device=device)
        
        # self.mpm_model.friction_angle = wp.zeros(shape=self.mpm_model.object_nums, dtype=float, device=device)
        # wp.launch(
        #     kernel=set_value_to_float_array,
        #     dim=self.mpm_model.object_nums,
        #     inputs=[self.mpm_model.friction_angle, 25.0],
        #     device=device,
        # )
        sin_phi = wp.sin(25.0 / 180.0 * 3.14159265)
        alpha_value = wp.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)
        # alpha_value = 0.267765
        # self.mpm_model.alpha = wp.zeros(shape=self.mpm_model.object_nums, dtype=float, device=device)
        self.mpm_model.alpha = alpha_value
        # wp.launch(
        #     kernel=set_value_to_float_subarray,
        #     dim=n_particles,
        #     inputs=[self.mpm_model.sand_s, alpha_value, 2],
        #     device=device,
        # )
        # breakpoint()
        
        self.mpm_model.gravitational_accelaration = wp.vec3(0.0, 0.0, 0.0)
        
        self.mpm_model.rpic_damping = wp.zeros(shape=self.mpm_model.object_nums, dtype=float, device=device)
        
        
        self.mpm_state = MPMStateStruct()

        self.mpm_state.particle_x = wp.empty(
            shape=n_particles, dtype=wp.vec3, device=device
        )  # current position
        
        self.mpm_state.particle_object_link = wp.zeros(
            shape=n_particles, dtype=int, device=device
        )

        self.mpm_state.particle_v = wp.zeros(
            shape=n_particles, dtype=wp.vec3, device=device
        )  # particle velocity

        self.mpm_state.particle_F = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )  # particle F elastic
        
        # self.mpm_state.static = wp.zeros(shape=self.mpm_model.object_nums, dtype=int, device=device)

        # self.mpm_state.particle_R = wp.zeros(
        #     shape=n_particles, dtype=wp.mat33, device=device
        # )  # particle R rotation

        self.mpm_state.particle_init_cov = wp.zeros(
            shape=n_particles * 6, dtype=float, device=device
        )  # initial covariance matrix

        # self.mpm_state.particle_cov = wp.zeros(
        #     shape=n_particles * 6, dtype=float, device=device
        # )  # current covariance matrix

        self.mpm_state.particle_init_shs = wp.zeros(
            shape=n_particles, dtype=wp.vec3, device=device
        )

        # self.mpm_state.particle_F_trial = wp.zeros(
        #     shape=n_particles, dtype=wp.mat33, device=device
        # )  # apply return mapping will yield

        self.mpm_state.particle_stress = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )

        self.mpm_state.particle_vol = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )  # particle volume
        self.mpm_state.particle_mass = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )  # particle mass
        self.mpm_state.particle_density = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )
        self.mpm_state.particle_C = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )
        self.mpm_state.particle_Jp = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )
        self.mpm_state.particle_Jf = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )

        # self.mpm_state.particle_selection = wp.zeros(
        #     shape=n_particles, dtype=int, device=device
        # )
        
        self.mpm_state.grid_m = wp.zeros(
            shape=((self.mpm_model.object_nums + 1), self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=float,
            device=device,
        )
        self.mpm_state.grid_v_in = wp.zeros(
            shape=((2 * self.mpm_model.object_nums + 2), self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=wp.vec3,
            device=device,
        )

        self.mpm_state.grid_v = wp.zeros(
            shape=((self.mpm_model.object_nums + 1), self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=wp.vec3,
            device=device,
        )
        
        self.mpm_state.grid_vol = wp.zeros(
            shape=((self.mpm_model.object_nums + 1), self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=float,
            device=device,
        )
        
        self.mpm_state.grid_data = wp.zeros(
            shape=((2 * self.mpm_model.object_nums + 2), self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=wp.vec3,
            device=device,
        )
        
        self.mpm_state.grid_shs = wp.zeros(shape=(2, self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=wp.vec3,
            device=device,)
        
        # self.mpm_model.energy = 20.0

        # self.mpm_state.energy = wp.zeros(
        #     shape=n_particles, dtype=float, device=device
        # )

        self.time = 0.0

        self.fluid_exist = False

        self.grid_postprocess = []
        self.collider_params = []
        self.modify_bc = []
        self.static_object_params = []
        self.object_grid_process = []
        self.object_particle_process = []

        self.tailored_struct_for_bc = MPMtailoredStruct()
        self.pre_p2g_operations = []
        self.impulse_params = []

        self.particle_velocity_modifiers = []
        self.particle_velocity_modifier_params = []

    # the h5 file should store particle initial position and volume.
    def load_from_sampling(
        self, sampling_h5, n_grid=100, grid_lim=1.0, device="cuda:0"
    ):
        if not os.path.exists(sampling_h5):
            print("h5 file cannot be found at ", os.getcwd() + sampling_h5)
            exit()

        h5file = h5py.File(sampling_h5, "r")
        x, particle_volume = h5file["x"], h5file["particle_volume"]

        x = x[()].transpose()  # np vector of x # shape now is (n_particles, dim)

        self.dim, self.n_particles = x.shape[1], x.shape[0]

        self.initialize(self.n_particles, n_grid, grid_lim, device=device)

        print(
            "Sampling particles are loaded from h5 file. Simulator is re-initialized for the correct n_particles"
        )
        particle_volume = np.squeeze(particle_volume, 0)

        self.mpm_state.particle_x = wp.from_numpy(
            x, dtype=wp.vec3, device=device
        )  # initialize warp array from np

        # initial velocity is default to zero
        wp.launch(
            kernel=set_vec3_to_zero,
            dim=self.n_particles,
            inputs=[self.mpm_state.particle_v],
            device=device,
        )
        # initial velocity is default to zero

        # initial deformation gradient is set to identity
        wp.launch(
            kernel=set_mat33_to_identity,
            dim=self.n_particles,
            inputs=[self.mpm_state.particle_F],
            device=device,
        )
        # initial deformation gradient is set to identity

        self.mpm_state.particle_vol = wp.from_numpy(
            particle_volume, dtype=float, device=device
        )

        print("Particles initialized from sampling file.")
        print("Total particles: ", self.n_particles)

    # shape of tensor_x is (n, 3); shape of tensor_volume is (n,)
    def load_initial_data_from_torch(
        self,
        tensor_x,
        tensor_volume,
        # tensor_material,
        all_n_particles,
        tensor_cov=None,
        tensor_shs=None,
        tensor_link=None,
        n_grid=100,
        grid_lim=1.0,
        object_nums=1,
        shs_mix = False,
        device="cuda:0"
    ):
        self.dim, self.n_particles = tensor_x.shape[1], tensor_x.shape[0]
        assert tensor_x.shape[0] == tensor_volume.shape[0]
        # assert tensor_x.shape[0] == tensor_cov.reshape(-1, 6).shape[0]
        self.initialize(all_n_particles, n_grid, grid_lim, object_nums, device=device, shs_mix = shs_mix)

        self.import_particle_x_from_torch(tensor_x, device)
        # print(tensor_link)
        if tensor_link is not None:
            wp.copy(self.mpm_state.particle_object_link, wp.from_numpy(tensor_link.detach().clone().cpu().numpy(), dtype=int, device=device))
        # print(tensor_link)
        # print(self.mpm_state.particle_object_link)
        wp.copy(self.mpm_state.particle_vol, wp.from_numpy(
            tensor_volume.detach().clone().cpu().numpy(), dtype=float, device=device
        ))

        # wp.copy(self.mpm_model.material, wp.from_numpy(
        #     tensor_material.detach().clone().cpu().numpy(), dtype=int, device=device
        # ))
        
        if tensor_cov is not None:
            wp.copy(self.mpm_state.particle_init_cov, wp.from_numpy(
                tensor_cov.reshape(-1).detach().clone().cpu().numpy(),
                dtype=float,
                device=device,
            ))
            # wp.launch(
            #     kernel = copy_subarray_to_float_array,
            #     dim = tensor_cov.shape[0],
            #     inputs = [self.mpm_state.particle_init_cov, wp.from_numpy(
            #     tensor_cov.reshape(-1).detach().clone().cpu().numpy(), dtype=float, device=device), 0, 0],
            #     device=device,
            # )

            # if self.mpm_model.update_cov_with_F:
            #     self.mpm_state.particle_cov = self.mpm_state.particle_init_cov

        if tensor_shs is not None:
            wp.copy(self.mpm_state.particle_init_shs, wp.from_numpy(
                tensor_shs.squeeze(dim=1).detach().clone().cpu().numpy(),
                dtype=float,
                device=device,
            ))
            # wp.launch(
            #     kernel = copy_subarray_to_vec3_array,
            #     dim = tensor_cov.shape[0],
            #     inputs = [self.mpm_state.particle_init_shs, wp.from_numpy(
            #     tensor_shs.squeeze(dim=1).detach().clone().cpu().numpy(), dtype=wp.vec3, device=device), 0, 0],
            #     device=device,
            # )
        # breakpoint()

        # initial velocity is default to zero
        wp.launch(
            kernel=set_vec3_to_zero,
            dim=all_n_particles,
            inputs=[self.mpm_state.particle_v],
            device=device,
        )

        wp.launch(
            kernel=set_value_to_float_array,
            dim=all_n_particles,
            inputs=[self.mpm_state.particle_Jp, 1.0],
            device=device,
        )

        wp.launch(
            kernel=set_value_to_float_array,
            dim=all_n_particles,
            inputs=[self.mpm_state.particle_Jf, 1.0],
            device=device,
        )
        # initial velocity is default to zero

        # initial deformation gradient is set to identity
        wp.launch(
            kernel=set_mat33_to_identity,
            dim=all_n_particles,
            inputs=[self.mpm_state.particle_F],
            device=device,
        )
        # initial trial deformation gradient is set to identity

        print("Particles initialized from torch data.")
        print("Total particles: ", self.n_particles)
            # breakpoint()
            
    def modify_particle_material(self, material_params):
        material = self.mpm_model.material.numpy()
        E_numpy = self.mpm_model.E.numpy()
        nu = self.mpm_model.nu.numpy()
        yield_stress = self.mpm_model.yield_stress.numpy()
        bulk_modulus = self.mpm_model.Ef.numpy()
        mi = material_params["object_material_index"]
        if "material" in material_params:
            if material_params["material"] == "jelly":
                material[mi] = 0
            elif material_params["material"] == "metal":
                material[mi] = 1
            elif material_params["material"] == "sand":
                material[mi] = 2
            elif material_params["material"] == "foam":
                material[mi] = 3
            elif material_params["material"] == "snow":
                material[mi] = 4
            elif material_params["material"] == "plasticine":
                material[mi] = 5
            elif material_params["material"] == "fluid":
                material[mi] = 6
                self.fluid_exist = True
            elif material_params["material"] == "jam":
                material[mi] = 7
            else:
                raise TypeError("Undefined material type")
            
        if "E" in material_params:
            E_numpy[mi] = material_params["E"]

        if "nu" in material_params:
            nu[mi] = material_params["nu"]
            
        if "yield_stress" in material_params:
            yield_stress[mi] = material_params["yield_stress"]

        if "bulk_modulus" in material_params:
            bulk_modulus[mi] = material_params["bulk_modulus"]
            
        self.mpm_model.material = wp.from_numpy(material, dtype=int, device=self.device)
        
        self.mpm_model.E = wp.from_numpy(
            E_numpy, dtype=float, device=self.device
        )

        self.mpm_model.nu = wp.from_numpy(
            nu, dtype=float, device=self.device
        )

        self.mpm_model.yield_stress = wp.from_numpy(
            yield_stress, dtype=float, device=self.device
        )
        
        self.mpm_model.Ef = wp.from_numpy(
            bulk_modulus, dtype=float, device=self.device
        )
        
    def modify_cov_shs(self, tensor_shs, tensor_cov, nums1):
        particle_cov = wp.from_numpy(tensor_cov.reshape(-1).detach().clone().cpu().numpy(), dtype = float, device = self.device)
        wp.copy(self.mpm_state.particle_init_cov, particle_cov, dest_offset = nums1 * 6)
        particle_shs = wp.from_numpy(tensor_shs.squeeze(dim=1).detach().clone().cpu().numpy(), dtype = float, device = self.device)
        wp.copy(self.mpm_state.particle_init_shs, particle_shs, dest_offset = nums1)
        
        
    def add_particle(self, add_tensor_x, add_tensor_vol, 
                    #  add_tensor_material, 
                     add_tensor_cov = None, add_tensor_shs = None,
                     add_tensor_link = None, add_material_params = {}, device = "cuda:0"):
        add_particle_nums = add_tensor_x.shape[0]
        
        add_tensor_x = add_tensor_x.clone().detach()
        add_particle_x = torch2warp_vec3(add_tensor_x, dvc=device)
        # breakpoint()
        wp.copy(self.mpm_state.particle_x, add_particle_x, dest_offset=self.n_particles)
        
        add_particle_vol = wp.from_numpy(add_tensor_vol.detach().clone().cpu().numpy(), dtype= float, device=device)
        wp.copy(self.mpm_state.particle_vol, add_particle_vol, dest_offset= self.n_particles)
        
        # add_particle_material = wp.from_numpy(add_tensor_material.detach().clone().cpu().numpy(), dtype = float, device = device)
        # wp.copy(self.mpm_model.material, add_particle_material, dest_offset = self.n_particles)
        
        add_particle_cov = wp.from_numpy(add_tensor_cov.reshape(-1).detach().clone().cpu().numpy(), dtype = float, device = device)
        wp.copy(self.mpm_state.particle_init_cov, add_particle_cov, dest_offset = self.n_particles * 6)
        # wp.launch(
        #     kernel = copy_subarray_to_float_array,
        #     dim = add_tensor_cov.shape[0],
        #     inputs = [self.mpm_state.particle_init_cov, add_particle_cov, self.n_particles, 0],
        #     device=device,
        # )

        add_particle_shs = wp.from_numpy(add_tensor_shs.squeeze(dim=1).detach().clone().cpu().numpy(), dtype = float, device = device)
        wp.copy(self.mpm_state.particle_init_shs, add_particle_shs, dest_offset = self.n_particles)
        # wp.launch(
        #     kernel = copy_subarray_to_vec3_array,
        #     dim = add_tensor_shs.shape[0],
        #     inputs = [self.mpm_state.particle_init_shs, add_particle_shs, self.n_particles, 0],
        #     device=device,
        # )
        
        add_particle_link = wp.from_numpy(add_tensor_link.detach().clone().cpu().numpy(), dtype = int, device = device)
        wp.copy(self.mpm_state.particle_object_link, add_particle_link, dest_offset = self.n_particles)
        
        # if "E" in add_material_params:
        #     wp.launch(
        #             kernel=set_value_to_float_array_range,
        #             dim=add_particle_nums,
        #             inputs=[self.mpm_model.E, add_material_params["E"], self.n_particles],
        #             device=device,
        #         )
        # if "nu" in add_material_params:
        #     wp.launch(
        #             kernel=set_value_to_float_array_range,
        #             dim=add_particle_nums,
        #             inputs=[self.mpm_model.nu, add_material_params["nu"], self.n_particles],
        #             device=device,
        #         )
        # if "yield_stress" in add_material_params:
        #     wp.launch(
        #             kernel=set_value_to_float_array_range,
        #             dim=add_particle_nums,
        #             inputs=[self.mpm_model.yield_stress, add_material_params["yield_stress"], self.n_particles],
        #             device=device,
        #         )
        # if "bulk_modulus" in add_material_params:
        #     wp.launch(
        #             kernel=set_value_to_float_array_range,
        #             dim=add_particle_nums,
        #             inputs=[self.mpm_model.Ef, add_material_params["bulk_modulus"], self.n_particles],
        #             device=device,
        #         )
        if "density" in add_material_params:
            wp.launch(
                kernel=set_value_to_float_array_range,
                dim=add_particle_nums,
                inputs=[self.mpm_state.particle_density, add_material_params["density"], self.n_particles],
                device=device,
            )
            
        self.n_particles = self.n_particles + add_particle_nums
        wp.launch(
            kernel=get_float_array_product,
            dim=self.n_particles,
            inputs=[
                self.mpm_state.particle_density,
                self.mpm_state.particle_vol,
                self.mpm_state.particle_mass,
            ],
            device=device,
        )
        # import ipdb; ipdb.set_trace()
        
    # must give density. mass will be updated as density * volume
    def set_parameters_dict(self, grid_params = {}, materials_params={}, fill_gs_nums = [], device="cuda:0"):

        # with wp.ScopedMempoolAccess("cuda:0", "cuda:1", True):
        if "grid_lim" in grid_params:
            self.mpm_model.grid_lim = grid_params["grid_lim"]
        if "n_grid" in grid_params:
            self.mpm_model.n_grid = grid_params["n_grid"]
        self.mpm_model.grid_dim_x = self.mpm_model.n_grid
        self.mpm_model.grid_dim_y = self.mpm_model.n_grid
        self.mpm_model.grid_dim_z = self.mpm_model.n_grid
        (
            self.mpm_model.dx,
            self.mpm_model.inv_dx,
        ) = self.mpm_model.grid_lim / self.mpm_model.n_grid, float(
            self.mpm_model.n_grid / self.mpm_model.grid_lim
        )
        self.mpm_state.grid_m = wp.zeros(
            shape=((self.mpm_model.object_nums + 1), self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=float,
            device=device,
        )
        # self.mpm_state.grid_v_in = wp.zeros(
        #     shape=(2, self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
        #     dtype=wp.vec3,
        #     device=device,
        # )
        self.mpm_state.grid_v = wp.zeros(
            shape=((self.mpm_model.object_nums + 1), self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=wp.vec3,
            device=device,
        )
        
        self.mpm_state.grid_vol = wp.zeros(
            shape=((self.mpm_model.object_nums + 1), self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=float,
            device=device,
        )
        
        self.mpm_state.grid_data = wp.zeros(
            shape=((2 * self.mpm_model.object_nums + 2), self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=wp.vec3,
            device=device,
        )
        
        if "g" in grid_params:
            self.mpm_model.gravitational_accelaration = wp.vec3(grid_params["g"][0], grid_params["g"][1], grid_params["g"][2])
            
        if "grid_v_damping_scale" in grid_params:
            self.mpm_model.grid_v_damping_scale = grid_params["grid_v_damping_scale"]

        # hardening = self.mpm_model.hardening.numpy()
        xi = self.mpm_model.xi.numpy()
        diff_coe = self.mpm_model.diff_coe.numpy()
        # friction_angle = self.mpm_model.friction_angle.numpy()
        # alpha = self.mpm_model.alpha.numpy()
        rpic_damping = self.mpm_model.rpic_damping.numpy()
        # plastic_viscosity = self.mpm_model.plastic_viscosity.numpy()
        # softening = self.mpm_model.softening.numpy()
        material = self.mpm_model.material.numpy()
        E_numpy = self.mpm_model.E.numpy()
        nu = self.mpm_model.nu.numpy()
        yield_stress = self.mpm_model.yield_stress.numpy()
        bulk_modulus = self.mpm_model.Ef.numpy()
        
        # static = self.mpm_state.static.numpy()
        
        for mi in range(self.mpm_model.object_nums):
            material_params = materials_params[mi]

            if "material" in material_params:
                if material_params["material"] == "jelly":
                    material[mi] = 0
                elif material_params["material"] == "metal":
                    material[mi] = 1
                elif material_params["material"] == "sand":
                    material[mi] = 2
                elif material_params["material"] == "foam":
                    material[mi] = 3
                elif material_params["material"] == "snow":
                    material[mi] = 4
                elif material_params["material"] == "plasticine":
                    material[mi] = 5
                elif material_params["material"] == "fluid":
                    material[mi] = 6
                    self.fluid_exist = True
                elif material_params["material"] == "jam":
                    material[mi] = 7

                else:
                    raise TypeError("Undefined material type")

            # if "E" in material_params:
            #     wp.launch(
            #         kernel=set_value_to_float_array_arange,
            #         dim=self.n_particles,
            #         inputs=[self.mpm_model.E, material_params["E"], fill_gs_nums[mi], fill_gs_nums[mi+1]],
            #         device=device,
            #     )
            # if "nu" in material_params:
            #     wp.launch(
            #         kernel=set_value_to_float_array_arange,
            #         dim=self.n_particles,
            #         inputs=[self.mpm_model.nu, material_params["nu"], fill_gs_nums[mi], fill_gs_nums[mi+1]],
            #         device=device,
            #     )
            # if "yield_stress" in material_params:
            #     val = material_params["yield_stress"]
            #     wp.launch(
            #         kernel=set_value_to_float_array_arange,
            #         dim=self.n_particles,
            #         inputs=[self.mpm_model.yield_stress, val, fill_gs_nums[mi], fill_gs_nums[mi+1]],
            #         device=device,
            #     )
            
            # if "bulk_modulus" in material_params:
            #     wp.launch(
            #         kernel=set_value_to_float_array_arange,
            #         dim=self.n_particles,
            #         inputs=[self.mpm_model.Ef, material_params["bulk_modulus"], fill_gs_nums[mi], fill_gs_nums[mi+1]],
            #         device=device
            #     )
                
            if "density" in material_params:
                density_value = material_params["density"]
                wp.launch(
                    kernel=set_value_to_float_array_arange,
                    dim=self.n_particles,
                    inputs=[self.mpm_state.particle_density, density_value, fill_gs_nums[mi], fill_gs_nums[mi+1]],
                    device=device,
                )
                wp.launch(
                    kernel=get_float_array_product,
                    dim=self.n_particles,
                    inputs=[
                        self.mpm_state.particle_density,
                        self.mpm_state.particle_vol,
                        self.mpm_state.particle_mass,
                    ],
                    device=device,
                )
                
            if "hardening" in material_params:
                self.mpm_model.hardening = material_params["hardening"]

            if "porosity" in material_params:
                self.mpm_model.porosity = material_params["porosity"]

            if "permeability" in material_params:
                self.mpm_model.permeability = material_params["permeability"]
                
            if "compression" in material_params:
                self.mpm_model.compression = material_params["compression"]
                
            if "stretch" in material_params:
                self.mpm_model.stretch = material_params["stretch"]
                
            if "gamma" in material_params:
                self.mpm_model.gamma = material_params["gamma"]
                
            if "hardening_coe" in material_params:
                self.mpm_model.hardening_coe = material_params["hardening_coe"]

            if "diff_coe" in material_params:
                diff_coe[mi] = material_params["diff_coe"]
            if "xi" in material_params:
                xi[mi] = material_params["xi"]
            if "friction_angle" in material_params:
                friction_angle = material_params["friction_angle"]
                sin_phi = wp.sin(friction_angle / 180.0 * 3.14159265)
                alpha_value = wp.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)
                self.mpm_model.alpha = alpha_value
                # wp.launch(
                #     kernel=set_value_to_float_subarray_arange,
                #     dim=self.n_particles,
                #     inputs=[self.mpm_model.sand_s, alpha_value, fill_gs_nums[mi], fill_gs_nums[mi+1], 2],
                #     device=device
                # )

            if "rpic_damping" in material_params:
                rpic_damping[mi] = material_params["rpic_damping"]
                # set_value_index_array(self.mpm_model.rpic_damping, mi, material_params["rpic_damping"])
            if "plastic_viscosity" in material_params:
                self.mpm_model.plastic_viscosity = material_params["plastic_viscosity"]
            # if "softening" in material_params:
            #     softening[mi] = material_params["softening"]
            if "E" in material_params:
                E_numpy[mi] = material_params["E"]

            if "nu" in material_params:
                nu[mi] = material_params["nu"]
                
            if "static" in material_params and material_params["static"]:
                self.object_static(object_index=mi, static_inf = material_params["static_inf"])
            #     # self.enforce_object_particle_velocity_translation(
            #     #     object_index=mi,
            #     #     start_time=0,
            #     #     end_time=1e3,
            #     # )
            #     static[mi] = 1

            if "yield_stress" in material_params:
                yield_stress[mi] = material_params["yield_stress"]

            if "bulk_modulus" in material_params:
                bulk_modulus[mi] = material_params["bulk_modulus"]

            if "additional_material_params" in material_params:
                for params in material_params["additional_material_params"]:
                    param_modifier = MaterialParamsModifier()
                    param_modifier.point = wp.vec3(params["point"])
                    param_modifier.size = wp.vec3(params["size"])
                    param_modifier.density = params["density"]
                    param_modifier.E = params["E"]
                    param_modifier.nu = params["nu"]
                    wp.launch(
                        kernel=apply_additional_params,
                        dim=self.n_particles,
                        inputs=[self.mpm_state, self.mpm_model, param_modifier],
                        device=device,
                    )
        # self.mpm_model.hardening = wp.from_numpy(
        #     hardening, dtype=int, device=device
        # )
        self.mpm_model.material = wp.from_numpy(material, dtype=int, device=device)
        
        # self.mpm_state.static = wp.from_numpy(static, dtype = int, device = device)

        self.mpm_model.xi = wp.from_numpy(
            xi, dtype=float, device=device
        )

        self.mpm_model.diff_coe = wp.from_numpy(
            diff_coe, dtype=float, device=device
        )
        # self.mpm_model.friction_angle = wp.from_numpy(
        #     friction_angle, dtype=float, device=device
        # )
        # self.mpm_model.alpha = wp.from_numpy(
        #     alpha, dtype=float, device=device
        # )
        self.mpm_model.rpic_damping = wp.from_numpy(
            rpic_damping, dtype=float, device=device
        )

        self.mpm_model.Ef = wp.from_numpy(
            bulk_modulus, dtype=float, device=device
        )

        self.mpm_model.E = wp.from_numpy(
            E_numpy, dtype=float, device=device
        )

        self.mpm_model.nu = wp.from_numpy(
            nu, dtype=float, device=device
        )

        self.mpm_model.yield_stress = wp.from_numpy(
            yield_stress, dtype=float, device=device
        )
        # self.mpm_model.plastic_viscosity = wp.from_numpy(
        #     plastic_viscosity, dtype=float, device=device
        # )
        # self.mpm_model.softening = wp.from_numpy(
        #     softening, dtype=float, device=device
        # )
        # print(self.mpm_model.hardening)
        # print(self.mpm_model.xi)
        # # print(self.mpm_model.friction_angle)
        # print(self.mpm_model.alpha)
        # print(self.mpm_model.rpic_damping)
        # print(self.mpm_model.plastic_viscosity)
        # print(self.mpm_model.softening)
        # print(self.mpm_state.particle_object_link)
        wp.launch(
            kernel=get_float_array_product,
            dim=self.n_particles,
            inputs=[
                self.mpm_state.particle_density,
                self.mpm_state.particle_vol,
                self.mpm_state.particle_mass,
            ],
            device=device,
        )

    
    def finalize_mu_lam(self, device="cuda:0"):
        wp.launch(
            kernel=compute_mu_lam_from_E_nu,
            dim=self.mpm_model.object_nums,
            inputs=[self.mpm_state, self.mpm_model],
            device=device,
        )

    def p2g2p(self, step, dt, device="cuda"):
        # breakpoint()
        # devices = wp.get_cuda_devices()
        # device_count = len(devices)
        grid_size = (
            self.mpm_model.grid_dim_x,
            self.mpm_model.grid_dim_y,
            self.mpm_model.grid_dim_z,
        )
        k_grid_size = (
            self.mpm_model.object_nums + 1,
            self.mpm_model.grid_dim_x,
            self.mpm_model.grid_dim_y,
            self.mpm_model.grid_dim_z
        )   # multi phase grid size
        wp.launch(
            kernel=zero_grid,
            dim=(grid_size),
            inputs=[self.mpm_state, self.mpm_model],
            device=device,
        )
        # apply pre-p2g operations on particles
        for k in range(len(self.pre_p2g_operations)):
            wp.launch(
                kernel=self.pre_p2g_operations[k],
                dim=self.n_particles,
                inputs=[self.time, dt, self.mpm_state, self.impulse_params[k]],
                device=device,
            )
        # apply dirichlet particle v modifier
        for k in range(len(self.particle_velocity_modifiers)):
            wp.launch(
                kernel=self.particle_velocity_modifiers[k],
                dim=self.n_particles,
                inputs=[
                    self.time,
                    self.mpm_state,
                    self.particle_velocity_modifier_params[k],
                ],
                device=device,
            )

        # compute stress = stress(returnMap(F_trial))
        with wp.ScopedTimer(
            "compute_stress_from_F_trial",
            synchronize=True,
            print=False,
            dict=self.time_profile,
        ):
            fluid_type = False
            wp.launch(
                kernel=compute_stress_from_F_trial,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model, dt, fluid_type],
                device=device,
            )  # F and stress are updated

        # p2g
        # pre_count = 0
        # sample_n_particles = int(self.n_particles / device_count)
        # for iter, sample_device in enumerate(devices):
        #     # use a ScopedDevice to set the target device
        # with wp.ScopedDevice(sample_device):
        with wp.ScopedTimer(
            "p2g",
            synchronize=True,
            print=False,
            dict=self.time_profile,
        ):
            wp.launch(
                kernel=p2g_apic_with_stress,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model, dt],
                device=device,
            )  # apply p2g'
        
        for k in range(len(self.object_grid_process)):
            wp.launch(
                kernel=self.object_grid_process[k],
                dim=(grid_size),
                inputs=[
                    self.time,
                    self.mpm_state,
                    self.mpm_model,
                    self.static_object_params[k],
                ],
                device=device,
            )
            
        # with wp.ScopedTimer(
        #     "p2g2",
        #     synchronize=True,
        #     print=False,
        #     dict=self.time_profile,
        # ):
        #     stiffness = 1e-3
        #     rest_density = 7.5e-3
        #     dynamic_viscosity = 0
        #     gamma = 8.0
        #     # stiffness = 3.0
        #     # rest_density = 4e-5
        #     # dynamic_viscosity = 1e2
        #     # gamma = 5.0
        #     # breakpoint()
        #     wp.launch(
        #         kernel=p2g_water_2,
        #         dim=self.n_particles,
        #         inputs=[self.mpm_state, self.mpm_model, dt, stiffness, rest_density, dynamic_viscosity, gamma],
        #         device=device,
        #     )  # apply p2g'
        # breakpoint()
        with wp.ScopedTimer(
            "grid_sum", 
            synchronize=True, 
            print=False, dict=self.time_profile
        ):
            wp.launch(
                kernel=grid_objects_sum,
                dim=(grid_size),
                inputs=[self.mpm_state, self.mpm_model],
                device=device,
            )
        
        # breakpoint()
        # grid update
        with wp.ScopedTimer(
            "grid_update", 
            synchronize=True, 
            print=False, dict=self.time_profile
        ):
            eta = float(1e-7)
            wp.launch(
                kernel=grid_normalization_and_gravity,
                dim=(grid_size),
                inputs=[self.mpm_state, self.mpm_model, dt, eta],
                device=device,
            )

        judge = wp.zeros(shape=self.mpm_model.object_nums, dtype=int, device=device)
        # theta = 1.0
        wp.launch(
            kernel=grid_diffusion,
            dim=(grid_size),
            inputs=[self.mpm_state, self.mpm_model, judge],
            device=device,
        )

        wp.launch(
            kernel = sand_water_mixture,
            dim = (grid_size),
            inputs = [self.mpm_state, self.mpm_model, dt],
            device = device
        )
        wp.launch(
            kernel = solid_fluid_inter,
            dim = (grid_size),
            inputs = [self.mpm_state, self.mpm_model, 0.005],
            device = device
        )
        # breakpoint()

        if self.mpm_model.grid_v_damping_scale < 1.0:
            wp.launch(
                kernel=add_damping_via_grid,
                dim=(k_grid_size),
                inputs=[self.mpm_state, self.mpm_model.grid_v_damping_scale],
                device=device,
            )
        # breakpoint()
        # apply BC on grid
        with wp.ScopedTimer(
            "apply_BC_on_grid", 
            synchronize=True, 
            print=False, dict=self.time_profile
        ):
            # breakpoint()
            for k in range(len(self.grid_postprocess)):
                wp.launch(
                    kernel=self.grid_postprocess[k],
                    dim=k_grid_size,
                    inputs=[
                        self.time,
                        dt,
                        self.mpm_state,
                        self.mpm_model,
                        self.collider_params[k],
                    ],
                    device=device,
                )
                if self.modify_bc[k] is not None:
                    self.modify_bc[k](self.time, dt, self.collider_params[k])
            # breakpoint()
            
        for k in range(len(self.object_grid_process)):
            wp.launch(
                kernel=self.object_grid_process[k],
                dim=(grid_size),
                inputs=[
                    self.time,
                    self.mpm_state,
                    self.mpm_model,
                    self.static_object_params[k],
                ],
                device=device,
            )
            
        for k in range(len(self.object_particle_process)):
            wp.launch(
                kernel=self.object_particle_process[k],
                dim=self.n_particles,
                inputs=[
                    self.time,
                    self.mpm_state,
                    self.static_object_params[k],
                    dt
                ],
                device=device,
            )
            
        # g2p
        # breakpoint()
        with wp.ScopedTimer(
            "g2p", 
            synchronize=True, 
            print=False, dict=self.time_profile
        ):
            wp.launch(
                kernel=g2p,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model, dt, self.n_particles],
                device=device,
            )  # x, v, C, F_trial are updated
        
        # with wp.ScopedTimer(
        #     "compute_energy",
        #     synchronize=True,
        #     print=False,
        #     dict=self.time_profile,
        # ):
        #     wp.launch(
        #         kernel=compute_energy,
        #         dim=self.n_particles,
        #         inputs=[self.mpm_state, self.mpm_model],
        #         device=device,
        #     )
        # print('%.6f' % self.mpm_state.energy[0])
        # breakpoint()
        #### CFL check ####
        # particle_v = self.mpm_state.particle_v.numpy()
        # if np.max(np.abs(particle_v)) > self.mpm_model.dx / dt:
        #     print("max particle v: ", np.max(np.abs(particle_v)))
        #     print("max allowed  v: ", self.mpm_model.dx / dt)
        #     print("does not allow v*dt>dx")
        #     input()
        #### CFL check ####
        self.time = self.time + dt

    # set particle densities to all_particle_densities,
    def reset_densities_and_update_masses(
        self, all_particle_densities, device="cuda:0"
    ):
        all_particle_densities = all_particle_densities.clone().detach()
        self.mpm_state.particle_density = torch2warp_float(
            all_particle_densities, dvc=device
        )
        wp.launch(
            kernel=get_float_array_product,
            dim=self.n_particles,
            inputs=[
                self.mpm_state.particle_density,
                self.mpm_state.particle_vol,
                self.mpm_state.particle_mass,
            ],
            device=device,
        )

    # clone = True makes a copy, not necessarily needed
    def import_particle_x_from_torch(self, tensor_x, clone=True, device="cuda:0"):
        if tensor_x is not None:
            if clone:
                tensor_x = tensor_x.clone().detach()
            init_particle_x = torch2warp_vec3(tensor_x, dvc=device)
            warp.copy(self.mpm_state.particle_x, init_particle_x)
            
    def import_link_from_torch(self, tensor_link, clone = True, device='cuda:0'):
        if tensor_link is not None:
            if clone:
                tensor_link = tensor_link.clone().detach()
            self.mpm_state.particle_object_link = torch2warp_int(tensor_link, dvc=device)

    # clone = True makes a copy, not necessarily needed
    def import_particle_v_from_torch(self, tensor_v, clone=True, device="cuda:0"):
        if tensor_v is not None:
            if clone:
                tensor_v = tensor_v.clone().detach()
            self.mpm_state.particle_v = torch2warp_vec3(tensor_v, dvc=device)

    # clone = True makes a copy, not necessarily needed
    def import_particle_F_from_torch(self, tensor_F, clone=True, device="cuda:0"):
        if tensor_F is not None:
            if clone:
                tensor_F = tensor_F.clone().detach()
            tensor_F = torch.reshape(tensor_F, (-1, 3, 3))  # arranged by rowmajor
            self.mpm_state.particle_F = torch2warp_mat33(tensor_F, dvc=device)

    # clone = True makes a copy, not necessarily needed
    def import_particle_C_from_torch(self, tensor_C, clone=True, device="cuda:0"):
        if tensor_C is not None:
            if clone:
                tensor_C = tensor_C.clone().detach()
            tensor_C = torch.reshape(tensor_C, (-1, 3, 3))  # arranged by rowmajor
            self.mpm_state.particle_C = torch2warp_mat33(tensor_C, dvc=device)

    # def export_energy_to_torch(self, device = 'cuda:0'):
    #     # self.mpm_model.energy = 0.0
    #     # self.mpm_state.energy = wp.zeros(
    #     #     shape=self.n_particles, dtype=float, device=device
    #     # )

    #     # with wp.ScopedTimer(
    #     #     "compute_stress_from_F_trial",
    #     #     synchronize=True,
    #     #     print=False,
    #     #     dict=self.time_profile,
    #     # ):
    #     wp.launch(
    #         kernel = compute_energy,
    #         dim = self.n_particles,
    #         inputs = [self.mpm_state, self.mpm_model],
    #         device = device,
    #     )
    #     # print('%.6f' % self.mpm_model.energy)
    #     # return self.mpm_model.energy
    #     return wp.to_torch(self.mpm_state.energy)
    
    def export_particle_x_to_torch(self):
        return wp.to_torch(self.mpm_state.particle_x)

    def export_particle_v_to_torch(self):
        return wp.to_torch(self.mpm_state.particle_v)
    
    def export_particle_shs_to_torch(self, device = 'cuda:0'):
        if self.mpm_model.shs_mix and self.fluid_exist:
            # breakpoint()
            particle_shs = wp.zeros(shape=self.n_particles, dtype=wp.vec3, device=device)
            wp.launch(
                kernel=compute_shs_from_grid,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model, particle_shs],
                device=device,
            )
            return wp.to_torch(particle_shs)
        else:
            return wp.to_torch(self.mpm_state.particle_init_shs)

        # breakpoint()

    def export_particle_F_to_torch(self):
        F_tensor = wp.to_torch(self.mpm_state.particle_F)
        F_tensor = F_tensor.reshape(-1, 9)
        return F_tensor

    # def export_particle_R_to_torch(self, device="cuda:0"):
    #     with wp.ScopedTimer(
    #         "compute_R_from_F",
    #         synchronize=True,
    #         print=False,
    #         dict=self.time_profile,
    #     ):
    #         wp.launch(
    #             kernel=compute_R_from_F,
    #             dim=self.n_particles,
    #             inputs=[self.mpm_state, self.mpm_model],
    #             device=device,
    #         )

    #     R_tensor = wp.to_torch(self.mpm_state.particle_R)
    #     R_tensor = R_tensor.reshape(-1, 9)
    #     return R_tensor

    def export_particle_C_to_torch(self):
        C_tensor = wp.to_torch(self.mpm_state.particle_C)
        C_tensor = C_tensor.reshape(-1, 9)
        return C_tensor

    def export_particle_cov_to_torch(self, device="cuda:0"):
        # breakpoint()
        particle_cov = wp.zeros(shape=self.n_particles * 6, dtype=float, device=device)
        wp.launch(kernel=compute_cov_from_F, dim=self.n_particles, inputs=[self.mpm_state, self.mpm_model, particle_cov],device=device)
        # breakpoint()
        # cov = wp.to_torch(self.mpm_state.particle_init_cov)
        cov = wp.to_torch(particle_cov)
        return cov

    def print_time_profile(self):
        print("MPM Time profile:")
        for key, value in self.time_profile.items():
            print(key, sum(value))

    # a surface specified by a point and the normal vector
    def add_surface_collider(
        self,
        point,
        normal,
        surface="sticky",
        friction=0.0,
        start_time=0.0,
        end_time=999.0,
    ):
        point = list(point)
        # Normalize normal
        normal_scale = 1.0 / wp.sqrt(float(sum(x**2 for x in normal)))
        normal = list(normal_scale * x for x in normal)

        collider_param = Dirichlet_collider()
        collider_param.start_time = start_time
        collider_param.end_time = end_time

        collider_param.point = wp.vec3(point[0], point[1], point[2])
        collider_param.normal = wp.vec3(normal[0], normal[1], normal[2])

        if surface == "sticky" and friction != 0:
            raise ValueError("friction must be 0 on sticky surfaces.")
        if surface == "sticky":
            collider_param.surface_type = 0
        elif surface == "slip":
            collider_param.surface_type = 1
        elif surface == "cut":
            collider_param.surface_type = 11
        else:
            collider_param.surface_type = 2
        # frictional
        collider_param.friction = friction

        self.collider_params.append(collider_param)

        @wp.kernel
        def collide(
            time: float,
            dt: float,
            state: MPMStateStruct,
            model: MPMModelStruct,
            param: Dirichlet_collider,
        ):
            k, grid_x, grid_y, grid_z = wp.tid()
            if time >= param.start_time and time < param.end_time:
                offset = wp.vec3(
                    float(grid_x) * model.dx - param.point[0],
                    float(grid_y) * model.dx - param.point[1],
                    float(grid_z) * model.dx - param.point[2],
                )
                n = wp.vec3(param.normal[0], param.normal[1], param.normal[2])
                dotproduct = wp.dot(offset, n)

                if dotproduct < 0.0:
                    if param.surface_type == 0:
                        state.grid_v[k, grid_x, grid_y, grid_z] = wp.vec3(
                            0.0, 0.0, 0.0
                        )
                    elif param.surface_type == 11:
                        if (
                            float(grid_z) * model.dx < 0.4
                            or float(grid_z) * model.dx > 0.53
                        ):
                            state.grid_v[k, grid_x, grid_y, grid_z] = wp.vec3(
                                0.0, 0.0, 0.0
                            )
                        else:
                            v_in = state.grid_v[k, grid_x, grid_y, grid_z]
                            state.grid_v[k, grid_x, grid_y, grid_z] = (
                                wp.vec3(v_in[0], 0.0, v_in[2]) * 0.3
                            )
                    else:
                        v = state.grid_v[k, grid_x, grid_y, grid_z]
                        normal_component = wp.dot(v, n)
                        if param.surface_type == 1:
                            v = (
                                v - normal_component * n
                            )  # Project out all normal component
                        else:
                            v = (
                                v - wp.min(normal_component, 0.0) * n
                            )  # Project out only inward normal component
                        if normal_component < 0.0 and wp.length(v) > 1e-20:
                            v = wp.max(
                                0.0, wp.length(v) + normal_component * param.friction
                            ) * wp.normalize(
                                v
                            )  # apply friction here
                        state.grid_v[k, grid_x, grid_y, grid_z] = wp.vec3(
                            0.0, 0.0, 0.0
                        )

        self.grid_postprocess.append(collide)
        self.modify_bc.append(None)

    # a cubiod is a rectangular cube'
    # centered at `point`
    # dimension is x: point[0]±size[0]
    #              y: point[1]±size[1]
    #              z: point[2]±size[2]
    # all grid nodes lie within the cubiod will have their speed set to velocity
    # the cuboid itself is also moving with const speed = velocity
    # set the speed to zero to fix BC
    def set_velocity_on_cuboid(
        self,
        point,
        size,
        velocity,
        start_time=0.0,
        end_time=999.0,
        reset=0,
    ):
        point = list(point)

        collider_param = Dirichlet_collider()
        collider_param.start_time = start_time
        collider_param.end_time = end_time
        collider_param.point = wp.vec3(point[0], point[1], point[2])
        collider_param.size = size
        collider_param.velocity = wp.vec3(velocity[0], velocity[1], velocity[2])
        # collider_param.threshold = threshold
        collider_param.reset = reset
        self.collider_params.append(collider_param)

        @wp.kernel
        def collide(
            time: float,
            dt: float,
            state: MPMStateStruct,
            model: MPMModelStruct,
            param: Dirichlet_collider,
        ):
            k, grid_x, grid_y, grid_z = wp.tid()
            if time >= param.start_time and time < param.end_time:
                offset = wp.vec3(
                    float(grid_x) * model.dx - param.point[0],
                    float(grid_y) * model.dx - param.point[1],
                    float(grid_z) * model.dx - param.point[2],
                )
                if (
                    wp.abs(offset[0]) < param.size[0]
                    and wp.abs(offset[1]) < param.size[1]
                    and wp.abs(offset[2]) < param.size[2]
                ):
                    state.grid_v[k, grid_x, grid_y, grid_z] = param.velocity
                    # print('yes')
            elif param.reset == 1:
                if time < param.end_time + 15.0 * dt:
                    state.grid_v[k, grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)

        def modify(time, dt, param: Dirichlet_collider):
            if time >= param.start_time and time < param.end_time:
                param.point = wp.vec3(
                    param.point[0] + dt * param.velocity[0],
                    param.point[1] + dt * param.velocity[1],
                    param.point[2] + dt * param.velocity[2],
                )  # param.point + dt * param.velocity

        self.grid_postprocess.append(collide)
        self.modify_bc.append(None)

    def add_bounding_box(self, start_time=0.0, end_time=999.0):
        collider_param = Dirichlet_collider()
        collider_param.start_time = start_time
        collider_param.end_time = end_time

        self.collider_params.append(collider_param)

        @wp.kernel
        def collide(
            time: float,
            dt: float,
            state: MPMStateStruct,
            model: MPMModelStruct,
            param: Dirichlet_collider,
        ):
            k, grid_x, grid_y, grid_z = wp.tid()
            padding = 5
            if time >= param.start_time and time < param.end_time:
                x_n = float(grid_x) + state.grid_v[k, grid_x, grid_y, grid_z][0] * dt
                if x_n <= padding and state.grid_v[k, grid_x, grid_y, grid_z][0] < 0.0:
                    state.grid_v[k, grid_x, grid_y, grid_z] = wp.vec3(
                        0.0,
                        state.grid_v[k, grid_x, grid_y, grid_z][1],
                        state.grid_v[k, grid_x, grid_y, grid_z][2],
                    )
                if (
                    x_n >= model.grid_dim_x - padding
                    and state.grid_v[k, grid_x, grid_y, grid_z][0] > 0.0
                ):
                    state.grid_v[k, grid_x, grid_y, grid_z] = wp.vec3(
                        0.0,
                        state.grid_v[k, grid_x, grid_y, grid_z][1],
                        state.grid_v[k, grid_x, grid_y, grid_z][2],
                    )
                y_n = float(grid_y) + state.grid_v[k, grid_x, grid_y, grid_z][1] * dt
                if y_n <= padding and state.grid_v[k, grid_x, grid_y, grid_z][1] < 0.0:
                    state.grid_v[k, grid_x, grid_y, grid_z] = wp.vec3(
                        state.grid_v[k, grid_x, grid_y, grid_z][0],
                        0.0,
                        state.grid_v[k, grid_x, grid_y, grid_z][2],
                    )
                if (
                    y_n >= model.grid_dim_y - padding
                    and state.grid_v[k, grid_x, grid_y, grid_z][1] > 0.0
                ):
                    state.grid_v[k, grid_x, grid_y, grid_z] = wp.vec3(
                        state.grid_v[k, grid_x, grid_y, grid_z][0],
                        0.0,
                        state.grid_v[k, grid_x, grid_y, grid_z][2],
                    )

                z_n = float(grid_z) + state.grid_v[k, grid_x, grid_y, grid_z][2] * dt
                if z_n <= padding and state.grid_v[k, grid_x, grid_y, grid_z][2] < 0.0:
                    state.grid_v[k, grid_x, grid_y, grid_z] = wp.vec3(
                        state.grid_v[k, grid_x, grid_y, grid_z][0],
                        state.grid_v[k, grid_x, grid_y, grid_z][1],
                        0.0,
                    )
                if (
                    z_n >= model.grid_dim_z - padding
                    and state.grid_v[k, grid_x, grid_y, grid_z][2] > 0.0
                ):
                    state.grid_v[k, grid_x, grid_y, grid_z] = wp.vec3(
                        state.grid_v[k, grid_x, grid_y, grid_z][0],
                        state.grid_v[k, grid_x, grid_y, grid_z][1],
                        0.0,
                    )

        self.grid_postprocess.append(collide)
        self.modify_bc.append(None)

    def add_plane(self, x_plane, y_plane, z_plane, start_time=0.0, end_time=999.0):
        collider_param = Dirichlet_collider()
        collider_param.start_time = start_time
        collider_param.end_time = end_time
        collider_param.x_plane = x_plane
        collider_param.y_plane = y_plane
        collider_param.z_plane = z_plane
        
        self.collider_params.append(collider_param)
        self.impulse_params.append(collider_param)
        
        @wp.kernel
        def collide(
            time: float,
            dt: float,
            state: MPMStateStruct,
            model: MPMModelStruct,
            param: Dirichlet_collider,
        ):
            if time >= param.start_time and time < param.end_time:
                k, grid_x, grid_y, grid_z = wp.tid()
                dpos_x = float(grid_x) * model.dx
                dpos_y = float(grid_y) * model.dx
                dpos_z = float(grid_z) * model.dx

                x_n = dpos_x + state.grid_v[k, grid_x, grid_y, grid_z][0] * dt
                if x_n <= param.x_plane[0] and state.grid_v[k, grid_x, grid_y, grid_z][0] < 0.0:
                # if x_n <= param.x_plane[0]:
                    state.grid_v[k, grid_x, grid_y, grid_z] = wp.vec3(
                        0.0,
                        state.grid_v[k, grid_x, grid_y, grid_z][1],
                        state.grid_v[k, grid_x, grid_y, grid_z][2],
                    )

                if x_n >= param.x_plane[1] and state.grid_v[k, grid_x, grid_y, grid_z][0] > 0.0:
                # if x_n >= param.x_plane[1]:
                    state.grid_v[k, grid_x, grid_y, grid_z] = wp.vec3(
                        0.0,
                        state.grid_v[k, grid_x, grid_y, grid_z][1],
                        state.grid_v[k, grid_x, grid_y, grid_z][2],
                    )
                 
                y_n = dpos_y + state.grid_v[k, grid_x, grid_y, grid_z][1] * dt
                if y_n <= param.y_plane[0] and state.grid_v[k, grid_x, grid_y, grid_z][1] < 0.0:
                # if y_n <= param.y_plane[0]:
                    state.grid_v[k, grid_x, grid_y, grid_z] = wp.vec3(
                        state.grid_v[k, grid_x, grid_y, grid_z][0],
                        0.0,
                        state.grid_v[k, grid_x, grid_y, grid_z][2],
                    )
                if y_n >= param.y_plane[1] and state.grid_v[k, grid_x, grid_y, grid_z][1] > 0.0:
                # if y_n >= param.y_plane[1]:
                    state.grid_v[k, grid_x, grid_y, grid_z] = wp.vec3(
                        state.grid_v[k, grid_x, grid_y, grid_z][0],
                        0.0,
                        state.grid_v[k, grid_x, grid_y, grid_z][2],
                    )
                
                z_n = dpos_z + state.grid_v[k, grid_x, grid_y, grid_z][2] * dt
                if z_n <= param.z_plane[0] and state.grid_v[k, grid_x, grid_y, grid_z][2] < 0.0:
                # if z_n <= param.z_plane[0]:
                    state.grid_v[k, grid_x, grid_y, grid_z] = wp.vec3(
                        state.grid_v[k, grid_x, grid_y, grid_z][0],
                        state.grid_v[k, grid_x, grid_y, grid_z][1],
                        0.0,
                    )
                if z_n >= param.z_plane[1] and state.grid_v[k, grid_x, grid_y, grid_z][2] > 0.0:
                # if z_n >= param.z_plane[1]:
                    state.grid_v[k, grid_x, grid_y, grid_z] = wp.vec3(
                        state.grid_v[k, grid_x, grid_y, grid_z][0],
                        state.grid_v[k, grid_x, grid_y, grid_z][1],
                        0.0)
        @wp.kernel
        def collide2(
            time: float,
            dt: float,
            state: MPMStateStruct,
            param: Dirichlet_collider,
        ):
            if time >= param.start_time and time < param.end_time:
                p = wp.tid()

                x_n = state.particle_x[p][0] + state.particle_v[p][0] * dt
                if x_n <= param.x_plane[0] and state.particle_v[p][0] < 0:
                # if x_n <= param.x_plane[0]:
                    state.particle_v[p] = wp.vec3(
                        0.0,
                        state.particle_v[p][1],
                        state.particle_v[p][2])

                if x_n >= param.x_plane[1] and state.particle_v[p][0] > 0:
                # if x_n >= param.x_plane[1]:
                    state.particle_v[p] = wp.vec3(
                        0.0,
                        state.particle_v[p][1],
                        state.particle_v[p][2])
                 
                y_n = state.particle_x[p][1] + state.particle_v[p][1] * dt
                if y_n <= param.y_plane[0] and state.particle_v[p][1] < 0:
                # if y_n <= param.y_plane[0]:
                    state.particle_v[p] = wp.vec3(
                        state.particle_v[p][0],
                        0.0,
                        state.particle_v[p][2])
                if y_n >= param.y_plane[1] and state.particle_v[p][1] > 0:
                # if y_n >= param.y_plane[1]:
                    state.particle_v[p] = wp.vec3(
                        state.particle_v[p][0],
                        0.0,
                        state.particle_v[p][2])
                
                z_n = state.particle_x[p][2] + state.particle_v[p][2] * dt
                if z_n <= param.z_plane[0] and state.particle_v[p][2] < 0:
                # if z_n <= param.z_plane[0]:
                    state.particle_v[p] = wp.vec3(
                        state.particle_v[p][0],
                        state.particle_v[p][1],
                        0.0)
                if z_n >= param.z_plane[1] and state.particle_v[p][2] > 0:
                # if z_n >= param.z_plane[1]:
                    state.particle_v[p] = wp.vec3(
                        state.particle_v[p][0],
                        state.particle_v[p][1],
                        0.0)
        
        self.pre_p2g_operations.append(collide2)
        self.grid_postprocess.append(collide)
        self.modify_bc.append(None)
                
    def object_static(self, object_index, start_time = 0.0, end_time = 1e3, reset=0, static_inf = False):
        static_object_param = Static_object()
        static_object_param.object_index = object_index
        static_object_param.start_time = start_time
        static_object_param.end_time = end_time
        static_object_param.reset = reset
        static_object_param.static_inf = static_inf
        # breakpoint()
        
        self.static_object_params.append(static_object_param)
        @wp.kernel
        def collide(
            time: float,
            state: MPMStateStruct,
            model: MPMModelStruct,
            param: Static_object,
        ):
            grid_x, grid_y, grid_z = wp.tid()
            object_index = param.object_index
            if time >= param.start_time and time < param.end_time:
                # grid_v_in_mean = (state.grid_v_in[object_index * 2, grid_x, grid_y, grid_z][0] + state.grid_v_in[object_index * 2, grid_x, grid_y, grid_z][1] + state.grid_v_in[object_index * 2, grid_x, grid_y, grid_z][2])
                # if grid_v_in_mean == 0:
                state.grid_v[object_index, grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)
                # state.grid_v_in[object_index * 2, grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)
                state.grid_v_in[object_index * 2 + 1, grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0) 
                if param.static_inf and state.grid_m[object_index, grid_x, grid_y, grid_z] > 1e-15:
                    for i in range(0, model.object_nums):
                        state.grid_v[i, grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)
                
        @wp.kernel
        def collide2(
            time: float,
            state: MPMStateStruct,
            param: Static_object,
            dt: float
        ):
            p = wp.tid()
            if (
                time >= start_time
                and time < end_time
            ):
                if state.particle_object_link[p] == param.object_index:
                    # breakpoint()
                    # state.particle_C[p] = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                    state.particle_x[p] = state.particle_x[p] + dt * state.particle_v[p]
                    state.particle_v[p] = wp.vec3(0.0, 0.0, 0.0)
                    state.particle_F[p] = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

        self.object_grid_process.append(collide)
        self.object_particle_process.append(collide2)
        
    # particle_v += force/particle_mass * dt
    # this is applied from start_dt, ends after num_dt p2g2p's
    # particle velocity is changed before p2g at each timestep
    def add_impulse_on_particles(
        self,
        force,
        dt,
        point=[1, 1, 1],
        size=[1, 1, 1],
        num_dt=1,
        start_time=0.0,
        end_time = 0.0,
        interval=1e3,
        device="cuda:0",
    ):
        impulse_param = Impulse_modifier()
        # impulse_param.start_time = start_time
        # impulse_param.end_time = start_time + dt * num_dt
        if not end_time:
            end_time = start_time + dt * num_dt
        # breakpoint()
        interval = float(interval)
        # breakpoint()
        start_time_range = np.arange(start_time, end_time, interval)
        end_time_range = np.arange(start_time + dt * num_dt, end_time + dt * num_dt, interval)
        impulse_param.start_time = wp.from_numpy(start_time_range, dtype=float, device=device)
        impulse_param.end_time = wp.from_numpy(end_time_range, dtype=float, device=device)
        # breakpoint()

        impulse_param.point = wp.vec3(point[0], point[1], point[2])
        impulse_param.size = wp.vec3(size[0], size[1], size[2])
        impulse_param.mask = wp.zeros(shape=self.n_particles, dtype=int, device=device)
        

        impulse_param.force = wp.vec3(
            force[0],
            force[1],
            force[2],
        )

        # wp.launch(
        #     kernel=selection_add_impulse_on_particles,
        #     dim=self.n_particles,
        #     inputs=[self.mpm_state, impulse_param],
        #     device=device,
        # )

        self.impulse_params.append(impulse_param)

        @wp.kernel
        def apply_force(
            time: float, dt: float, state: MPMStateStruct, param: Impulse_modifier
        ):
            p = wp.tid()
            time_index = int(np.floor((time - start_time) / interval))
            if time >= param.start_time[time_index] and time < param.end_time[time_index]:
                offset = state.particle_x[p] - param.point
                if (
                    wp.abs(offset[0]) < param.size[0]
                    and wp.abs(offset[1]) < param.size[1]
                    and wp.abs(offset[2]) < param.size[2]
                ):
                    impulse = wp.vec3(
                        param.force[0] / state.particle_mass[p],
                        param.force[1] / state.particle_mass[p],
                        param.force[2] / state.particle_mass[p],
                    )
                    state.particle_v[p] = state.particle_v[p] + impulse * dt

        self.pre_p2g_operations.append(apply_force)

    def enforce_particle_velocity_translation(
        self, point, size, velocity, start_time, end_time, object_index = 0, device="cuda:0"
    ):

        # first select certain particles based on position

        velocity_modifier_params = ParticleVelocityModifier()

        velocity_modifier_params.point = wp.vec3(point[0], point[1], point[2])
        velocity_modifier_params.size = wp.vec3(size[0], size[1], size[2])

        velocity_modifier_params.velocity = wp.vec3(
            velocity[0], velocity[1], velocity[2]
        )

        velocity_modifier_params.start_time = start_time
        velocity_modifier_params.end_time = end_time

        velocity_modifier_params.mask = wp.zeros(
            shape=self.n_particles, dtype=int, device=device
        )
        velocity_modifier_params.object_index = object_index

        wp.launch(
            kernel=selection_enforce_particle_velocity_translation,
            dim=self.n_particles,
            inputs=[self.mpm_state, velocity_modifier_params],
            device=device,
        )
        self.particle_velocity_modifier_params.append(velocity_modifier_params)

        @wp.kernel
        def modify_particle_v_before_p2g(
            time: float,
            state: MPMStateStruct,
            velocity_modifier_params: ParticleVelocityModifier,
        ):
            p = wp.tid()
            if (
                time >= velocity_modifier_params.start_time
                and time < velocity_modifier_params.end_time
            ):
                if velocity_modifier_params.mask[p] == 1:
                    state.particle_v[p] = velocity_modifier_params.velocity

        self.particle_velocity_modifiers.append(modify_particle_v_before_p2g)
        
    
    def enforce_object_particle_velocity_translation(
        self, object_index, start_time, end_time, device="cuda:0"
    ):

        # first select certain particles based on position

        velocity_modifier_params = ParticleVelocityModifier()

        velocity_modifier_params.object_index = object_index

        velocity_modifier_params.start_time = start_time
        velocity_modifier_params.end_time = end_time

        velocity_modifier_params.mask = wp.zeros(
            shape=self.n_particles, dtype=int, device=device
        )
        
        velocity_modifier_params.velocity = wp.vec3(0.0, 0.0, 0.0)

        wp.launch(
            kernel=selection_enforce_object_particle_velocity_translation,
            dim=self.n_particles,
            inputs=[self.mpm_state, velocity_modifier_params],
            device=device,
        )
        self.particle_velocity_modifier_params.append(velocity_modifier_params)

        @wp.kernel
        def modify_particle_v_before_p2g(
            time: float,
            state: MPMStateStruct,
            velocity_modifier_params: ParticleVelocityModifier,
        ):
            p = wp.tid()
            if (
                time >= velocity_modifier_params.start_time
                and time < velocity_modifier_params.end_time
            ):
                if state.particle_object_link[p] == velocity_modifier_params.object_index and velocity_modifier_params.mask[p] == 1:
                    state.particle_v[p] = velocity_modifier_params.velocity

        self.particle_velocity_modifiers.append(modify_particle_v_before_p2g)

    # define a cylinder with center point, half_height, radius, normal
    # particles within the cylinder are rotating along the normal direction
    # may also have a translational velocity along the normal direction
    def enforce_particle_velocity_rotation(
        self,
        point,
        normal,
        half_height_and_radius,
        rotation_scale,
        translation_scale,
        object_index,
        start_time,
        end_time,
        device="cuda:0",
    ):

        normal_scale = 1.0 / wp.sqrt(
            float(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
        )
        normal = list(normal_scale * x for x in normal)

        velocity_modifier_params = ParticleVelocityModifier()

        velocity_modifier_params.point = wp.vec3(point[0], point[1], point[2])
        velocity_modifier_params.half_height_and_radius = wp.vec2(
            half_height_and_radius[0], half_height_and_radius[1]
        )
        velocity_modifier_params.normal = wp.vec3(normal[0], normal[1], normal[2])

        horizontal_1 = wp.vec3(1.0, 1.0, 1.0)
        if wp.abs(wp.dot(velocity_modifier_params.normal, horizontal_1)) < 0.01:
            horizontal_1 = wp.vec3(0.72, 0.37, -0.67)
        horizontal_1 = (
            horizontal_1
            - wp.dot(horizontal_1, velocity_modifier_params.normal)
            * velocity_modifier_params.normal
        )
        horizontal_1 = horizontal_1 * (1.0 / wp.length(horizontal_1))
        horizontal_2 = wp.cross(horizontal_1, velocity_modifier_params.normal)

        velocity_modifier_params.horizontal_axis_1 = horizontal_1
        velocity_modifier_params.horizontal_axis_2 = horizontal_2

        velocity_modifier_params.rotation_scale = rotation_scale
        velocity_modifier_params.translation_scale = translation_scale

        velocity_modifier_params.start_time = start_time
        velocity_modifier_params.end_time = end_time

        velocity_modifier_params.mask = wp.zeros(
            shape=self.n_particles, dtype=int, device=device
        )
        velocity_modifier_params.object_index = object_index

        wp.launch(
            kernel=selection_enforce_particle_velocity_cylinder,
            dim=self.n_particles,
            inputs=[self.mpm_state, velocity_modifier_params],
            device=device,
        )
        self.particle_velocity_modifier_params.append(velocity_modifier_params)

        @wp.kernel
        def modify_particle_v_before_p2g(
            time: float,
            state: MPMStateStruct,
            velocity_modifier_params: ParticleVelocityModifier,
        ):
            p = wp.tid()
            if (
                time >= velocity_modifier_params.start_time
                and time < velocity_modifier_params.end_time
            ):
                if velocity_modifier_params.mask[p] == 1:
                    offset = state.particle_x[p] - velocity_modifier_params.point
                    horizontal_distance = wp.length(
                        offset
                        - wp.dot(offset, velocity_modifier_params.normal)
                        * velocity_modifier_params.normal
                    )
                    cosine = (
                        wp.dot(offset, velocity_modifier_params.horizontal_axis_1)
                        / horizontal_distance
                    )
                    theta = wp.acos(cosine)
                    if wp.dot(offset, velocity_modifier_params.horizontal_axis_2) > 0:
                        theta = theta
                    else:
                        theta = -theta
                    axis1_scale = (
                        -horizontal_distance
                        * wp.sin(theta)
                        * velocity_modifier_params.rotation_scale
                    )
                    axis2_scale = (
                        horizontal_distance
                        * wp.cos(theta)
                        * velocity_modifier_params.rotation_scale
                    )
                    axis_vertical_scale = translation_scale
                    state.particle_v[p] = (
                        axis1_scale * velocity_modifier_params.horizontal_axis_1
                        + axis2_scale * velocity_modifier_params.horizontal_axis_2
                        + axis_vertical_scale * velocity_modifier_params.normal
                    )

        self.particle_velocity_modifiers.append(modify_particle_v_before_p2g)

    # given normal direction, say [0,0,1]
    # gradually release grid velocities from start position to end position
    def release_particles_sequentially(
        self, normal, start_position, end_position, num_layers, start_time, end_time, point
    ):
        num_layers = num_layers
        # point = [0, 0, 0]
        size = [0, 0, 0]
        axis = -1
        for i in range(3):
            if normal[i] == 0:
                # point[i] = 1
                size[i] = 0.5
            else:
                axis = i
                point[i] = end_position

        half_length_portion = wp.abs(start_position - end_position) / num_layers
        end_time_portion = end_time / num_layers
        for i in range(num_layers):
            size[axis] = half_length_portion * (num_layers - i)
            # breakpoint()
            self.enforce_particle_velocity_translation(
                point=point,
                size=size,
                velocity=[0, 0, 0],
                start_time=start_time,
                end_time=end_time_portion * (i + 1),
            )
