<h3 align="center">A Versatile Framework for Generating Multiphase Dynamics with 3D Gaussians</h3>

<p align="center">
  Ruijie Su<sup></sup>, Lingxiao Yang<sup></sup>, Xiaohua Xie<sup></sup>, Jianhuang Lai<sup></sup>
</p>

<p align="center">
  <em>Accepted to IEEE International Conference on Multimedia and Expo (ICME) 2026</em>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Paper-ICME%202026-blue.svg" alt="Paper"></a>
  <a href="https://github.com/Elowen-surj/VersaGauss"><img src="https://img.shields.io/badge/Code-GitHub-black.svg" alt="Code"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/Python-3.9-yellow.svg" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/GPU-NVIDIA%20A100-76B900.svg" alt="GPU"></a>
</p>

---

## Abstract

Recent progress has been made in 3D Gaussian representation for reconstruction, generation, and physical simulation. However, current approaches mainly concentrate on physics-based dynamic generation of solid objects and only handle single-phase collision interactions. We introduce VersaGauss, a unified framework for generation, simulation, and rendering that supports versatile physics-based dynamic generation, particularly for multiphase interactions. Our system takes a few images as input and produces a realistic, physics-driven 3D dynamic scene with multiple objects. To optimize the Gaussian kernel distribution, we develop a particle pruning algorithm. We also propose the Coupled Multiphase Point Method (CMPM) to effectively model and generate multiphase interactions. Additionally, harmonic interpolation within CMPM and a Gaussian evolution strategy are introduced to achieve realistic fluid rendering. Extensive experiments demonstrate that our framework can simulate interactions among various materials such as fluid, rubber, sand, snow, and others.

---

## 🔧 Installation

VersaGauss has been tested with **Python 3.9** on a single **NVIDIA A100 GPU**
(CUDA-enabled). We recommend using `conda` to manage the environment.

```bash
# 0. Clone the repo together with its submodules
#    (gaussian-splatting for rendering, TRELLIS for 3D asset generation)
git clone --recursive https://github.com/Elowen-surj/VersaGauss.git
cd VersaGauss
# if you already cloned without --recursive:
# git submodule update --init --recursive

# 1. Create and activate the environment
conda create -n VersaGauss python=3.9
conda activate VersaGauss

# 2. Install core dependencies
pip install -r requirements.txt

# 3. Install 3D Gaussian Splatting CUDA extensions
pip install -e gaussian-splatting/submodules/diff-gaussian-rasterization/
pip install -e gaussian-splatting/submodules/simple-knn/

# 4. Set up TRELLIS (3D generation backbone) and its CUDA-accelerated extras
cd preprocess/TRELLIS
. ./setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --kaolin --nvdiffrast
cd ../..

# 5. Install Segment Anything (used for object segmentation in preprocessing)
pip install git+https://github.com/facebookresearch/segment-anything.git
```

> ⚠️ Steps 3–4 compile CUDA extensions and require a matching local CUDA
> toolkit / compiler. If installation fails, check that your CUDA version
> matches the PyTorch build specified in `requirements.txt`.

### Pretrained models

Download the SAM (Segment Anything) checkpoint used for object segmentation:

```bash
mkdir -p checkpoints
wget -O checkpoints/sam_vit_h_4b8939.pth \
  https://huggingface.co/HCMUE-Research/SAM-vit-h/blob/main/sam_vit_h_4b8939.pth
```

---

## 🚀 Quick Start

### 1. Preprocessing (optional): segmentation → 3D asset generation

> This step turns a plain image into per-object 3D Gaussian assets
> (`.ply` files under `model/`). **Skip it if you already have your own
> reconstructed Gaussian assets** — in that case, drop your `.ply` files
> into `model/`, point your config's `model_path` at them, and go straight
> to step 2 below.

This step needs the `TRELLIS` submodule and a SAM checkpoint:

```bash
# Make sure the TRELLIS submodule is initialized (see Installation step 0)
git submodule update --init preprocess/TRELLIS

# Download the SAM (Segment Anything) checkpoint used for object segmentation
mkdir -p checkpoints
wget -O checkpoints/sam_vit_h_4b8939.pth \
  https://huggingface.co/HCMUE-Research/SAM-vit-h/blob/main/sam_vit_h_4b8939.pth
```

```bash
cd preprocess

# Click on each object in the image to segment it with SAM;
# press 'a' to save the mask and move on to the next object.
python mask_segment.py --image path/to/image.jpg --checkpoint ../checkpoints/sam_vit_h_4b8939.pth

# Estimate each object's relative position/scale in the scene from the saved masks
python pos_get.py --image path/to/image.jpg

# Reconstruct each masked object as a static 3D Gaussian asset
cd TRELLIS
python example.py
cd ../..
```

### 2. Multiphase simulation & rendering

```bash
python gs_simulation.py \
  --output_path <path/to/output_folder> \
  --config <path/to/config.json> \
  --render_img \
  --compile_video \
  --sh_degree 0 \
  --white_bg
```

| Flag | Description |
|---|---|
| `--output_path` | Directory where rendered frames / video are saved |
| `--config` | Path to the scene's JSON configuration file (see below) |
| `--render_img` | Export per-frame rendered images |
| `--compile_video` | Additionally compile a video from the rendered frames |
| `--sh_degree` | Spherical harmonics degree used for color |
| `--white_bg` | Render against a white background |

Rendered images and the compiled video are saved to `<output_path>`.

For example, to reproduce the "snowball & toy" scene from the paper:

```bash
python gs_simulation.py \
  --output_path output/snowball_toy \
  --config config/snowball_toy.json \
  --render_img --compile_video --sh_degree 0 --white_bg
```

For example, to reproduce the paper's fluid–fluid mixing result with harmonic color interpolation enabled:

```bash
python gs_simulation.py \
  --output_path output/water_remix \
  --config config/water_remix_from_images.json \
  --render_img --compile_video --sh_degree 0 --shs_mix --white_bg
```

Or to composite the simulated scene onto a custom photo background instead of a plain white one:

```bash
python gs_simulation.py \
  --output_path output/faucet_ducks \
  --config config/faucet_ducks_from_image.json \
  --render_img --compile_video --sh_degree 0 \
  --background_img path/to/background.jpg
```

---

## ⚙️ Configuration

Each scene is fully described by a single `.json` file covering data
preprocessing, simulation, and export settings. Five example scenes from the
paper are included under `config/`:

| Config | Scene |
|---|---|
| `ball_snowman_from_image.json` | Snow & Elasticity — a tennis ball impacts a snowman |
| `snowball_toy_from_image_floor.json` | Snow & Elasticity — a snowball impacts a monkey doll |
| `faucet_ducks_from_image.json` | Fluid & Elasticity — rubber ducks carried by water flow |
| `faucet_sand_from_images.json` | Fluid & Sand — a sandcastle collapses under running water |
| `water_remix_from_images.json` | Fluid–Fluid Mixing — two fluids of different densities mixing |

### Data preprocessing parameters

| Parameter | Description |
|---|---|
| `opacity_threshold` | Filters out Gaussian kernels with opacity below this threshold |
| `rotation_degree`, `rotation_axis` | Rotates the scene to align with the simulation grid |
| `sim_area` | Bounding box of particles included in simulation: `[xmin, xmax, ymin, ymax, zmin, zmax]` |
| `particle_filling` | Dict specifying a cubic area for internal particle filling (see below) |

### Simulation parameters

| Parameter | Description |
|---|---|
| `material` | One of `jelly`, `metal`, `sand`, `foam`, `snow`, `plasticine` |
| `E` | Young's modulus |
| `nu` | Poisson's ratio |
| `density` | Material density |
| `g` | Gravitational acceleration |
| `substep_dt` | Simulation time-step size |
| `n_grid` | MPM background grid resolution |
| `boundary_conditions` | List of boundary conditions applied to particles or grid nodes |

### Export parameters

| Parameter | Description |
|---|---|
| `frame_dt` | Duration represented by each exported frame |
| `frame_num` | Total number of frames to export |
| `default_camera_index` | Camera view index (from the training set) used for rendering |

### Particle filling

An optional ray-collision-based method for filling interior particles of a
reconstructed object (useful since TRELLIS Gaussians concentrate near the
surface):

| Parameter | Description |
|---|---|
| `n_grid` | Grid resolution used for particle filling |
| `density_threshold` | Grid cells with density above this value are treated as surface shell |
| `search_exclude_direction` | List of directions excluded from ray casting for filling condition 1 (0–5 → +x,−x,+y,−y,+z,−z) |
| `ray_cast_direction` | Direction used to count ray collisions for filling condition 2 (same 0–5 mapping) |
| `max_particles_per_cell` | Maximum number of particles to fill per grid cell |
| `boundary` | Cubic region (well-reconstructed) within which filling is performed |

> **Note:** particle filling is sensitive to Gaussian kernel distribution and
> may produce unsatisfying results if the source Gaussians are noisy.

### Boundary conditions

| Type | Description |
|---|---|
| `bounding_box` | Prevents particles from leaving the MPM simulation domain |
| `cuboid` | Grid-level boundary condition applied over a cubic region. Requires `point` (center), `size` (half-extents), `velocity`, `start_time`, `end_time` |
| `enforce_particle_translation` | Particle-level boundary condition with parameters analogous to `cuboid` |

### Continuous particle injection (`add_particle`)

The optional top-level `add_particle` field simulates continuous inflow —
e.g. water pouring from a faucet — by re-injecting the **same seed batch of
particles every frame**, for as long as the current simulation time falls
within `[start_time, end_time)`. It is not a one-off addition: each frame in
that window spawns a fresh copy of the seed batch, so a scene with `N`
seed particles and `frame_num` frames inside the window accumulates roughly
`N × frame_num` new particles by `end_time`.

| Parameter | Description |
|---|---|
| `boundary` | Cubic region `[xmin, xmax, ymin, ymax, zmin, zmax]` used to select the seed particles (typically a thin slice at a fluid source, e.g. inside a faucet spout) |
| `object_material_index` | Index into `material_params` giving the material assigned to the injected particles |
| `move_pos` | Offset `[dx, dy, dz]` applied to the seed particles before injection (e.g. to move them from the source object to the actual spout opening) |
| `start_time`, `end_time` | Simulation time window (in seconds) during which particles are injected every frame |
| `additional` | If truthy, the seed batch is drawn from an additional/auxiliary particle set rather than the main scene particles (used in scenes like `faucet_sand_from_images.json`) |

---

## 📁 Repository Structure

```
VersaGauss/
├── config/                  # example scene configuration JSON files
├── model/                   # place your reconstructed .ply Gaussian assets here
├── preprocess/
│   ├── mask_segment.py      # interactive SAM-based object segmentation
│   ├── pos_get.py           # relative object position/scale estimation
│   └── TRELLIS/             # (git submodule) 3D asset generation backbone
├── mpm_solver_warp/         # CMPM: Warp-based multiphase MPM simulator
│   ├── mpm_solver_warp.py
│   ├── mpm_utils.py
│   ├── engine_utils.py
│   └── warp_utils.py
├── particle_filling/
│   └── filling.py           # ray-collision-based interior particle filling
├── utils/
│   ├── decode_param.py      # scene config JSON parsing
│   ├── transformation_utils.py
│   ├── camera_view_utils.py
│   ├── render_utils.py
│   ├── sh_utils.py
│   └── local_utils.py
├── gaussian-splatting/       # (git submodule) 3DGS rendering backend
├── gs_simulation.py          # CMPM simulation + rendering entry point
├── requirements.txt
├── .gitmodules
└── README.md
```

---

## 📝 Citation

If you find VersaGauss useful for your research, please consider citing:

```bibtex
@inproceedings{su2026versagauss,
  title     = {VersaGauss: A Versatile Framework for Generating Multiphase Dynamics with 3D Gaussians},
  author    = {Su, Ruijie and Yang, Lingxiao and Xie, Xiaohua and Lai, Jianhuang},
  booktitle = {IEEE International Conference on Multimedia and Expo (ICME)},
  year      = {2026}
}
```

---

## 🙏 Acknowledgements

Corresponding author: Xiaohua Xie (xiexiaoh6@mail.sysu.edu.cn). This work was
supported by the Project of Guangdong Provincial Key Laboratory of Information
Security Technology under Grant 2023B1212060026.

VersaGauss builds on prior work including [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/),
[PhysGaussian](https://xpandora.github.io/PhysGaussian/), [TRELLIS](https://github.com/microsoft/TRELLIS),
[Segment Anything](https://github.com/facebookresearch/segment-anything), and the
[Warp](https://github.com/NVIDIA/warp) GPU simulation framework. We thank the
authors of these projects for their open-source contributions.

## 📄 License

This project is released under the [MIT License](LICENSE).

## 🛠️ TODO

- [ ] Add more pretrained models
- [ ] Release additional example configs and scenes
