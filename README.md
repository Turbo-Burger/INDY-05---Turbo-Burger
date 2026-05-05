# Turbo Burger

An NVIDIA Isaac Sim project building toward a UR5e robot arm that
autonomously makes smash burgers — picks up a beef ball, smashes it on
a flat-top grill, watches it cook with a camera, and assembles the
finished burger.

> **CS 4850 · Section 02 · Spring 2026 — INDY-05 prototype**
> Will Martin · Michael Collins · Logan Nelson

The robot uses a [TripleA Robotics WINGMAN](https://triplea-robotics.com/)–style
automatic tool changer to swap between utensils (spatula, etc.) that
live in a holder rack on the table. Tool attach/detach is abstracted in
sim by creating and deleting fixed joints rather than simulating the
magnetic coupler physics.

## Why Isaac Sim

Earlier robotics projects in the program leaned on general-purpose
game engines, where teams have to build workarounds to approximate
physics — and the resulting "sim-to-real gap" means policies that work
in sim fail unpredictably on real hardware. Isaac Sim is built on
NVIDIA Omniverse with physically accurate PhysX dynamics, RTX-rendered
sensor simulation, and a workflow designed for robotics from the
start, which closes that gap and makes simulation actually useful for
training and validation before any code touches a physical robot.

The project is the first chunk of an end-to-end **design → train →
validate → deploy** pipeline. Deployment to physical hardware is out
of scope for the course, but every choice in the sim is made with that
eventual handoff in mind.

## Tech stack

| Tool | Role |
|---|---|
| **Isaac Sim** | Simulation core — scene, physics (PhysX), RTX rendering, sensor sim |
| **ROS 2** | Control interface; Isaac publishes sensor data, subscribes to joint commands |
| **Python** | Extension code, AI/ML pipelines, Replicator scripts |
| **Blender** | Custom 3D geometry for assets not available off-the-shelf |
| **Anchorpoint** | File locking on top of Git for binary Blender assets — prevents merge conflicts |
| **Git + Git LFS** | Code + USD/binary asset versioning |

### System requirements

- **GPU:** NVIDIA RTX (RTX 2070 minimum). Isaac Sim relies on PhysX +
  RTX — there is no software-only fallback.
- **CPU / RAM:** Intel i7 / Ryzen 7 or better, 32 GB RAM minimum.
- **OS / storage:** Ubuntu 20.04 or 22.04 (ROS + Isaac Sim are
  Linux-native), SSD with 50+ GB free for the Isaac Sim install.

## Robot

| | |
|---|---|
| Model | Universal Robots UR5e |
| DOF | 6 |
| Payload | 5 kg |
| Reach | 850 mm |
| Control | Position / velocity / effort via ROS joint state interfaces |
| Sensors | Simulated RGB-D depth camera for perception + doneness CV |

## Repo layout

```
.
├── scenes/                       Primary USD stages (Git LFS)
│   ├── turboburger.usd           ← live working scene
│   └── Collected_turboburger_for_repo/
│                                 self-contained "Collected" export with
│                                 SubUSDs/ and textures/ — opens out of
│                                 the box without external asset paths
├── scripts/                      Isaac Script Editor utilities
│   ├── cooking_sim.py            Animates patty diffuse color raw → cooked
│   ├── detect_doneness.py        CV that watches the patty and flags DONE
│   ├── replicator_food_dataset.py
│   │                             Replicator pipeline → RGB + semantic +
│   │                             instance + 2D bbox dataset (~2000 frames)
│   │                             used to fine-tune a MobileNet classifier
│   │                             for order validation
│   └── spatula_pickup.py         Scripted IK sequence: approach → grab →
│                                 lift spatula out of holder → staging pose
├── src/exts/turboburger.ext/     Isaac Sim extension (the runtime; see below)
└── assets/                       Local meshes / materials / Blender sources
```

## The extension (`src/exts/turboburger.ext`)

This is where the robot's runtime control code lives. It auto-loads at
Isaac startup and, on Play, takes over the UR5e and drives it to
follow `/World/ToolApproachTarget`. Everything that should "just work
when I press Play" goes here — anything experimental or one-shot stays
in `scripts/`.

To install the extension in Isaac Sim:

1. `Window → Extensions → Settings (gear)`
2. Add an extension search path pointing at this repo's `src/exts/` folder
3. Find **Turbo Burger Mainframe**, enable it, check *Autoload*

Useful constants are at the top of
`src/exts/turboburger.ext/turboburger/main/extension.py`:

| Name | Purpose |
|---|---|
| `ROBOT_PRIM_PATH` | path to the UR5e articulation root |
| `TARGET_PRIM_PATH` | path to the IK goal cube (auto-created if missing) |
| `TOOL_CONNECTOR_PRIM_PATH` | when set, IK targets the active tool's connector instead of the goal cube |
| `USE_TARGET_ORIENTATION` | follow target rotation (`True`) or hold a fixed wrist pose (`False`) |
| `TOOLS_TO_RESET_ON_STOP` | tool joints to clear on Stop so each Play starts with empty hand |

`stage_dumper.py` (alongside `extension.py`) auto-saves several JSON
snapshots of the open stage to `llm-helper-files/ai-context/` whenever
the file is saved — these are intentionally checked in as LLM context
(full prim list, physics summary, transforms, scene tree, USDA
export).

## Scenes & Git LFS

USD files in this repo are tracked via Git LFS (see `.gitattributes`).
After cloning, you need LFS to actually pull the binary content:

```
git lfs install
git lfs pull
```

`scenes/turboburger.usd` is the working scene. The
`Collected_turboburger_for_repo/` folder is an Omniverse "Collect"
export — it inlines every external reference into a local `SubUSDs/`
tree, so it opens cleanly on a fresh machine without needing assets
mounted at the original author's paths.

## Scripts

The files in `scripts/` are not part of the extension — they're meant
to be opened in Isaac Sim's **Script Editor** and run with `Ctrl+Enter`
while a stage is open and the timeline is playing. Each file's
docstring at the top explains its setup, run sequence, and how to stop
it (each script keeps a module-level subscription handle, e.g.
`_cooking_sub.unsubscribe()`).

`replicator_food_dataset.py` can also be run headless against the
Isaac Python interpreter for full-size data-generation runs:

```
cd <isaacsim build>
./python.sh /path/to/scripts/replicator_food_dataset.py
```

## Scope: what we set out to do vs. what we delivered

The original plan included a **physical Denso VS-087** 6-axis
industrial arm working alongside the UR5e in sim, with a ROS 2 bridge
([DENSORobot/denso_robot_ros2](https://github.com/DENSORobot/denso_robot_ros2))
driving the real hardware. Hardware access and configuration issues at
the school made that unworkable inside the project window, so we
pivoted to running everything in simulation against the UR5e.

**What we got working:**

- A complete Isaac Sim scene — UR5e cobot, kitchen environment, custom
  hand-modeled assets (utensils, beef-ball trays, patty, refrigerator,
  TripleA tool holder), active physics, and RTX rendering.
- Machine-vision pipeline — Replicator-driven synthetic dataset
  (~2000 frames of randomized food placements with RGB + semantic +
  instance segmentation + 2D bounding boxes) used to fine-tune a
  MobileNet CNN for order-validation classification.
- Cooking simulation + doneness detection — Python animation of the
  patty's diffuse color raw → cooked, plus an HSV-value-histogram CV
  detector that flags DONE based on traditional pixel math.
- Tool-change system — TripleA WINGMAN-style attach/detach abstracted
  with runtime fixed-joint creation; UR5e successfully picks up the
  spatula from its holder and moves to the grill.

**What we didn't reach:**

- Full robot path planning for the multi-step food-prep sequence. We
  landed at end-effector tool swaps plus rudimentary obstacle
  avoidance and IK-target following — the orchestrated
  ball-pick → smash → flip → assemble motion plan was out of reach in
  the time available.
- Physical hardware deployment (Denso pivot). Always was a stretch
  goal beyond the course scope, but the Denso path being closed off
  meant we couldn't even attempt the sim-to-real handoff.
