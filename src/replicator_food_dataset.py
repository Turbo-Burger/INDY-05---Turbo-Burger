"""Replicator data generation for the order-validation CV pipeline.

Produces RGB + semantic seg + instance seg + 2D bbox + camera params
for free-form placement of the 5 food classes on the table.

Run from Isaac's Script Editor:
  File -> Open -> select this file, then Ctrl+Enter.

To do a quick smoke test, drop NUM_FRAMES to 10 first.
Output appears in OUTPUT_DIR — check there for rgb_*.png.

For the full 2000-frame run, you'll want to run this headless:
  cd <isaacsim build>; ./python.sh "/home/michael/turbo burger/scripts/replicator_food_dataset.py"
(faster than rendering through the GUI).
"""

import re
import shutil
from pathlib import Path

import omni.replicator.core as rep
from pxr import UsdGeom

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NUM_FRAMES = 100               # bump to 2000 for the real run
OUTPUT_DIR = "/home/michael/turbo burger/dataset"
RESOLUTION = (640, 480)

# Wipe the output dir before each run. BasicWriter doesn't clean up,
# so without this you'd get stale frames from previous runs mixed in
# with new ones (and the labels CSV would silently be wrong).
# Set to False if you ever want to accumulate runs into one dataset.
CLEAR_OUTPUT_DIR = True

# Clear generated Replicator graphs from prior Script Editor runs. Without
# this, stale randomizers can keep executing alongside the new graph.
CLEAR_REPLICATOR_GRAPH = True
GENERATED_REPLICATOR_PRIMS = (
    "/Replicator",
    "/WriterOrchestrator",
    "/Orchestrator",
)

# Each entry maps a class label (matches the Semantic data we authored)
# to a regex that selects all clone prims for that class.
#
# IMPORTANT: patterns MUST be anchored (^...$) and only match the food xform
# itself, not its descendants. Otherwise Replicator tries to set visibility
# and translate on Materials/Shaders/Meshes, producing the noisy warnings
# we hit before. The (_\d+)? optional suffix matches both the original
# (e.g. /World/soda) and its numbered clones (/World/soda_01 ... _04).
FOOD_CLASSES = {
    "hamburger":  r"^/World/finishedburger(_\d+)?$",
    "fries":      r"^/World/fries(_\d+)?$",
    "hotdog":     r"^/World/hotdog(_\d+)?$",
    "milkshake":  r"^/World/milkshake(_\d+)?$",
    "soda":       r"^/World/soda(_\d+)?$",
}

# Region of the table where items can land (meters, world space).
# Measured by laying all 25 clones out non-overlapping — this is the
# usable tabletop area within the camera frame.
TABLE_X_RANGE = (1.7589, 2.4030)
TABLE_Y_RANGE = (-1.4232, -1.1156)

# Per-class Z (height of the prim's origin so the mesh sits flat on the
# table top). Each food's mesh has its origin at a different relative
# height because of how Sketchfab's FBX exporter chose pivots, so we
# need different Z values per class to avoid items sinking into the
# table or floating above it. Measured by Michael 2026-05-03.
FLOOR_Z = {
    "hamburger":  0.7918,
    "fries":      0.7486,
    "hotdog":     0.7714,
    "milkshake":  0.8025,
    "soda":       0.7481,
}

# Camera: use an existing camera prim authored in the scene.
# (Edit this path if you named yours differently.)
CAMERA_PRIM_PATH = "/World/orderUpCamera"

# Rotation parameters.
#
# The imported food assets have a top-level rotateX:unitsResolve(90) op.
# Replicator's rotation writer expects Euler degrees and replaces the prim's
# effective rotation stack, so include that +90deg X correction in the sampled
# pose instead of trying to write quaternion values to xformOp:orient.
UPRIGHT_X_DEG       = 90.0
YAW_RANGE_DEG       = (0.0, 360.0)
HOTDOG_TILT_DEG     = 30.0     # +/- around the hotdog's long axis


def clear_replicator_graphs():
    """Remove generated Replicator graph prims left behind by previous runs."""
    if not CLEAR_REPLICATOR_GRAPH:
        return

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    for prim_path in GENERATED_REPLICATOR_PRIMS:
        if stage.GetPrimAtPath(prim_path).IsValid():
            stage.RemovePrim(prim_path)


def normalize_food_rotation_ops():
    """Make rotateXYZ the active top-level rotation op on every food clone."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    food_patterns = [re.compile(pattern) for pattern in FOOD_CLASSES.values()]
    for prim in stage.Traverse():
        prim_path = prim.GetPath().pathString
        if not any(pattern.match(prim_path) for pattern in food_patterns):
            continue

        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            continue

        rotate_attr = prim.GetAttribute("xformOp:rotateXYZ")
        if not rotate_attr:
            rotate_attr = xformable.AddRotateXYZOp(UsdGeom.XformOp.PrecisionFloat).GetAttr()
        rotate_attr.Set((UPRIGHT_X_DEG, 0.0, 0.0))

        current_order = list(prim.GetAttribute("xformOpOrder").Get() or [])
        new_order = []
        if prim.HasAttribute("xformOp:translate"):
            new_order.append("xformOp:translate")
        new_order.append("xformOp:rotateXYZ")

        for op_name in current_order:
            if op_name.startswith("xformOp:scale") and op_name not in new_order:
                new_order.append(op_name)

        for op_name in ("xformOp:scale", "xformOp:scale:unitsResolve"):
            if prim.HasAttribute(op_name) and op_name not in new_order:
                new_order.append(op_name)

        prim.GetAttribute("xformOpOrder").Set(new_order)


# ---------------------------------------------------------------------------
# Pre-flight: clear the output directory so we don't mix runs
# ---------------------------------------------------------------------------
output_path = Path(OUTPUT_DIR)
if CLEAR_OUTPUT_DIR and output_path.exists():
    shutil.rmtree(output_path)
output_path.mkdir(parents=True, exist_ok=True)

clear_replicator_graphs()
normalize_food_rotation_ops()

# ---------------------------------------------------------------------------
# Build the randomization graph
# ---------------------------------------------------------------------------

with rep.new_layer():

    # Resolve all 25 food prims, grouped by class.
    food_groups = {
        cls: rep.get.prims(path_pattern=pattern)
        for cls, pattern in FOOD_CLASSES.items()
    }

    # Use the existing hand-placed camera prim from the stage.
    render_product = rep.create.render_product(CAMERA_PRIM_PATH, RESOLUTION)

    # Per-frame randomization
    with rep.trigger.on_frame(num_frames=NUM_FRAMES):

        # For each food class, independently sample visibility + pose for every
        # clone in the class. Pass input_prims explicitly; relying on `with prims`
        # produced blank OgnGetPrims target nodes in this Isaac build.
        for cls, prims in food_groups.items():
            z = FLOOR_Z[cls]
            rotation_lower = (
                UPRIGHT_X_DEG - HOTDOG_TILT_DEG if cls == "hotdog" else UPRIGHT_X_DEG,
                0.0,
                YAW_RANGE_DEG[0],
            )
            rotation_upper = (
                UPRIGHT_X_DEG + HOTDOG_TILT_DEG if cls == "hotdog" else UPRIGHT_X_DEG,
                0.0,
                YAW_RANGE_DEG[1],
            )

            rep.modify.visibility(
                rep.distribution.choice([True, False]),
                input_prims=prims,
            )
            rep.modify.attribute(
                "xformOp:translate",
                rep.distribution.uniform(
                    (TABLE_X_RANGE[0], TABLE_Y_RANGE[0], z),
                    (TABLE_X_RANGE[1], TABLE_Y_RANGE[1], z),
                ),
                "double3",
                input_prims=prims,
            )
            rep.modify.attribute(
                "xformOp:rotateXYZ",
                rep.distribution.uniform(rotation_lower, rotation_upper),
                "double3",
                input_prims=prims,
            )

        # NOTE: camera jitter is intentionally skipped in v1 — you manually
        # placed orderUpCamera, so let's render from that exact pose first
        # and confirm the data looks right. To add jitter later, read the
        # camera's current world translate at script start and randomize
        # within +/- CAM_JITTER of that fixed base. (Don't randomize relative
        # to the camera's *current* position each frame — that drifts.)

    # ----------------------------------------------------------------------
    # Writer: turn on every annotation we might want, render once.
    # ----------------------------------------------------------------------
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=OUTPUT_DIR,
        rgb=True,
        semantic_segmentation=True,
        instance_segmentation=True,
        bounding_box_2d_tight=True,
        camera_params=True,
        distance_to_camera=True,
        colorize_semantic_segmentation=True,   # human-eyeable seg PNG
        colorize_instance_segmentation=True,
    )
    writer.attach([render_product])

# Kick off rendering. In the GUI this runs asynchronously — watch the
# console for progress and check OUTPUT_DIR for files appearing.
rep.orchestrator.run()
