"""
Stage Dumper module for the Turbo Burger Control extension.
Serializes the current USD stage into structured files that an LLM can read.

Usage:
    from .stage_dumper import dump_everything
    dump_everything()  # writes files to OUTPUT_DIR
"""

import json
import os

import carb
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

# ── OUTPUT DIRECTORY ──
OUTPUT_DIR = "/home/michael/turbo burger/llm-helper-files/ai-context"


def dump_everything():
    """Dump the full USD stage to structured files for LLM consumption."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stage = omni.usd.get_context().get_stage()

    if stage is None:
        carb.log_warn("[stage_dumper] No USD stage is open — skipping dump.")
        return

    carb.log_info(f"[stage_dumper] Dumping stage to {OUTPUT_DIR} ...")

    _dump_scene_tree(stage)
    _dump_full_stage(stage)
    _dump_physics_summary(stage)
    _dump_materials(stage)
    _dump_transforms(stage)
    _dump_usda_export(stage)

    carb.log_info(f"[stage_dumper] DONE. All files written to: {OUTPUT_DIR}")


# ── INDIVIDUAL DUMP FUNCTIONS ──────────────────────────────────────────────


def _dump_scene_tree(stage):
    """Compact scene tree for quick orientation."""
    tree_lines = []
    for prim in stage.Traverse():
        depth = prim.GetPath().pathString.count("/") - 1
        indent = "  " * depth
        try:
            applied = [str(s) for s in prim.GetPrimTypeInfo().GetAppliedAPISchemas()]
        except Exception:
            applied = []
        tree_lines.append(
            f"{indent}{prim.GetPath()} [{prim.GetTypeName()}] schemas={applied}"
        )

    _write_text("scene_tree.txt", "\n".join(tree_lines))
    carb.log_info(f"[stage_dumper] scene_tree.txt — {len(tree_lines)} prims")


def _dump_full_stage(stage):
    """Every prim, every attribute, every value."""
    all_prims = {}
    for prim in stage.Traverse():
        prim_data = {
            "type": prim.GetTypeName(),
            "path": str(prim.GetPath()),
            "active": prim.IsActive(),
            "children": [str(c.GetPath()) for c in prim.GetChildren()],
            "attributes": {},
            "relationships": {},
            "applied_schemas": [],
            "metadata": {},
        }

        try:
            prim_data["applied_schemas"] = [
                str(s) for s in prim.GetPrimTypeInfo().GetAppliedAPISchemas()
            ]
        except Exception:
            pass

        for attr in prim.GetAttributes():
            try:
                if attr.HasValue():
                    val = attr.Get()
                    prim_data["attributes"][attr.GetName()] = {
                        "value": _serialize_value(val),
                        "type": str(attr.GetTypeName()),
                        "custom": attr.IsCustom(),
                        "authored": attr.HasAuthoredValue(),
                    }
            except Exception as e:
                prim_data["attributes"][attr.GetName()] = {"error": str(e)}

        for rel in prim.GetRelationships():
            targets = rel.GetTargets()
            if targets:
                prim_data["relationships"][rel.GetName()] = [str(t) for t in targets]

        for key in ["kind", "instanceable", "hidden", "documentation"]:
            val = prim.GetMetadata(key)
            if val is not None:
                prim_data["metadata"][key] = str(val)

        all_prims[str(prim.GetPath())] = prim_data

    _write_json("full_stage.json", all_prims)
    carb.log_info(
        f"[stage_dumper] full_stage.json — {len(all_prims)} prims with all properties"
    )


def _dump_physics_summary(stage):
    """Rigid bodies, colliders, joints, and physics scenes."""
    physics = {
        "rigid_bodies": [],
        "colliders": [],
        "joints": [],
        "physics_scene": [],
    }
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        type_name = prim.GetTypeName()

        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            physics["rigid_bodies"].append(path)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            physics["colliders"].append(path)
        if "Joint" in type_name:
            physics["joints"].append({"path": path, "type": type_name})
        if type_name == "PhysicsScene":
            physics["physics_scene"].append(path)

    _write_json("physics_summary.json", physics)
    carb.log_info(
        f"[stage_dumper] physics_summary.json — "
        f"{len(physics['rigid_bodies'])} rigid bodies, "
        f"{len(physics['colliders'])} colliders, "
        f"{len(physics['joints'])} joints"
    )


def _dump_materials(stage):
    """All materials and shader prims with authored attributes."""
    materials = {}
    for prim in stage.Traverse():
        if prim.GetTypeName() in ("Material", "Shader"):
            mat_data = {"type": prim.GetTypeName(), "attributes": {}}
            for attr in prim.GetAttributes():
                if attr.HasAuthoredValue():
                    mat_data["attributes"][attr.GetName()] = {
                        "value": _serialize_value(attr.Get()),
                        "type": str(attr.GetTypeName()),
                    }
            materials[str(prim.GetPath())] = mat_data

    _write_json("materials.json", materials)
    carb.log_info(f"[stage_dumper] materials.json — {len(materials)} material/shader prims")


def _dump_transforms(stage):
    """Local and world-space translations for all xformable prims."""
    transforms = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Xformable):
            try:
                xformable = UsdGeom.Xformable(prim)
                local = xformable.GetLocalTransformation(Usd.TimeCode.Default())
                world = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                transforms[str(prim.GetPath())] = {
                    "local_translate": list(local.ExtractTranslation()),
                    "world_translate": list(world.ExtractTranslation()),
                }
            except Exception:
                pass

    _write_json("transforms.json", transforms)
    carb.log_info(f"[stage_dumper] transforms.json — {len(transforms)} xformable prims")


def _dump_usda_export(stage):
    """Human-readable ASCII export of the full stage."""
    usda_path = os.path.join(OUTPUT_DIR, "stage_export.usda")
    stage.GetRootLayer().Export(usda_path)
    carb.log_info("[stage_dumper] stage_export.usda — full ASCII export")


# ── HELPERS ────────────────────────────────────────────────────────────────


def _serialize_value(val):
    """Best-effort serialization of USD values to JSON-friendly types."""
    if val is None:
        return None
    if isinstance(val, (bool, int, float, str)):
        return val
    if hasattr(val, "__len__") and hasattr(val, "__getitem__"):
        try:
            lst = list(val)
            if len(lst) > 50:
                return f"[array of {len(lst)} elements: {str(lst[:5])}...]"
            return [_serialize_value(v) for v in lst]
        except Exception:
            pass
    return str(val)


def _write_json(filename, data):
    """Write a dict to a JSON file in OUTPUT_DIR."""
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _write_text(filename, text):
    """Write a string to a text file in OUTPUT_DIR."""
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w") as f:
        f.write(text)
