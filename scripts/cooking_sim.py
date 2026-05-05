"""
Cooking simulation script — animates burger patty color from raw to cooked.

Approach:
  - Creates an OmniPBR material with no texture
  - Animates diffuse_color_constant each frame from raw -> cooked
  - Simple uniform color change first, then can be extended with radial texture

Usage:
  1. Open the scene with the burger_patty prim at /World/burger_patty
  2. Run this script in Isaac Sim's Script Editor while the sim is playing
  3. Stop with:  _cooking_sub.unsubscribe()
"""

import omni.usd
import omni.kit.app
import carb
import numpy as np
from pxr import Gf, Sdf, UsdShade

# ── Configuration ─────────────────────────────────────────────────────────────

PATTY_MESH_PATH = "/World/burger_patty/Object_30/Object_12"
COOK_MATERIAL_PATH = "/World/Looks/cookingpattymaterial"
SHADER_PATH = COOK_MATERIAL_PATH + "/Shader"

# Colors (linear RGB, 0-1)
RAW_COLOR    = np.array([0.93, 0.09, 0.09])
COOKED_COLOR = np.array([0.25, 0.12, 0.05])

# Total cook time in seconds (one side)
COOK_TIME = 7.0


# ── Material setup ────────────────────────────────────────────────────────────

def _create_cooking_material(stage):
    """Create an OmniPBR material with just a diffuse color (no texture)."""
    mat_prim = stage.GetPrimAtPath(COOK_MATERIAL_PATH)

    if not mat_prim.IsValid():
        material = UsdShade.Material.Define(stage, COOK_MATERIAL_PATH)
        shader = UsdShade.Shader.Define(stage, SHADER_PATH)
        shader.SetSourceAsset("OmniPBR.mdl", "mdl")
        shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")

        shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*RAW_COLOR.tolist())
        )
        shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.9)

        material.CreateSurfaceOutput("mdl").ConnectToSource(
            UsdShade.ConnectableAPI(shader), "out"
        )
        carb.log_info(f"[cooking_sim] Created material at {COOK_MATERIAL_PATH}")
    else:
        material = UsdShade.Material(mat_prim)

    return material


def _bind_material(stage, mesh_path, material):
    mesh_prim = stage.GetPrimAtPath(mesh_path)
    UsdShade.MaterialBindingAPI(mesh_prim).Bind(material)
    carb.log_info(f"[cooking_sim] Bound material to {mesh_path}")


def _set_color(stage, color):
    """Update the shader's diffuse_color_constant."""
    shader_prim = stage.GetPrimAtPath(SHADER_PATH)
    if not shader_prim.IsValid():
        return
    shader = UsdShade.Shader(shader_prim)
    color_input = shader.GetInput("diffuse_color_constant")
    if color_input:
        color_input.Set(Gf.Vec3f(*color.tolist()))


# ── Main cooking controller ───────────────────────────────────────────────────

class CookingSim:
    def __init__(self):
        self._timer = 0.0
        self._initialized = False
        self._done = False

    def _initialize(self, stage):
        carb.log_info("[cooking_sim] Initializing...")

        prim = stage.GetPrimAtPath(PATTY_MESH_PATH)
        if not prim.IsValid():
            carb.log_error(f"[cooking_sim] Mesh not found: {PATTY_MESH_PATH}")
            self._done = True
            return

        # Delete old material if it exists (clean slate)
        old_mat = stage.GetPrimAtPath(COOK_MATERIAL_PATH)
        if old_mat.IsValid():
            stage.RemovePrim(COOK_MATERIAL_PATH)
            carb.log_info("[cooking_sim] Removed old cooking material")

        material = _create_cooking_material(stage)
        _bind_material(stage, PATTY_MESH_PATH, material)
        _set_color(stage, RAW_COLOR)

        self._initialized = True
        carb.log_info(f"[cooking_sim] Ready! Cook time: {COOK_TIME}s")

    def step(self, dt):
        if self._done:
            return

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        if not self._initialized:
            self._initialize(stage)
            return

        self._timer += dt
        cook_progress = min(self._timer / COOK_TIME, 1.0)

        # Lerp from raw to cooked
        color = RAW_COLOR * (1.0 - cook_progress) + COOKED_COLOR * cook_progress
        _set_color(stage, color)

        # Log every 25%
        pct = int(cook_progress * 100)
        if pct % 25 == 0 and pct > 0:
            carb.log_info(f"[cooking_sim] {pct}% cooked")

        if cook_progress >= 1.0:
            carb.log_info("[cooking_sim] Cooking complete!")
            self._done = True


# ── Start ─────────────────────────────────────────────────────────────────────

_cook = CookingSim()

def _on_update(event):
    _cook.step(event.payload["dt"])

app = omni.kit.app.get_app()
_cooking_sub = app.get_update_event_stream().create_subscription_to_pop(
    _on_update, name="cooking_sim"
)

carb.log_info("[cooking_sim] Started. Stop with: _cooking_sub.unsubscribe()")
