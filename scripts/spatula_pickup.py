"""
Spatula pickup → settle at staging pose above grill.

Sequence (after Play, run from Script Editor):
  1. Disable robot_joint, set spatula kinematic so it stays put while approaching
  2. IK target → 0.2m above spatula
  3. IK target → spatula (descend, slow)
  4. Enable robot_joint, set spatula dynamic again (robot now holds it)
  5. IK target → +0.2m in world Y (lift spatula straight out of holder, slow)
  6. IK target → straight up to z=1.19 (clear of holder before traversing)
  7. IK target → staging pose over grill [1.3229, 0.38237, 1.19652]

Requires extension running with TOOL_CONNECTOR_PRIM_PATH = None
(arm follows ToolApproachTarget directly).

Stop mid-sequence with:
    _spatula_sub.unsubscribe()
"""

import omni.usd
import omni.kit.app
import carb
import numpy as np
from pxr import UsdGeom, Gf

# ── Prim paths ────────────────────────────────────────────────────────────────
SPATULA_PRIM         = "/World/spatula"
SPATULA_ROBOT_JOINT  = "/World/spatula/connector_piece_triplea_tool_part_spatula/robot_joint"
TARGET_PRIM_PATH     = "/World/ToolApproachTarget"

# ── Key world positions ───────────────────────────────────────────────────────
POS_SPATULA = np.array([1.24542, -0.54637, 0.84373])
POS_FINAL   = np.array([1.3229,   0.38237, 1.19652])

# ── Timing (seconds) ──────────────────────────────────────────────────────────
T_MOVE_FAST = 1.5
T_MOVE_SLOW = 2.5
T_ACTION    = 0.5


# ── Helpers ───────────────────────────────────────────────────────────────────
def _set_attr(stage, path, attr_name, value):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        carb.log_error(f"[spatula_pickup] Prim not found: {path}")
        return
    attr = prim.GetAttribute(attr_name)
    if attr:
        attr.Set(value)
    else:
        carb.log_warn(f"[spatula_pickup] Attribute {attr_name} not on {path}")


def _move_target(stage, pos):
    prim = stage.GetPrimAtPath(TARGET_PRIM_PATH)
    if not prim.IsValid():
        return
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(*pos.tolist()))
            return


SPATULA_JOINT_LOCAL_ROT1 = Gf.Quatf(0.0, 0.0, 1.0, 0.0)  # 180° about joint Y (≈ spatula parent Z)


def _release_to_holder(stage):
    # Park spatula in place: kinematic on, joint off. Robot is free of it.
    _set_attr(stage, SPATULA_ROBOT_JOINT, "physics:jointEnabled", False)
    _set_attr(stage, SPATULA_PRIM, "physics:kinematicEnabled", True)
    # Bake the 180° wrist/connector offset into the joint so engaging it
    # doesn't snap the spatula. Tune the axis if the snap is around X or Y.
    _set_attr(stage, SPATULA_ROBOT_JOINT, "physics:localRot1", SPATULA_JOINT_LOCAL_ROT1)
    carb.log_info("[spatula_pickup] Released spatula (joint off, kinematic on, localRot1 baked)")


def _grab_spatula(stage):
    # Hand off from kinematic anchor to fixed joint with the robot.
    _set_attr(stage, SPATULA_ROBOT_JOINT, "physics:jointEnabled", True)
    _set_attr(stage, SPATULA_PRIM, "physics:kinematicEnabled", False)
    carb.log_info("[spatula_pickup] Grabbed spatula (joint on, kinematic off)")


def _noop(stage):
    pass


# ── State machine ─────────────────────────────────────────────────────────────
p = POS_SPATULA
f = POS_FINAL

exit_pos     = p + [0, 0.2, 0]
lift_pos     = np.array([exit_pos[0], exit_pos[1], 1.19])
mid_waypoint = np.array([0.96067, -0.17247, 1.19652])

STATES = [
    ("init_release",          p + [0, 0, 0.2], _release_to_holder, T_ACTION),
    ("approach_above",        p + [0, 0, 0.2], _noop,              T_MOVE_FAST),
    ("descend_to_spatula",    p,               _noop,              T_MOVE_SLOW),
    ("grab",                  p,               _grab_spatula,      T_ACTION),
    ("exit_y_positive",       exit_pos,        _noop,              T_MOVE_SLOW),
    ("lift_straight_up",      lift_pos,        _noop,              T_MOVE_FAST),
    ("traverse_mid_waypoint", mid_waypoint,    _noop,              T_MOVE_FAST),
    ("move_to_final",         f,               _noop,              T_MOVE_FAST),
    ("hold_final",            f,               _noop,              0.0),
]


def _smoothstep(t):
    # 3t^2 - 2t^3, ease-in/ease-out so the robot accelerates and decelerates
    # smoothly instead of jerking at the start/end of each segment.
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


class SpatulaPickup:
    def __init__(self):
        self._state_idx = 0
        self._timer = 0.0
        self._entered = False
        self._done = False
        # Track the last target so each segment lerps from there to the next
        # waypoint over `wait` seconds. Initialised on first step.
        self._segment_start = None

    def step(self, dt):
        if self._done:
            return

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        if self._state_idx >= len(STATES):
            carb.log_info("[spatula_pickup] Sequence complete — holding final pose.")
            self._done = True
            return

        label, target_pos, action_fn, wait = STATES[self._state_idx]
        target_pos = np.asarray(target_pos, dtype=np.float64)

        if not self._entered:
            carb.log_info(f"[spatula_pickup] -> {label}")
            if self._segment_start is None:
                self._segment_start = target_pos.copy()
            action_fn(stage)
            self._timer = 0.0
            self._entered = True

        self._timer += dt

        if wait <= 0.0:
            _move_target(stage, target_pos)
            self._segment_start = target_pos.copy()
            self._state_idx += 1
            self._entered = False
            return

        alpha = _smoothstep(self._timer / wait)
        interp = self._segment_start + (target_pos - self._segment_start) * alpha
        _move_target(stage, interp)

        if self._timer >= wait:
            self._segment_start = target_pos.copy()
            self._state_idx += 1
            self._entered = False


_test = SpatulaPickup()

def _on_update(event):
    _test.step(event.payload["dt"])

app = omni.kit.app.get_app()
_spatula_sub = app.get_update_event_stream().create_subscription_to_pop(
    _on_update, name="spatula_pickup"
)

carb.log_info("[spatula_pickup] Started. Stop with: _spatula_sub.unsubscribe()")
