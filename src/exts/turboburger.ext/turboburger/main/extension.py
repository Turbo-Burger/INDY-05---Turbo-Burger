import os

import carb
import numpy as np
import omni.ext
import omni.timeline
import omni.usd
from pxr import Gf, UsdGeom

from .stage_dumper import dump_everything

# Isaac Sim extension modules (Articulation, XFormPrim, kinematics solvers)
# are imported lazily in _initialize_follow_stack() so that this extension
# can load before those dependencies are fully available at startup.


PROJECT_ROOT = "/home/michael/turbo burger"
LULA_CONFIG_DIR = os.path.join(PROJECT_ROOT, "assets", "ur5e", "rmpflow")

ROBOT_PRIM_PATH = "/World/ur5e"
TARGET_PRIM_PATH = "/World/ToolApproachTarget"
ROBOT_NAME = "UR5e"
USE_TARGET_ORIENTATION = False
FIXED_TARGET_EULER_XYZ = np.array([0.0, np.pi, 0.0], dtype=np.float64)
END_EFFECTOR_FRAME_CANDIDATES = ["tool0", "ee_link", "flange", "wrist_3_link"]
TARGET_POSITION = Gf.Vec3d(0.45, -0.25, 0.25)
TARGET_SCALE = Gf.Vec3f(0.04, 0.04, 0.04)
TARGET_COLOR = Gf.Vec3f(0.9, 0.15, 0.15)

# Path to the active tool's connector prim. Set to None to fall back to ToolApproachTarget.
#TOOL_CONNECTOR_PRIM_PATH = "/World/spatula/connector_piece_triplea_tool_part_spatula"
TOOL_CONNECTOR_PRIM_PATH = None

# Tool prims whose attach-to-robot joint should be cleared on Stop, so each Play
# starts with the tool sitting in its holder and the robot empty-handed.
TOOLS_TO_RESET_ON_STOP = [
    {
        "tool_prim": "/World/spatula",
        "robot_joint": "/World/spatula/connector_piece_triplea_tool_part_spatula/robot_joint",
    },
]

# Local offset from the IK EE frame (flange) to the robot connector coupling face, in meters.
# Derived from triplea_robot_part local transform under wrist_3_link.
FLANGE_TO_CONNECTOR_OFFSET = np.array([0.0, 0.0, 0.0], dtype=np.float64)

# ── Tool attachment joint offsets ─────────────────────────────────────────────
# Fixed joint between robot connector (body0) and tool connector (body1).
# Dialed in visually for the spatula.
#
# Body 0: /World/ur5e/wrist_3_link/triplea_robot_part
# Body 1: /World/spatula/connector_piece_triplea_tool_part_spatula
SPATULA_JOINT_LOCAL_POS_0 = np.array([0.2, -0.5, 0.2], dtype=np.float64)
SPATULA_JOINT_LOCAL_ROT_0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
SPATULA_JOINT_LOCAL_POS_1 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
SPATULA_JOINT_LOCAL_ROT_1 = np.array([0.0, 0.0, 0.0], dtype=np.float64)

# Body 0: /World/ur5e/wrist_3_link/triplea_robot_part
# Body 1: /World/smasher/tripleA_tool_part_smasher
SMASHER_JOINT_LOCAL_POS_0 = np.array([0.2, -0.5, 0.2], dtype=np.float64)
SMASHER_JOINT_LOCAL_ROT_0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
SMASHER_JOINT_LOCAL_POS_1 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
SMASHER_JOINT_LOCAL_ROT_1 = np.array([0.0, 0.0, 0.0], dtype=np.float64)


def _quat_from_euler_xyz(euler_xyz):
    roll, pitch, yaw = euler_xyz
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


def _pick_end_effector_frame(kinematics_solver):
    frame_names = list(kinematics_solver.get_all_frame_names())
    for candidate in END_EFFECTOR_FRAME_CANDIDATES:
        if candidate in frame_names:
            return candidate
    raise RuntimeError(
        f"No supported end-effector frame found. Candidates={END_EFFECTOR_FRAME_CANDIDATES}, "
        f"available={frame_names}"
    )


def _ensure_target_prim():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is open.")

    target_xform = UsdGeom.Xform.Define(stage, TARGET_PRIM_PATH)
    ordered_ops = target_xform.GetOrderedXformOps()
    translate_op = None
    for op in ordered_ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if translate_op is None:
        translate_op = target_xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)

    if not ordered_ops:
        translate_op.Set(TARGET_POSITION)

    target_cube = UsdGeom.Cube.Define(stage, f"{TARGET_PRIM_PATH}/marker")
    target_cube.CreateSizeAttr(1.0)

    scale_ops = target_cube.GetOrderedXformOps()
    scale_op = None
    for op in scale_ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_op = op
            break
    if scale_op is None:
        scale_op = target_cube.AddScaleOp()
    scale_op.Set(TARGET_SCALE)

    target_cube.CreateDisplayColorAttr([TARGET_COLOR])


class TurboBurgerExtension(omni.ext.IExt):
    def on_startup(self, ext_id):
        self._ext_id = ext_id
        self._robot = None
        self._target = None
        self._tool_connector = None
        self._kinematics_solver = None
        self._articulation_solver = None
        self._update_sub = None
        self._timeline_sub = None
        self._stage_event_sub = None

        # ── Timeline events (play/stop → start/stop IK follow loop) ──
        timeline = omni.timeline.get_timeline_interface()
        stream = timeline.get_timeline_event_stream()
        self._timeline_sub = stream.create_subscription_to_pop(self._on_timeline_event)

        # ── Stage events (save → dump stage for LLM context) ──
        stage_event_stream = omni.usd.get_context().get_stage_event_stream()
        self._stage_event_sub = stage_event_stream.create_subscription_to_pop(
            self._on_stage_event, name="turboburger_stage_dumper"
        )

        carb.log_info("Turbo Burger Control extension started.")
        carb.log_info("A target prim is available at /World/ToolApproachTarget.")
        carb.log_info("Press Play to start UR5e target following.")
        carb.log_info("Stage will auto-dump to LLM context files on every save.")

    def on_shutdown(self):
        self._stop_following()
        self._timeline_sub = None
        self._stage_event_sub = None
        carb.log_info("Turbo Burger Control extension stopped.")

    # ── Stage event handler (save → dump) ──────────────────────────────

    def _on_stage_event(self, event):
        if event.type == int(omni.usd.StageEventType.SAVED):
            carb.log_info("[stage_dumper] Save detected — dumping stage for LLM context...")
            try:
                dump_everything()
            except Exception as exc:
                carb.log_error(f"[stage_dumper] Failed to dump stage: {exc}")

    # ── Timeline event handler (play/stop) ─────────────────────────────

    def _on_timeline_event(self, event):
        event_type = int(event.type)
        timeline = omni.timeline.TimelineEventType

        if event_type == int(timeline.PLAY):
            self._start_following()
        elif event_type == int(timeline.STOP):
            self._stop_following()
            self._reset_tool_attachments()

    def _reset_tool_attachments(self):
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        for entry in TOOLS_TO_RESET_ON_STOP:
            tool_prim = stage.GetPrimAtPath(entry["tool_prim"])
            joint_prim = stage.GetPrimAtPath(entry["robot_joint"])
            if joint_prim and joint_prim.IsValid():
                attr = joint_prim.GetAttribute("physics:jointEnabled")
                if attr:
                    attr.Set(False)
            if tool_prim and tool_prim.IsValid():
                attr = tool_prim.GetAttribute("physics:kinematicEnabled")
                if attr:
                    attr.Set(True)
        carb.log_info("[turboburger] Tool attachments reset on Stop.")

    def _start_following(self):
        if self._update_sub is not None:
            return

        try:
            self._initialize_follow_stack()
        except Exception as exc:
            carb.log_error(f"Failed to initialize UR5e follow stack: {exc}")
            return

        app = omni.kit.app.get_app()
        self._update_sub = app.get_update_event_stream().create_subscription_to_pop(
            self._on_update, name="turboburger_follow_target"
        )
        carb.log_info("UR5e follow-target loop started.")

    def _stop_following(self):
        if self._update_sub is not None:
            self._update_sub.unsubscribe()
            self._update_sub = None

        self._robot = None
        self._target = None
        self._kinematics_solver = None
        self._articulation_solver = None

    def _initialize_follow_stack(self):
        # Lazy-import Isaac Sim modules — they come from extensions that may
        # not be loaded when our extension first initialises at app startup.
        try:
            from isaacsim.core.prims import SingleArticulation as Articulation
            from isaacsim.core.prims import SingleXFormPrim as XFormPrim
            from isaacsim.robot_motion.motion_generation import (
                ArticulationKinematicsSolver,
                LulaKinematicsSolver,
            )
        except ImportError:
            from omni.isaac.core.articulations import Articulation
            from omni.isaac.core.prims import XFormPrim
            from omni.isaac.motion_generation import (
                ArticulationKinematicsSolver,
                LulaKinematicsSolver,
            )

        _ensure_target_prim()

        self._robot = Articulation(ROBOT_PRIM_PATH)
        self._robot.initialize()

        self._target = XFormPrim(TARGET_PRIM_PATH)

        if TOOL_CONNECTOR_PRIM_PATH is not None:
            self._tool_connector = XFormPrim(TOOL_CONNECTOR_PRIM_PATH)
            carb.log_info(f"Tracking tool connector: {TOOL_CONNECTOR_PRIM_PATH}")
        else:
            self._tool_connector = None
            carb.log_info("No tool connector set — falling back to ToolApproachTarget.")

        self._kinematics_solver = LulaKinematicsSolver(
            robot_description_path=os.path.join(LULA_CONFIG_DIR, "ur5e_robot_description.yaml"),
            urdf_path=os.path.join(LULA_CONFIG_DIR, "..", "ur5e.urdf"),
        )
        end_effector_frame = _pick_end_effector_frame(self._kinematics_solver)

        self._articulation_solver = ArticulationKinematicsSolver(
            self._robot, self._kinematics_solver, end_effector_frame
        )
        carb.log_info(f"Using end-effector frame: {end_effector_frame}")

    def _on_update(self, event):
        try:
            robot_base_translation, robot_base_orientation = self._robot.get_world_pose()
        except Exception:
            # PhysX backend not ready yet — skip this frame and retry next tick.
            return

        try:
            if self._tool_connector is not None:
                desired_connector_pos, _ = self._tool_connector.get_world_pose()
            else:
                desired_connector_pos, _ = self._target.get_world_pose()

            if USE_TARGET_ORIENTATION:
                _, target_orientation = self._target.get_world_pose()
            else:
                target_orientation = _quat_from_euler_xyz(FIXED_TARGET_EULER_XYZ)

            # Rotate the local flange→connector offset into world space using the
            # target orientation, then back-calculate where the flange needs to be.
            w, x, y, z = target_orientation
            R = np.array([
                [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
                [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
                [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
            ], dtype=np.float64)
            connector_offset_world = R @ FLANGE_TO_CONNECTOR_OFFSET
            flange_target_position = desired_connector_pos - connector_offset_world

            self._kinematics_solver.set_robot_base_pose(
                robot_base_translation, robot_base_orientation
            )

            action, success = self._articulation_solver.compute_inverse_kinematics(
                flange_target_position, target_orientation
            )

            if success:
                self._robot.apply_action(action)
            else:
                carb.log_warn(
                    f"IK did not converge for target position {np.round(flange_target_position, 4)}"
                )
        except Exception as exc:
            carb.log_error(f"UR5e follow-target loop stopped: {exc}")
            self._stop_following()
