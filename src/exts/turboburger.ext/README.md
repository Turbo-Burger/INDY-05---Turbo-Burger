Turbo Burger Control Extension

This extension creates `/World/ToolApproachTarget` if it is missing and starts the UR5e follow-target loop when the Isaac Sim timeline enters Play.

Setup in Isaac Sim

1. Open `Window -> Extensions`.
2. Open the gear icon or extension settings area.
3. Add this folder as an extension search path:
   `/home/michael/turbo burger/exts`
4. Search for `Turbo Burger Control`.
5. Enable the extension.

Workflow

1. Open your burger stage.
2. Make sure the robot prim path is `/World/ur5e`.
3. Enable the extension once.
4. Press Play.
5. Move `/World/ToolApproachTarget` in the viewport. The robot will follow it automatically.

Important settings

- Robot path: `ROBOT_PRIM_PATH` in `turboburger/control/extension.py`
- Target path: `TARGET_PRIM_PATH` in `turboburger/control/extension.py`
- Fixed wrist orientation: `FIXED_TARGET_EULER_XYZ` in `turboburger/control/extension.py`
- If you want the tool to inherit the target's rotation, set `USE_TARGET_ORIENTATION = True`
