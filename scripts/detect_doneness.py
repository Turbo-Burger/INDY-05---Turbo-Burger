"""
Doneness detection — watches the burger patty cook and flags when it's done.

Approach:
  - Creates a Replicator render product on /World/orderUpCamera
  - Each N app updates, pulls the RGB frame
  - Masks to patty-like pixels via an HSV color band (red → orange-brown)
  - Builds a Value (brightness) histogram on the masked pixels and tracks mean H/S/V
  - Declares DONE when mean V drops below DONE_V_THRESHOLD for SUSTAIN_CHECKS
    consecutive checks (debounce against single-frame spikes)

Pairs with cooking_sim.py — that script lerps the patty diffuse from
RAW (0.93, 0.09, 0.09) → COOKED (0.25, 0.12, 0.05) over 7s. After sRGB
encoding the camera sees roughly V≈0.95 raw → V≈0.55 cooked, S drops from
~0.67 → ~0.54, and hue drifts from pure red (0°) into orange-brown (~25°).

Usage (Script Editor):
  1. Make sure the scene is playing and cooking_sim.py is animating the patty.
  2. Run this script — it prints histogram stats every check.
  3. Stop with: _doneness_sub.unsubscribe()
"""

import omni.kit.app
import omni.replicator.core as rep
import carb
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────

CAMERA_PRIM_PATH = "/World/orderUpCamera"
RESOLUTION       = (640, 480)

# How often to sample (in app update ticks). Every tick is overkill for a 7s
# cook and slows the viewport — sample a few times per second.
CHECK_EVERY_N_TICKS = 6

# HSV patty mask. Channels are 0-1 floats here.
# Hue 0.0-0.10 ≈ 0°-36° (red → orange-brown). Saturation/Value floors keep us
# off the dark grill and washed-out background.
HUE_RANGE = (0.0, 0.10)
SAT_MIN   = 0.20
VAL_MIN   = 0.10

# Doneness criterion on the masked region.
DONE_V_THRESHOLD = 0.65   # mean V must drop below this
SUSTAIN_CHECKS   = 3      # for this many consecutive checks
MIN_PATTY_PIXELS = 200    # require at least this many masked pixels to trust a reading

# Histogram bins for the printout.
V_HIST_BINS = 8


# ── HSV conversion (no cv2 dependency) ────────────────────────────────────────

def rgb_to_hsv(rgb):
    """rgb: (H, W, 3) float in [0, 1]. Returns (H, W, 3) HSV in [0, 1]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = np.max(rgb, axis=-1)
    cmin = np.min(rgb, axis=-1)
    delta = cmax - cmin

    h = np.zeros_like(cmax)
    nz = delta > 1e-8
    rmax = nz & (cmax == r)
    gmax = nz & (cmax == g) & ~rmax
    bmax = nz & (cmax == b) & ~rmax & ~gmax

    h[rmax] = ((g[rmax] - b[rmax]) / delta[rmax]) % 6.0
    h[gmax] = ((b[gmax] - r[gmax]) / delta[gmax]) + 2.0
    h[bmax] = ((r[bmax] - g[bmax]) / delta[bmax]) + 4.0
    h = (h / 6.0) % 1.0

    s = np.where(cmax > 1e-8, delta / np.maximum(cmax, 1e-8), 0.0)
    v = cmax
    return np.stack([h, s, v], axis=-1)


# ── Detector ──────────────────────────────────────────────────────────────────

class DonenessDetector:
    def __init__(self):
        self._render_product = rep.create.render_product(CAMERA_PRIM_PATH, RESOLUTION)
        self._rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        self._rgb.attach([self._render_product])
        carb.log_info(f"[doneness] Render product attached on {CAMERA_PRIM_PATH} @ {RESOLUTION}")

        self._tick = 0
        self._consecutive_done = 0
        self._done = False

    def shutdown(self):
        try:
            self._rgb.detach([self._render_product])
            self._render_product.destroy()
        except Exception as e:
            carb.log_warn(f"[doneness] cleanup: {e}")

    def step(self, _dt):
        if self._done:
            return
        self._tick += 1
        if self._tick % CHECK_EVERY_N_TICKS != 0:
            return

        frame = self._rgb.get_data()
        if frame is None or not hasattr(frame, "shape") or frame.size == 0:
            return  # first frames before the renderer has produced anything

        # Annotator returns uint8 RGBA (H, W, 4). Drop alpha and normalize.
        rgb = frame[..., :3].astype(np.float32) / 255.0
        hsv = rgb_to_hsv(rgb)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        mask = (h >= HUE_RANGE[0]) & (h <= HUE_RANGE[1]) & (s >= SAT_MIN) & (v >= VAL_MIN)
        n = int(mask.sum())
        if n < MIN_PATTY_PIXELS:
            carb.log_info(f"[doneness] tick {self._tick}: only {n} patty-like px — skipping")
            return

        mean_h = float(h[mask].mean())
        mean_s = float(s[mask].mean())
        mean_v = float(v[mask].mean())

        # Value histogram on patty pixels — useful for spotting bimodal cooking
        # (e.g. crust vs. raw center) once we move beyond a uniform color.
        hist, _ = np.histogram(v[mask], bins=V_HIST_BINS, range=(0.0, 1.0))
        hist_str = " ".join(f"{c:>4d}" for c in hist)

        carb.log_info(
            f"[doneness] tick {self._tick}: px={n:>6d}  "
            f"H={mean_h:.3f}  S={mean_s:.3f}  V={mean_v:.3f}  "
            f"V-hist=[{hist_str}]"
        )

        if mean_v < DONE_V_THRESHOLD:
            self._consecutive_done += 1
        else:
            self._consecutive_done = 0

        if self._consecutive_done >= SUSTAIN_CHECKS:
            self._done = True
            carb.log_warn(
                f"[doneness] *** PATTY DONE *** mean V={mean_v:.3f} "
                f"(threshold {DONE_V_THRESHOLD}, sustained {SUSTAIN_CHECKS} checks)"
            )


# ── Start ─────────────────────────────────────────────────────────────────────

_detector = DonenessDetector()

def _on_update(event):
    _detector.step(event.payload["dt"])

app = omni.kit.app.get_app()
_doneness_sub = app.get_update_event_stream().create_subscription_to_pop(
    _on_update, name="doneness_detector"
)

carb.log_info("[doneness] Started. Stop with: _doneness_sub.unsubscribe(); _detector.shutdown()")
