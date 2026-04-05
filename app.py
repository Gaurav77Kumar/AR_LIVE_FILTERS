import cv2
import numpy as np
import os
import av
import time
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# --- ROBUST MEDIAPIPE LOADER ---
MEDIAPIPE_AVAILABLE = False
mp = None
HAND_LANDMARKER_TASK_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    print("WARNING: MediaPipe failed to load on this server:", e)


def create_hand_tracker():
    if not MEDIAPIPE_AVAILABLE:
        return None, None

    hands_module = getattr(getattr(mp, "solutions", None), "hands", None)
    if hands_module is not None:
        return "solutions", hands_module

    try:
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        if not os.path.exists(HAND_LANDMARKER_TASK_PATH):
            raise FileNotFoundError(f"Missing hand landmark model: {HAND_LANDMARKER_TASK_PATH}")

        base_options = mp_tasks.BaseOptions(model_asset_path=HAND_LANDMARKER_TASK_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        detector = vision.HandLandmarker.create_from_options(options)
        return "tasks", detector
    except Exception:
        return None, None

# ─────────────────────────────────────────────
# Filter Functions
# ─────────────────────────────────────────────
def anime_filter(frame):
    smooth = frame.copy()
    for _ in range(3):
        smooth = cv2.bilateralFilter(smooth, 9, 75, 75)
    div = 32
    quantized = (smooth // div) * div + div // 2
    hsv = cv2.cvtColor(quantized, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255).astype(np.uint8)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255).astype(np.uint8)
    vivid = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(vivid, edges_bgr)

def xray_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(inverted)
    edges = cv2.Canny(gray, 30, 100)
    edges = cv2.dilate(edges, None, iterations=1)
    result = cv2.addWeighted(enhanced, 0.8, edges, 0.5, 0)
    colored = np.zeros_like(frame)
    colored[:, :, 0] = np.clip(result * 1.0, 0, 255).astype(np.uint8)
    colored[:, :, 1] = np.clip(result * 0.8, 0, 255).astype(np.uint8)
    colored[:, :, 2] = np.clip(result * 0.3, 0, 255).astype(np.uint8)
    bright = cv2.GaussianBlur(result, (15, 15), 0)
    glow = np.zeros_like(frame)
    glow[:, :, 0] = np.clip(bright * 0.3, 0, 255).astype(np.uint8)
    glow[:, :, 1] = np.clip(bright * 0.6, 0, 255).astype(np.uint8)
    glow[:, :, 2] = np.clip(bright * 0.1, 0, 255).astype(np.uint8)
    return cv2.add(colored, glow)

def thermal_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

def neon_edges_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges1 = cv2.Canny(gray, 50, 150)
    edges2 = cv2.Canny(gray, 20, 80)
    edges1 = cv2.dilate(edges1, None, iterations=1)
    edges2 = cv2.dilate(edges2, None, iterations=1)
    result = np.zeros_like(frame)
    result[:, :, 0] = np.clip(edges1.astype(np.int16) * 1.0, 0, 255).astype(np.uint8)
    result[:, :, 1] = np.clip(edges1.astype(np.int16) * 0.9, 0, 255).astype(np.uint8)
    result[:, :, 2] = np.clip(result[:, :, 2].astype(np.int16) + edges2.astype(np.int16) * 0.8, 0, 255).astype(np.uint8)
    result[:, :, 0] = np.clip(result[:, :, 0].astype(np.int16) + edges2.astype(np.int16) * 0.3, 0, 255).astype(np.uint8)
    glow = cv2.GaussianBlur(result, (9, 9), 0)
    result = cv2.add(result, glow)
    dark_bg = cv2.convertScaleAbs(frame, alpha=0.1, beta=0)
    return cv2.add(result, dark_bg)

def pencil_sketch_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, cv2.bitwise_not(blur), scale=256)
    colored = np.zeros_like(frame)
    colored[:, :, 0] = np.clip(sketch * 0.85, 0, 255).astype(np.uint8)
    colored[:, :, 1] = np.clip(sketch * 0.9, 0, 255).astype(np.uint8)
    colored[:, :, 2] = np.clip(sketch * 0.95, 0, 255).astype(np.uint8)
    return colored

def oil_painting_filter(frame):
    result = frame.copy()
    for _ in range(4):
        result = cv2.bilateralFilter(result, 9, 75, 75)
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255).astype(np.uint8)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(result, edges_3ch)

def pixel_art_filter(frame):
    pixel_size = 8
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
    div = 48
    quantized = (small // div) * div + div // 2
    return cv2.resize(quantized, (w, h), interpolation=cv2.INTER_NEAREST)

def glitch_filter(frame):
    h, w = frame.shape[:2]
    result = frame.copy()
    shift_r = np.random.randint(-15, 15)
    shift_b = np.random.randint(-15, 15)
    result[:, :, 2] = np.roll(frame[:, :, 2], shift_r, axis=1)
    result[:, :, 0] = np.roll(frame[:, :, 0], shift_b, axis=1)
    for _ in range(5):
        y = np.random.randint(0, h)
        thickness = np.random.randint(1, 4)
        shift = np.random.randint(-30, 30)
        result[y:y+thickness] = np.roll(result[y:y+thickness], shift, axis=1)
    if np.random.random() < 0.3:
        bx = np.random.randint(0, max(1, w - 50))
        by = np.random.randint(0, max(1, h - 20))
        bw = np.random.randint(30, 100)
        bh = np.random.randint(5, 20)
        if bx+bw <= w and by+bh <= h:
            result[by:by+bh, bx:bx+bw] = result[by:by+bh, bx:bx+bw][:, :, ::-1]
    return result

FILTERS = {
    "ANIME / CARTOON":  anime_filter,
    "X-RAY":            xray_filter,
    "THERMAL VISION":   thermal_filter,
    "NEON EDGES":       neon_edges_filter,
    "PENCIL SKETCH":    pencil_sketch_filter,
    "OIL PAINTING":     oil_painting_filter,
    "PIXEL ART / 8-BIT": pixel_art_filter,
    "GLITCH":           glitch_filter,
}
FILTER_NAMES = list(FILTERS.keys())
CYCLE_DURATION = 6  # seconds per filter

# ─────────────────────────────────────────────
# Overlay helpers
# ─────────────────────────────────────────────

# Accent colours per filter (BGR)
FILTER_COLORS = {
    "ANIME / CARTOON":   (255, 100, 200),
    "X-RAY":             (60,  220, 255),
    "THERMAL VISION":    (50,  220,  50),
    "NEON EDGES":        (255, 255,   0),
    "PENCIL SKETCH":     (200, 200, 200),
    "OIL PAINTING":      (100, 200, 255),
    "PIXEL ART / 8-BIT": (255, 180,  50),
    "GLITCH":            (180,  50, 255),
}

def draw_hud(frame, filter_name, elapsed, total, idx, count):
    """
    Draws a clean, minimal HUD onto the frame:
      • Bottom bar with filter name + progress bar
      • Small index badge top-right (e.g. 3/8)
    """
    h, w = frame.shape[:2]
    accent = FILTER_COLORS.get(filter_name, (0, 255, 200))

    # ── Bottom dark panel ──────────────────────────────────────────────────
    bar_h = 54
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # ── Progress bar ──────────────────────────────────────────────────────
    progress = min(elapsed / total, 1.0)
    bar_y   = h - 6
    bar_thick = 5
    # background track
    cv2.line(frame, (0, bar_y), (w, bar_y), (60, 60, 60), bar_thick)
    # filled portion
    cv2.line(frame, (0, bar_y), (int(w * progress), bar_y), accent, bar_thick)

    # ── Filter name text ─────────────────────────────────────────────────
    font      = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.65
    thickness  = 1
    text_x     = 14
    text_y     = h - bar_h + 32
    # subtle shadow
    cv2.putText(frame, filter_name, (text_x + 1, text_y + 1), font, font_scale,
                (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, filter_name, (text_x, text_y), font, font_scale,
                accent, thickness, cv2.LINE_AA)

    # ── Countdown (right side of bar) ───────────────────────────────────
    secs_left = max(0, int(np.ceil(total - elapsed)))
    countdown_str = f"{secs_left}s"
    (tw, _), _ = cv2.getTextSize(countdown_str, font, 0.55, 1)
    cv2.putText(frame, countdown_str, (w - tw - 14, text_y), font, 0.55,
                (180, 180, 180), 1, cv2.LINE_AA)

    # ── Index badge top-right (e.g. "3 / 8") ───────────────────────────
    badge_str = f"{idx + 1} / {count}"
    (bw, bh), _ = cv2.getTextSize(badge_str, font, 0.5, 1)
    bx, by = w - bw - 12, 24
    # pill background
    cv2.rectangle(frame, (bx - 8, by - bh - 4), (bx + bw + 4, by + 6),
                  (20, 20, 20), -1)
    cv2.rectangle(frame, (bx - 8, by - bh - 4), (bx + bw + 4, by + 6),
                  accent, 1)
    cv2.putText(frame, badge_str, (bx, by), font, 0.5, accent, 1, cv2.LINE_AA)

    # ── Flash frame on filter switch (first 0.35 s) ─────────────────────
    if elapsed < 0.35:
        flash = frame.copy()
        alpha = 0.35 * (1.0 - elapsed / 0.35)
        cv2.rectangle(flash, (0, 0), (w, h), accent, -1)
        cv2.addWeighted(flash, alpha, frame, 1 - alpha, 0, frame)

    return frame


def order_quad(pts):
    pts = sorted(pts, key=lambda p: p[1])
    top_two = sorted(pts[:2], key=lambda p: p[0])
    bot_two = sorted(pts[2:], key=lambda p: p[0])
    return [top_two[0], top_two[1], bot_two[1], bot_two[0]]


# ─────────────────────────────────────────────
# Streamlit WebRTC Processor
# ─────────────────────────────────────────────
class LiveFilterProcessor(VideoProcessorBase):
    def __init__(self):
        self.filter_type   = FILTER_NAMES[0]
        self.enable_region = False
        self.auto_cycle    = False          # ← new flag
        self._cycle_start  = time.time()   # ← timer origin
        self._cycle_idx    = 0             # ← current filter index
        self.hand_tracker_mode = None
        self.hand_detector = None

        if MEDIAPIPE_AVAILABLE:
            try:
                self.hand_tracker_mode, self.hand_detector = create_hand_tracker()
                if self.hand_detector is None:
                    raise AttributeError("module 'mediapipe' has no accessible hands tracker")
            except Exception as e:
                print("Failed to initialize MediaPipe hands:", e)
                self.hand_detector = None
                self.hand_tracker_mode = None

    # ── called by WebRTC on every frame ─────────────────────────────────
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        # ── Determine active filter ──────────────────────────────────────
        if self.auto_cycle:
            now     = time.time()
            elapsed = now - self._cycle_start

            # Advance to next filter if 6 s have passed
            if elapsed >= CYCLE_DURATION:
                self._cycle_idx   = (self._cycle_idx + 1) % len(FILTER_NAMES)
                self._cycle_start = now
                elapsed           = 0.0

            active_name = FILTER_NAMES[self._cycle_idx]
        else:
            active_name = self.filter_type
            elapsed     = 0.0          # not shown in manual mode

        # ── Apply filter ─────────────────────────────────────────────────
        try:
            filtered = FILTERS[active_name](img)
        except Exception:
            filtered = img.copy()

        # ── Region Magic (hand bounding box) ────────────────────────────
        active_quad = None
        if self.enable_region and self.hand_detector:
            try:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if self.hand_tracker_mode == "solutions":
                    results = self.hand_detector.process(img_rgb)
                    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 2:
                        finger_pts = []
                        for hand_lm in results.multi_hand_landmarks[:2]:
                            finger_pts.append((int(hand_lm.landmark[4].x * w), int(hand_lm.landmark[4].y * h)))
                            finger_pts.append((int(hand_lm.landmark[8].x * w), int(hand_lm.landmark[8].y * h)))
                        active_quad = order_quad(finger_pts)
                elif self.hand_tracker_mode == "tasks":
                    timestamp_ms = int(time.time() * 1000)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                    results = self.hand_detector.detect_for_video(mp_image, timestamp_ms)
                    if results.hand_landmarks and len(results.hand_landmarks) >= 2:
                        finger_pts = []
                        for hand_landmarks in results.hand_landmarks[:2]:
                            thumb_tip = hand_landmarks[4]
                            index_tip = hand_landmarks[8]
                            finger_pts.append((int(thumb_tip.x * w), int(thumb_tip.y * h)))
                            finger_pts.append((int(index_tip.x * w), int(index_tip.y * h)))
                        active_quad = order_quad(finger_pts)
            except Exception:
                pass

        if active_quad is not None:
            quad_np    = np.array(active_quad, dtype=np.int32)
            mask       = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, quad_np, 255)
            mask_3ch   = cv2.merge([mask, mask, mask])
            inv_mask   = cv2.bitwise_not(mask_3ch)
            dark_bg    = cv2.convertScaleAbs(img, alpha=0.35, beta=5)
            result     = cv2.bitwise_and(filtered, mask_3ch) + cv2.bitwise_and(dark_bg, inv_mask)
            cv2.polylines(result, [quad_np], True, (0, 255, 255), 3, cv2.LINE_AA)
        else:
            result = filtered

        # ── HUD overlay (only in auto-cycle mode) ────────────────────────
        if self.auto_cycle:
            result = draw_hud(
                result,
                active_name,
                elapsed,
                CYCLE_DURATION,
                self._cycle_idx,
                len(FILTER_NAMES),
            )

        return av.VideoFrame.from_ndarray(result, format="bgr24")


# ─────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="AR Live Filters", layout="centered")

st.title("🎨 AR Live Video Filters")
st.markdown(
    "Live Python OpenCV filters running in your browser via **Streamlit WebRTC**. "
    "Toggle **Auto-Cycle** to tour every filter for 6 seconds each."
)

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    filter_choice = st.selectbox(
        "Select Filter Effect",
        FILTER_NAMES,
        disabled=False,    # stays enabled; just ignored while auto-cycle is on
    )

with col2:
    auto_cycle = st.toggle("🔄 Auto-Cycle (6 s)", value=False)
    if auto_cycle:
        st.caption("Cycling all 8 filters ↻")

with col3:
    st.markdown("<br/>", unsafe_allow_html=True)
    if MEDIAPIPE_AVAILABLE:
        region_toggle = st.checkbox("🔮 Magic Region")
        st.caption("Show 2 hands to create a portal!")
    else:
        st.error("Magic Region: disabled (MediaPipe driver conflict).")
        region_toggle = False

# ── If auto-cycle just toggled ON, show a quick legend ──────────────────
if auto_cycle:
    with st.expander("📋 Filter Schedule", expanded=False):
        for i, name in enumerate(FILTER_NAMES):
            st.markdown(f"**{i+1}.** {name} — *{CYCLE_DURATION} s*")

ctx = webrtc_streamer(
    key="live-filters",
    video_processor_factory=LiveFilterProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)

if ctx.video_processor:
    ctx.video_processor.filter_type   = filter_choice
    ctx.video_processor.enable_region = region_toggle
    ctx.video_processor.auto_cycle    = auto_cycle

    # Reset cycle timer whenever auto-cycle is toggled on so it starts cleanly
    if auto_cycle:
        ctx.video_processor._cycle_start = time.time()
        ctx.video_processor._cycle_idx   = 0