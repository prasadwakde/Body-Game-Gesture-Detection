"""
Temple Run Gesture Controller (Rule-based, 4 gestures)

Gestures:
- jump  -> press "up"    (both wrists above shoulders)
- duck  -> press "down"  (torso shrinks vs standing reference)
- left  -> press "left"  (shoulders center left of hips center)
- right -> press "right" (shoulders center right of hips center)

- EMA smoothing
- per-gesture cooldown
"""

from __future__ import annotations

import time
import csv
from dataclasses import dataclass
from collections import deque, Counter, defaultdict
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
import pyautogui

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

@dataclass(frozen=True)
class Config:
    # Smoothing & debounce
    ema_alpha: float = 0.35
    cooldown_s: float = 0.8
    show_debug: bool = True

    hands_above_shoulders_delta: float = 0.02
    crouch_torso_ratio: float = 0.62
    lean_thresh: float = 0.06
    require_both_hands_for_jump: bool = True

    # Standing calibration
    auto_calibrate_standing: bool = True
    calib_frames: int = 60

    # Evaluation mode
    eval_mode: bool = False
    step_duration: float = 2.0

    # Scripted eval sequence (5 times each)
    scripted_gestures: Tuple[str, ...] = (
        ("jump",) * 5 + ("duck",) * 5 + ("left",) * 5 + ("right",) * 5
    )

    gesture_events_csv: str = "gesture_events.csv"
    eval_ground_truth_csv: str = "eval_ground_truth.csv"
    eval_events_csv: str = "eval_events.csv"


class EMA:
    """Simple exponential moving average smoother."""
    def __init__(self, alpha: float, initial: Optional[float] = None) -> None:
        self.alpha = alpha
        self.value = initial

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value


def safe_press(key: str) -> None:
    """Press a key without crashing the loop."""
    try:
        pyautogui.press(key)
    except Exception:
        pass


def center_xy(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def visible_ok(*vis: float, thr: float = 0.5) -> bool:
    return all(v >= thr for v in vis)


@dataclass
class Event:
    gesture: str
    time_s: float
    lag_ms: float


@dataclass
class GroundTruthStep:
    index: int
    gesture: str
    t_start: float
    t_end: Optional[float] = None


class RuleBasedGestureDetector:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

        # EMA smoothers
        self.ema_dx = EMA(cfg.ema_alpha)
        self.ema_torso = EMA(cfg.ema_alpha)
        self.ema_wrist_left = EMA(cfg.ema_alpha)
        self.ema_wrist_right = EMA(cfg.ema_alpha)
        self.ema_shoulder_y = EMA(cfg.ema_alpha)

        # Standing calibration
        self._torso_ref_vals: Deque[float] = deque(maxlen=cfg.calib_frames)
        self.standing_torso_ref: Optional[float] = None

    def _update_standing_ref(self, torso_sm: float) -> None:
        """Collect torso samples until we can set standing reference."""
        if not self.cfg.auto_calibrate_standing or self.standing_torso_ref is not None:
            return

        self._torso_ref_vals.append(torso_sm)
        if len(self._torso_ref_vals) >= self.cfg.calib_frames:
            ref = float(np.median(self._torso_ref_vals))
            self.standing_torso_ref = max(0.15, min(ref, 0.6))

    def process_pose(
        self,
        pose_landmarks: mp.framework.formats.landmark_pb2.NormalizedLandmarkList,
    ) -> Optional[dict]:
        """
        Extract features from pose landmarks and return smoothed values.
        Returns None if visibility is insufficient.
        """
        lms = pose_landmarks.landmark
        PL = mp.solutions.holistic.PoseLandmark

        # landmark indices
        NOSE = PL.NOSE
        L_SHO, R_SHO = PL.LEFT_SHOULDER, PL.RIGHT_SHOULDER
        L_HIP, R_HIP = PL.LEFT_HIP, PL.RIGHT_HIP
        L_WR, R_WR = PL.LEFT_WRIST, PL.RIGHT_WRIST

        def xyv(idx):
            lm = lms[idx]
            return lm.x, lm.y, lm.visibility

        nose = xyv(NOSE)
        lsho = xyv(L_SHO)
        rsho = xyv(R_SHO)
        lhip = xyv(L_HIP)
        rhip = xyv(R_HIP)
        lwri = xyv(L_WR)
        rwri = xyv(R_WR)

        if not visible_ok(nose[2], lsho[2], rsho[2], lhip[2], rhip[2], lwri[2], rwri[2], thr=0.5):
            return None

        sh_cx, sh_cy = center_xy((lsho[0], lsho[1]), (rsho[0], rsho[1]))
        hip_cx, hip_cy = center_xy((lhip[0], lhip[1]), (rhip[0], rhip[1]))

        dx = sh_cx - hip_cx                     # lean indicator
        torso = abs(nose[1] - hip_cy)
        wl_y = lwri[1]
        wr_y = rwri[1]
        sh_y = (lsho[1] + rsho[1]) * 0.5        # shoulder height reference

        dx_sm = self.ema_dx.update(dx)
        torso_sm = self.ema_torso.update(torso)
        wl_sm = self.ema_wrist_left.update(wl_y)
        wr_sm = self.ema_wrist_right.update(wr_y)
        shy_sm = self.ema_shoulder_y.update(sh_y)

        self._update_standing_ref(torso_sm)

        return {
            "dx_sm": dx_sm,
            "torso_sm": torso_sm,
            "wl_sm": wl_sm,
            "wr_sm": wr_sm,
            "shy_sm": shy_sm,
            "sh_center": (sh_cx, sh_cy),
            "hip_center": (hip_cx, hip_cy),
        }

    def detect_gestures(self, feats: dict) -> Dict[str, bool]:
        """Return which gestures are currently active (rule-based)."""
        dx_sm = feats["dx_sm"]
        torso_sm = feats["torso_sm"]
        wl_sm = feats["wl_sm"]
        wr_sm = feats["wr_sm"]
        shy_sm = feats["shy_sm"]

        # Jump: wrists above shoulder line
        left_up = wl_sm < (shy_sm - self.cfg.hands_above_shoulders_delta)
        right_up = wr_sm < (shy_sm - self.cfg.hands_above_shoulders_delta)
        jump = (left_up and right_up) if self.cfg.require_both_hands_for_jump else (left_up or right_up)

        # Duck: torso shrink ratio vs standing reference
        duck = False
        ratio = None
        if self.standing_torso_ref is not None:
            ratio = torso_sm / self.standing_torso_ref
            duck = ratio < self.cfg.crouch_torso_ratio

        # Lean for left/right
        left = dx_sm <= -self.cfg.lean_thresh
        right = dx_sm >= self.cfg.lean_thresh

        return {
            "jump": jump,
            "duck": duck,
            "left": left,
            "right": right,
            "torso_ratio": ratio,
        }


class Evaluator:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.eval_start_time = time.time()
        self.current_step_index = -1
        self.steps: List[GroundTruthStep] = []

    def update_ground_truth(self, now: float) -> Optional[str]:
        """
        Returns the current scripted gesture instruction (e.g., "jump")
        or None if eval finished.
        """
        t_eval = now - self.eval_start_time
        step_index = int(t_eval // self.cfg.step_duration)

        if step_index >= len(self.cfg.scripted_gestures):
            if self.steps and self.steps[-1].t_end is None:
                self.steps[-1].t_end = now
            return None

        if step_index != self.current_step_index:
            if self.steps and self.steps[-1].t_end is None:
                self.steps[-1].t_end = now

            self.current_step_index = step_index
            gt = self.cfg.scripted_gestures[step_index]
            self.steps.append(GroundTruthStep(index=step_index, gesture=gt, t_start=now))
            print(f"[GT] Step {step_index}: please do {gt}")

        return self.cfg.scripted_gestures[step_index]

    def evaluate(self, events: List[Event]) -> None:
        if not self.steps:
            print("No ground truth steps recorded.")
            return

        correct = 0
        total = 0
        confusion = defaultdict(Counter)

        for step in self.steps:
            gt = step.gesture
            t0 = step.t_start
            t1 = step.t_end or (t0 + self.cfg.step_duration)

            step_events = [e for e in events if t0 <= e.time_s < t1]
            pred = step_events[0].gesture if step_events else "none"

            confusion[gt][pred] += 1
            total += 1
            if pred == gt:
                correct += 1

        acc = correct / total if total else 0.0
        print("\nEVALUATION RESULTS:")
        print(f"Total scripted gestures: {total}")
        print(f"Correctly detected:     {correct}")
        print(f"Accuracy:               {acc * 100:.1f}%\n")

        print("Confusion matrix (gt -> predicted counts):")
        for gt, row in confusion.items():
            print(f"{gt:>5} -> {dict(row)}")

    def save_csv(self, cfg: Config, events: List[Event]) -> None:
        with open(cfg.eval_ground_truth_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["index", "gesture", "t_start", "t_end"])
            for s in self.steps:
                w.writerow([s.index, s.gesture, s.t_start, s.t_end])

        with open(cfg.eval_events_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["gesture", "time", "lag_ms"])
            for e in events:
                w.writerow([e.gesture, e.time_s, e.lag_ms])

        print(f"Saved {cfg.eval_ground_truth_csv} and {cfg.eval_events_csv}")



def run(cfg: Config) -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    detector = RuleBasedGestureDetector(cfg)
    evaluator = Evaluator(cfg) if cfg.eval_mode else None

    last_trigger: Dict[str, float] = {"jump": 0.0, "duck": 0.0, "left": 0.0, "right": 0.0}
    events: List[Event] = []

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
        smooth_landmarks=True,
    ) as holistic:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_ts = time.time()
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)
            pose = res.pose_landmarks

            debug_text_lines: List[str] = []
            eval_instruction: Optional[str] = None
            eval_done = False

            now = time.time()

            if evaluator is not None:
                eval_instruction = evaluator.update_ground_truth(now)
                eval_done = eval_instruction is None

            if pose:
                feats = detector.process_pose(pose)

                if feats is not None:
                    states = detector.detect_gestures(feats)

                    def maybe_fire(gesture: str, key: str) -> None:
                        if now - last_trigger[gesture] < cfg.cooldown_s:
                            return
                        lag_ms = (time.time() - frame_ts) * 1000.0
                        safe_press(key)
                        last_trigger[gesture] = now
                        events.append(Event(gesture=gesture, time_s=now, lag_ms=lag_ms))
                        print(f"[EVENT] {gesture} | lag={lag_ms:.1f} ms")

                    if states["jump"]:
                        maybe_fire("jump", "up")

                    if states["duck"]:
                        maybe_fire("duck", "down")

                    if states["left"]:
                        maybe_fire("left", "left")
                    elif states["right"]:
                        maybe_fire("right", "right")

                    if cfg.show_debug:
                        mp_drawing.draw_landmarks(
                            frame,
                            res.pose_landmarks,
                            mp_holistic.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=2, circle_radius=2),
                            connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1),
                        )

                        sh_cx, sh_cy = feats["sh_center"]
                        hip_cx, hip_cy = feats["hip_center"]

                        sc = (int(sh_cx * w), int(sh_cy * h))
                        hc = (int(hip_cx * w), int(hip_cy * h))
                        cv2.circle(frame, sc, 6, (0, 255, 255), -1)
                        cv2.circle(frame, hc, 6, (255, 255, 0), -1)
                        cv2.line(frame, sc, hc, (200, 200, 50), 2)

                        yline = int(feats["shy_sm"] * h)
                        cv2.line(frame, (0, yline), (w, yline), (0, 120, 255), 1)

                        debug_text_lines.append(f"Lean dx: {feats['dx_sm']:+.3f} (thr {cfg.lean_thresh})")

                        if detector.standing_torso_ref is None:
                            debug_text_lines.append("Calibrating standing... hold upright")
                            debug_text_lines.append(f"Torso: {feats['torso_sm']:.3f}")
                        else:
                            ratio = states["torso_ratio"]
                            debug_text_lines.append(f"Torso ratio: {ratio:.3f} (duck<{cfg.crouch_torso_ratio})")

                        debug_text_lines.append(f"Wrists above: {states['jump']}")

            if evaluator is not None:
                if eval_done:
                    cv2.putText(
                        frame,
                        "EVAL DONE - PRESS 'q' TO EXIT",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )
                else:
                    cv2.rectangle(frame, (10, 10), (340, 80), (0, 0, 0), -1)
                    cv2.putText(
                        frame,
                        f"DO NOW: {eval_instruction.upper()}",
                        (20, 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 255),
                        2,
                    )

            if debug_text_lines:
                y = 24
                for line in debug_text_lines:
                    cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 220, 50), 2, cv2.LINE_AA)
                    y += 24

            cv2.imshow("Temple Run Gestures (q to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    if events:
        with open(cfg.gesture_events_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["gesture", "event_time", "lag_ms"])
            for e in events:
                w.writerow([e.gesture, e.time_s, e.lag_ms])
        print(f"Saved {cfg.gesture_events_csv}")

    if evaluator is not None:
        evaluator.evaluate(events)
        evaluator.save_csv(cfg, events)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run(Config())
