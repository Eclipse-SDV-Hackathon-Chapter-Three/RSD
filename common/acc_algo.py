# common/acc_algo.py — Road-aware ACC (statesub merged)
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np

# ====== base parameters (unchanged defaults) ======
BASE_TARGET_SPEED_KPH = 80.0
BASE_TARGET_SPEED_MPS = BASE_TARGET_SPEED_KPH / 3.6

BASE_TIME_HEADWAY = 1.8      # s
MIN_GAP           = 5.0      # m

KP_GAP    = 0.30
KV_SPEED  = 0.22
MAX_ACCEL = 1.2
MAX_DECEL = 5.0
THR_GAIN  = 0.40
BRK_GAIN  = 0.35

BASE_AEB_ENTER_TTC  = 3.0
BASE_AEB_EXIT_TTC   = 4.5
BASE_FOLLOW_ENTER_TTC = 8.0
BASE_FOLLOW_EXIT_TTC  = 10.0
AEB_ACTIVATION_SPEED_THRESHOLD = 2.8  # m/s
AEB_MIN_HOLD_SEC  = 0.8
MODE_COOLDOWN_SEC = 0.4

FOLLOW_MAX_DIST_M = 80.0

THR_RATE_UP    = 0.08
THR_RATE_DOWN  = 0.25
BRK_RATE_UP    = 0.25
BRK_RATE_DOWN  = 0.12

APPROACH_REL_GAIN = 0.06
TTC_SLOW_START    = 7.0
TTC_BIAS_GAIN     = 0.05

# ====== statesub.py merged: Δ and hysteresis ====== wer 1.0 , 0.8 / snow 2.5 2.3
DELTA = {
    "dry":  {"acc_headway_s": 0.0, "aeb_ttc_s": 0.0},
    "wet":  {"acc_headway_s": 2.5, "aeb_ttc_s": 2.0},
    "snow": {"acc_headway_s": 4.0, "aeb_ttc_s": 4.0},
    "icy":  {"acc_headway_s": 4.0, "aeb_ttc_s": 4.0},
}
CANON = {"dry":"dry","wet":"wet","snow":"snow","snowy":"snow","ice":"icy","icy":"icy","unknown":"unknown"}
PRIO  = {"dry":0,"wet":1,"snow":2,"icy":3,"unknown":-1}

MIN_HOLD_DEFAULT      = 0.60
UNKNOWN_GRACE_DEFAULT = 1.00
DWELL_IN  = {("dry","wet"):0.15, ("wet","snow"):0.20, ("snow","icy"):0.20,
             ("dry","snow"):0.25, ("wet","icy"):0.25, ("dry","icy"):0.30}
DWELL_OUT = {("wet","dry"):0.60, ("snow","wet"):0.80, ("icy","snow"):1.00,
             ("snow","dry"):1.00, ("icy","wet"):1.20, ("icy","dry"):1.50}

def canon_state(s):
    if s is None: return "unknown"
    s = str(s).strip().lower()
    return CANON.get(s, "unknown")

class HysteresisState:
    def __init__(self, init_state="dry",
                 min_hold=MIN_HOLD_DEFAULT,
                 unknown_grace=UNKNOWN_GRACE_DEFAULT):
        self.stable = canon_state(init_state)
        self.stable_since = time.time()
        self.candidate = None
        self.candidate_since = None
        self.min_hold = float(min_hold)
        self.unknown_grace = float(unknown_grace)

    def _needed_dwell(self, cur, cand):
        if cur == cand: return 0.0
        if PRIO.get(cand,-1) > PRIO.get(cur,-1):   # 위험 방향
            return DWELL_IN.get((cur,cand), 0.20)
        else:                                       # 안전 방향
            return DWELL_OUT.get((cur,cand), 0.80)

    def feed(self, raw_state: str) -> str:
        now = time.time()
        s = canon_state(raw_state)

        if s == "unknown" and (now - self.stable_since) < self.unknown_grace:
            return self.stable
        if s == "unknown":
            s = "dry"  # grace 초과 시 폴백

        if (now - self.stable_since) < self.min_hold:
            self.candidate, self.candidate_since = None, None
            return self.stable

        if s == self.stable:
            self.candidate, self.candidate_since = None, None
            return self.stable

        if self.candidate != s:
            self.candidate, self.candidate_since = s, now
            return self.stable

        need = self._needed_dwell(self.stable, self.candidate)
        have = now - (self.candidate_since or now)
        if have >= need:
            self.stable = self.candidate
            self.stable_since = now
            self.candidate, self.candidate_since = None, None
        return self.stable

# ====== utils ======
def _rate_limit(prev: float, desired: float, max_up: float, max_down: float) -> float:
    delta = desired - prev
    if delta >  max_up:   desired = prev + max_up
    if delta < -max_down: desired = prev - max_down
    return float(np.clip(desired, 0.0, 1.0))

def _acc_longitudinal_control(ego_v: float,
                              lead_v: float | None,
                              distance: float | None,
                              v_set_mps: float,
                              time_headway_s: float) -> tuple[float, float]:
    desired_gap = MIN_GAP + time_headway_s * ego_v
    gap_error   = (distance if distance is not None else float('inf')) - desired_gap
    speed_err   = (lead_v - ego_v) if (lead_v is not None) else (v_set_mps - ego_v)

    accel_cmd = KP_GAP * gap_error + KV_SPEED * speed_err
    accel_cmd = float(np.clip(accel_cmd, -MAX_DECEL, MAX_ACCEL))

    if accel_cmd >= 0.0:
        throttle, brake = min(1.0, THR_GAIN * accel_cmd), 0.0
    else:
        throttle, brake = 0.0, min(1.0, BRK_GAIN * (-accel_cmd))
    return float(throttle), float(brake)

# ====== I/O ======
@dataclass
class ACCInputs:
    distance: float | None
    rel_speed: float          # +면 접근
    ttc: float | None
    has_target: bool
    lead_speed_est: float | None = None

# ====== Controller ======
class ACCController:
    """
    Road-aware ACC controller.
    - set_road_state(raw) 로 도로상태를 공급하면 히스테리시스 + Δ 반영
    - step(sim_time, ego_speed_mps, acc) → (throttle, brake, mode)
    """
    def __init__(self, apply_hz: float = 20.0,
                 init_road_state: str = "dry",
                 min_hold: float = MIN_HOLD_DEFAULT,
                 unknown_grace: float = UNKNOWN_GRACE_DEFAULT,
                 target_speed_kph: float = BASE_TARGET_SPEED_KPH):
        self.apply_hz = max(1e-3, float(apply_hz))
        self.mode = "CRUISE"
        self.last_mode_change_time = -1e9
        self.safe_stop_locked = False
        self.last_thr = 0.0
        self.last_brk = 0.0

        # road-state & dynamic thresholds
        self._hyst = HysteresisState(init_state=init_road_state,
                                     min_hold=min_hold,
                                     unknown_grace=unknown_grace)
        self.road_state = self._hyst.stable
        self._apply_delta_from_state()

        self.v_set_mps = float(target_speed_kph) / 3.6

    # --- road state handling ---
    def _apply_delta_from_state(self):
        d = DELTA.get(self.road_state, DELTA["dry"])
        dh = float(d["acc_headway_s"])
        da = float(d["aeb_ttc_s"])

        # time headway ↑ on slippery
        self.time_headway_s = BASE_TIME_HEADWAY + dh

        # AEB thresholds ↑ (보수적으로)
        self.AEB_ENTER_TTC = BASE_AEB_ENTER_TTC + da
        self.AEB_EXIT_TTC  = BASE_AEB_EXIT_TTC  + da

        # FOLLOW TTC 게이트도 함께 ↑ (같은 Δ를 단순 가산)
        self.FOLLOW_ENTER_TTC = BASE_FOLLOW_ENTER_TTC + da
        self.FOLLOW_EXIT_TTC  = BASE_FOLLOW_EXIT_TTC  + da

    def set_road_state(self, raw_state: str):
        stable = self._hyst.feed(raw_state)
        if stable != self.road_state:
            self.road_state = stable
            self._apply_delta_from_state()

    # --- main step ---
    def step(self, sim_time: float, ego_speed_mps: float, acc: ACCInputs):
        distance = float(acc.distance) if (acc.distance is not None) else float('inf')
        ttc      = float(acc.ttc) if (acc.ttc is not None) else float('inf')
        rel_v    = float(acc.rel_speed)

        target_ready = (
            bool(acc.has_target)
            and (distance < FOLLOW_MAX_DIST_M)
            and np.isfinite(ttc)
            and (ttc < self.FOLLOW_ENTER_TTC)
        )

        tsc = sim_time - self.last_mode_change_time
        throttle_des = 0.0
        brake_des = 0.0

        # ---- state transitions ----
        if self.mode == "CRUISE":
            if target_ready and (tsc > MODE_COOLDOWN_SEC):
                self.mode, self.last_mode_change_time = "FOLLOW", sim_time
            if target_ready and (ttc < self.AEB_ENTER_TTC) and (ego_speed_mps > AEB_ACTIVATION_SPEED_THRESHOLD) and (tsc > MODE_COOLDOWN_SEC):
                self.mode, self.last_mode_change_time, self.safe_stop_locked = "AEB", sim_time, False

        elif self.mode == "FOLLOW":
            if target_ready and (ttc < self.AEB_ENTER_TTC) and (ego_speed_mps > AEB_ACTIVATION_SPEED_THRESHOLD) and (tsc > MODE_COOLDOWN_SEC):
                self.mode, self.last_mode_change_time, self.safe_stop_locked = "AEB", sim_time, False
            elif ((not target_ready) or (ttc >= self.FOLLOW_EXIT_TTC) or (distance == float('inf'))) and (tsc > MODE_COOLDOWN_SEC):
                self.mode, self.last_mode_change_time = "CRUISE", sim_time

        elif self.mode == "AEB":
            if (not self.safe_stop_locked) and (distance <= 6.0):
                self.safe_stop_locked = True
            if self.safe_stop_locked:
                throttle_des, brake_des = 0.0, 1.0
            else:
                throttle_des, brake_des = 0.0, min(1.0, max(0.0, distance / max(1e-3, 6.0)))
            if (sim_time - self.last_mode_change_time) >= AEB_MIN_HOLD_SEC and (ttc > self.AEB_EXIT_TTC):
                self.mode = "FOLLOW" if target_ready else "CRUISE"
                self.last_mode_change_time = sim_time
                self.safe_stop_locked = False

        # ---- CRUISE/FOLLOW control ----
        if self.mode != "AEB":
            lead_speed_for_acc = None
            if target_ready:
                lead_speed_for_acc = max(0.0, ego_speed_mps - rel_v)

            throttle_des, brake_des = _acc_longitudinal_control(
                ego_speed_mps, lead_speed_for_acc, distance, self.v_set_mps, self.time_headway_s
            )

            if self.mode == "FOLLOW":
                if rel_v > 0.0:
                    brake_des    = min(1.0, brake_des + APPROACH_REL_GAIN * rel_v)
                    throttle_des = max(0.0, throttle_des - APPROACH_REL_GAIN * rel_v)
                if np.isfinite(ttc) and ttc < TTC_SLOW_START:
                    bias = TTC_BIAS_GAIN * (TTC_SLOW_START - ttc)
                    throttle_des = max(0.0, throttle_des - bias)
                    brake_des    = min(1.0, brake_des + bias)

            if (not target_ready):
                speed_gap = max(0.0, self.v_set_mps - ego_speed_mps)
                boost = np.clip(0.20 + 0.10 * speed_gap, 0.20, 0.80)
                throttle_des = max(throttle_des, float(boost))
                brake_des = 0.0

        thr_cmd = _rate_limit(self.last_thr, throttle_des, THR_RATE_UP, THR_RATE_DOWN)
        brk_cmd = _rate_limit(self.last_brk, brake_des, BRK_RATE_UP, BRK_RATE_DOWN)
        self.last_thr, self.last_brk = float(thr_cmd), float(brk_cmd)

        return float(thr_cmd), float(brk_cmd), str(self.mode)
