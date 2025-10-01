#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
control_acc_kuksa.py — Passive controller (ACC + LK via Kuksa)
- Ego: Autopilot OFF, read Kuksa commands (ACC/LK) and applies them to CARLA
- Lead: TrafficManager Autopilot ON (optional; absolute speed can be specified)
- Kuksa reads prioritize target values; fall back to current values if unavailable
- Steering pipeline: deadband → EMA → I-term (anti-windup, center damping) → rate-limit → apply
- After each tick, optional extra sleep (extra_sleep_ratio) to stabilize rendering/behavior
"""

import time, math, argparse, traceback
from typing import Any, Optional
from queue import Queue
import carla

# ---- Kuksa ----
DatapointType = None
try:
    from kuksa_client.grpc import VSSClient as DataBrokerClient
    try:
        from kuksa_client.grpc import Datapoint as _DP
        DatapointType = _DP
    except Exception:
        DatapointType = None
except Exception:
    DataBrokerClient = None
    DatapointType = None

def clamp(x, lo, hi): return lo if x < lo else hi if x > hi else x

def to_float_safe(x, default=0.0):
    try: return float(x)
    except Exception:
        if DatapointType is not None and isinstance(x, DatapointType):
            try: return float(getattr(x, "value", default))
            except Exception: return default
        if isinstance(x, dict) and "value" in x:
            try: return float(x["value"])
            except Exception: return default
        if isinstance(x, (list, tuple)) and x:
            return to_float_safe(x[0], default)
        return default

def unwrap(x):
    if DatapointType is not None and isinstance(x, DatapointType):
        return getattr(x, "value", None)
    if isinstance(x, dict) and "value" in x:
        return x["value"]
    if isinstance(x, (list, tuple)) and x:
        return unwrap(x[0])
    return x

def speed_kmh(actor):
    v = actor.get_velocity()
    return 3.6 * math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)

def find_actor_by_role(world, role, wait_sec=15.0):
    deadline = time.time() + wait_sec
    while time.time() < deadline:
        for v in world.get_actors().filter("*vehicle*"):
            if v.attributes.get("role_name") == role:
                return v
        time.sleep(0.05)
    return None
def stabilize_vehicle_physics(veh):
    try:
        pc = veh.get_physics_control()
        pc.use_sweep_wheel_collision = True
        pc.substepping = True
        pc.max_substep_delta_time = 0.01
        pc.max_substeps = 10
        try:
            com = pc.center_of_mass
            com.z -= 0.05
            pc.center_of_mass = com
        except Exception:
            pass
        wheels = pc.wheels
        for w in wheels:
            w.tire_friction = 2.0
            try:
                w.suspension_stiffness = getattr(w, "suspension_stiffness", 20000.0) * 0.8
                w.damping_compression  = max(2000.0, getattr(w, "damping_compression", 1500.0) * 1.4)
                w.damping_relaxation   = max(2000.0, getattr(w, "damping_relaxation", 1800.0) * 1.3)
            except Exception:
                pass
            w.damping_rate = max(0.25, getattr(w, "damping_rate", 0.15) * 1.5)
        pc.wheels = wheels
        veh.apply_physics_control(pc)
        print("[PHYS] ego stabilized")
    except Exception as e:
        print("[PHYS] skip:", e)

class KUKSA:
    def __init__(self, host="127.0.0.1", port=55555):
        self.cli = None
        if DataBrokerClient is None:
            print("[KUKSA] sdk not available → zeros"); return
        try:
            try: self.cli = DataBrokerClient(host, port)
            except Exception: self.cli = DataBrokerClient(f"{host}:{port}")
            if hasattr(self.cli, "connect"): self.cli.connect()
            print(f"[KUKSA] connected @ {host}:{port}")
        except Exception as e:
            print("[KUKSA] connect failed:", e); self.cli = None

    def _get_map(self, getter_name, keys):
        if self.cli is None: return {}
        try: getter = getattr(self.cli, getter_name)
        except Exception: return {}
        try:
            res = getter(keys)
            if isinstance(res, dict): return res
            return {k: getattr(self.cli, getter_name[:-1], lambda kk: None)(k) for k in keys}
        except Exception:
            return {}

    def read_value(self, key, prefer_target=True):
        if self.cli is None: return None
        if prefer_target and hasattr(self.cli, "get_target_values"):
            m = self._get_map("get_target_values", [key])
            if key in m: return unwrap(m[key])
        if hasattr(self.cli, "get_current_values"):
            m = self._get_map("get_current_values", [key])
            if key in m: return unwrap(m[key])
        for name in ("get_target_value","get_current_value","get_value","get","read"):
            if hasattr(self.cli, name):
                try: return unwrap(getattr(self.cli, name)(key))
                except Exception: pass
        return None

def main():
    ap = argparse.ArgumentParser("control (ACC+LK via Kuksa)")
    # CARLA
    ap.add_argument("--carla_host", default="127.0.0.1")
    ap.add_argument("--carla_port", type=int, default=2000)
    ap.add_argument("--tm_port", type=int, default=8000)
    ap.add_argument("--ego_role", default="ego")
    ap.add_argument("--lead_role", default="lead")
    ap.add_argument("--lead_speed_kmh", type=float, default=40.0, help="TM lead absolute speed (km/h)")

    # Kuksa
    ap.add_argument("--kuksa_host", default="127.0.0.1")
    ap.add_argument("--kuksa_port", type=int, default=55555)

    # VSS keys
    ap.add_argument("--thr_key",  default="Vehicle.ADAS.ACC.Ctrl.Throttle")
    ap.add_argument("--brk_key",  default="Vehicle.ADAS.ACC.Ctrl.Brake")
    ap.add_argument("--mode_key", default="Vehicle.ADAS.ACC.Ctrl.Mode")
    ap.add_argument("--steer_acc_key", default="Vehicle.ADAS.ACC.Ctrl.Steer")
    ap.add_argument("--steer_lk_key",  default="Vehicle.ADAS.LK.Steering")
    ap.add_argument("--prefer_acc_steer", type=int, default=1, help="If 1, prefer ACC.Ctrl.Steer; otherwise use LK.Steering")

    # Filters (throttle/brake)
    ap.add_argument("--steer_deadband", type=float, default=0.01)
    ap.add_argument("--steer_alpha", type=float, default=0.00, help="Steering EMA coefficient (0 disables)")
    ap.add_argument("--steer_rate", type=float, default=2.0, help="Steering rate limit [norm/s]")
    ap.add_argument("--thr_alpha", type=float, default=0.0)
    ap.add_argument("--brk_alpha", type=float, default=0.0)

    # Steering I-term
    ap.add_argument("--ki_steer", type=float, default=0.02, help="Steering I gain (small)")
    ap.add_argument("--i_clip", type=float, default=0.20, help="I항 한계(anti-windup)")
    ap.add_argument("--i_decay_tau", type=float, default=2.0, help="중심 근처에서 I 감쇠 시간상수[s]")

    # pacing / logging
    ap.add_argument("--log_every", type=float, default=0.5)
    ap.add_argument("--extra_sleep_ratio", type=float, default=0.12,
                    help="틱 잔여시간에 비례한 추가 sleep 비율(0~0.3 권장)")

    args = ap.parse_args()

    client = carla.Client(args.carla_host, args.carla_port)
    client.set_timeout(5.0)
    world = client.get_world()
    settings = world.get_settings()
    print(f"[WORLD] connected. sync={settings.synchronous_mode}, fixed_dt={settings.fixed_delta_seconds}")

    ego = find_actor_by_role(world, args.ego_role)
    if not ego: raise RuntimeError(f"Ego(role='{args.ego_role}') not found")
    stabilize_vehicle_physics(ego)
    lead = find_actor_by_role(world, args.lead_role)

    # Ego manual
    try: ego.set_autopilot(False)
    except Exception: pass

    # Lead via TM (옵션)
    '''
    if lead is not None:
        try:
            tm = client.get_trafficmanager(args.tm_port)
            try: tm.set_synchronous_mode(True)
            except Exception: pass
            lead.set_autopilot(True, args.tm_port)
            try: tm.auto_lane_change(lead, False)
            except Exception: pass
            try: tm.set_desired_speed(lead, float(args.lead_speed_kmh))  # km/h 기준
            except Exception: pass
            print("[INFO] Lead autopilot ON via TM")
        except Exception as e:
            print("[TM] lead setup failed:", e)
'''
    kuksa = KUKSA(args.kuksa_host, args.kuksa_port)

    tick_q: Queue = Queue(maxsize=3)
    world.on_tick(tick_q.put)

    # filters state
    steer_ema = None if args.steer_alpha <= 0.0 else 0.0
    thr_ema   = None if args.thr_alpha   <= 0.0 else 0.0
    brk_ema   = None if args.brk_alpha   <= 0.0 else 0.0
    steer_prev = 0.0
    i_state = 0.0  # 조향 I항 상태
    last_log = time.time()

    print(f"[RUN] Control via Kuksa | steer={'ACC.Ctrl.Steer' if args.prefer_acc_steer else 'LK.Steering'}→LK fallback")
    try:
        while True:
            snap = tick_q.get()
            dt = snap.delta_seconds if snap and snap.delta_seconds else (settings.fixed_delta_seconds or 0.05)
            t0 = time.time()

            # --- read ACC commands ---
            thr_raw = to_float_safe(kuksa.read_value(args.thr_key,  prefer_target=True), 0.0)
            brk_raw = to_float_safe(kuksa.read_value(args.brk_key,  prefer_target=True), 0.0)
            _mode   = kuksa.read_value(args.mode_key,  prefer_target=True)

            # --- steer selection (ACC first, then LK) ---
            steer_raw = None
            if args.prefer_acc_steer:
                v = kuksa.read_value(args.steer_acc_key, prefer_target=True)
                if v is not None: steer_raw = to_float_safe(v, 0.0)
            if steer_raw is None:
                v = kuksa.read_value(args.steer_lk_key, prefer_target=True)
                steer_raw = to_float_safe(v, 0.0)

            # --- throttle/brake filters ---
            thr = clamp(thr_raw, 0.0, 1.0)
            brk = clamp(brk_raw, 0.0, 1.0)
            if thr_ema is not None:
                thr_ema = args.thr_alpha * thr + (1.0 - args.thr_alpha) * thr_ema
                thr = clamp(thr_ema, 0.0, 1.0)
            if brk_ema is not None:
                brk_ema = args.brk_alpha * brk + (1.0 - args.brk_alpha) * brk_ema
                brk = clamp(brk_ema, 0.0, 1.0)

            # --- steering pipeline ---
            # 1) clamp + deadband
            steer = clamp(steer_raw, -1.0, 1.0)
            if abs(steer) < args.steer_deadband:
                steer = 0.0

            # 2) EMA (frame-based)
            if steer_ema is not None:
                steer_ema = args.steer_alpha * steer + (1.0 - args.steer_alpha) * steer_ema
                steer_p = clamp(steer_ema, -1.0, 1.0)
            else:
                steer_p = steer

            # 3) I-term with decay near center
            # 작은 명령일수록 빠르게 감쇠(자연 복원), 큰 편차는 유지
            if args.i_decay_tau > 1e-6:
                decay = math.exp(-abs(steer_p) / max(args.i_decay_tau, 1e-6))
            else:
                decay = 1.0  # no decay
            i_state = i_state * decay + args.ki_steer * steer_p * dt
            i_state = clamp(i_state, -args.i_clip, args.i_clip)

            steer_cmd = steer_p + i_state

            # 4) rate limit (apply on final command)
            if args.steer_rate > 0.0 and dt and dt > 0.0:
                max_step = args.steer_rate * dt
                steer_cmd = clamp(steer_cmd, steer_prev - max_step, steer_prev + max_step)
            steer_cmd = clamp(steer_cmd, -1.0, 1.0)
            steer_prev = steer_cmd

            # --- apply control (exactly once per tick) ---
            ego.apply_control(carla.VehicleControl(
                throttle=float(thr),
                brake=float(brk),
                steer=float(steer_cmd),
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False
            ))

            # --- logging (wall clock interval) ---
            now = time.time()
            if now - last_log >= max(0.1, args.log_every):
                v = speed_kmh(ego)
                print(f"[CTL] v={v:5.1f} km/h | thr={thr:4.2f} brk={brk:4.2f} steer_p={steer_p:+.3f} I={i_state:+.3f} cmd={steer_cmd:+.3f}"
                      + (f" | mode={_mode}" if _mode is not None else ""))
                last_log = now

            # --- pacing: sleep remainder + margin ---
            proc_dt = time.time() - t0
            target_dt = float(dt if dt and dt > 0 else (settings.fixed_delta_seconds or 0.05))
            extra = args.extra_sleep_ratio * target_dt
            to_sleep = (target_dt - proc_dt) + extra
            if to_sleep > 0:
                time.sleep(to_sleep)

    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C")
    except Exception:
        traceback.print_exc()
    finally:
        try: ego.set_autopilot(False)
        except Exception: pass
        print("[CLEAN] control.py passive exit")

if __name__ == "__main__":
    main()
