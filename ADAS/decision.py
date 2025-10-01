#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
decision_lk_acc_kuksa.py — LK + ACC (Kuksa/Zenoh)  [2025-09-28 patched]
- 카메라(zenoh)로 차선 검출 → LK 스티어 계산 후 Kuksa에 publish
- Kuksa의 ACC 센서(거리/상대속도/ttc/타깃유무/리드추정속도)를 읽어 ACC 출력(thr/brk/mode) 계산 후 Kuksa에 publish
- publish는 dual-write(target + current)로 작성해 어떤 control도 값을 읽도록 보장
- CARLA 연결 시 ego 속도 사용:
  * lookahead_ratio( v_kmh ) 로 y_anchor 동적 조절
  * gains_for_speed( v_mps ) 로 조향 게인/클리핑 동적 조절
  (CARLA 미연결/실패 시 anchor_speed_kmh로 폴백)
- CARLA RADAR 부호 보정(+는 멀어짐) → rv_sign=-1.0 권장
- 접근일 때만 TTC 유한값, 아니면 ∞로 정리. 유효 타깃(eff_ht) 재판정.
- eff_ht=False(실질 타깃 없음)일 때는 cruise_thr_floor로 킥 없이도 출발
"""

import os, time, json, argparse, inspect
from typing import Optional
import numpy as np
import cv2
import zenoh
import math

# --- (옵션) CARLA ---
try:
    import carla
except Exception:
    carla = None

from types import SimpleNamespace
from kuksa_client.grpc import VSSClient, Datapoint
import common.acc_algo as ACCP  # <- 모듈 자체도 import

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# --- LK, ACC 알고리즘 ---
from common.LK_algo import (
    detect_lanes_and_center, fuse_lanes_with_memory,
    lane_mid_x, lookahead_ratio, gains_for_speed, speed_of
)
from common.acc_algo import ACCController, ACCInputs

SPEED_VSS_KEY = "Vehicle.Speed"

DEF_TOPIC_CAM = "carla/cam/front"
ACC_SENS_PATHS = [
    "Vehicle.ADAS.ACC.Distance",
    "Vehicle.ADAS.ACC.RelSpeed",
    "Vehicle.ADAS.ACC.TTC",
    "Vehicle.ADAS.ACC.HasTarget",
    "Vehicle.ADAS.ACC.LeadSpeedEst",
]
ACC_THR_PATH  = "Vehicle.ADAS.ACC.Ctrl.Throttle"
ACC_BRK_PATH  = "Vehicle.ADAS.ACC.Ctrl.Brake"
ACC_MODE_PATH = "Vehicle.ADAS.ACC.Ctrl.Mode"
LK_STEER_PATH = "Vehicle.ADAS.LK.Steering"
''' 이전상태 키값
ROAD_KEY = os.environ.get("ROAD_VSS_KEY", "Vehicle.Private.Road.State") #도로상태 일단 카메라만
ROAD_CONF_KEY  = os.environ.get("ROAD_CONF_KEY", "Vehicle.Private.Road.Confidence")
ROAD_TS_KEY    = os.environ.get("ROAD_TS_KEY", "Vehicle.Private.Road.Ts")
RS_POLL_SEC    = float(os.environ.get("ROAD_POLL_SEC", "0.3"))
ROAD_MIN_CONF  = float(os.environ.get("ROAD_MIN_CONF", "0.65"))
ROAD_STALE_SEC = float(os.environ.get("ROAD_STALE_SEC", "0.6"))
'''
ROAD_KEY       = os.environ.get("ROAD_VSS_KEY", "Vehicle.Private.Road.State")
ROAD_CONF_KEY  = os.environ.get("ROAD_CONF_KEY", "Vehicle.Private.Road.Confidence")
ROAD_TS_KEY    = os.environ.get("ROAD_TS_KEY", "Vehicle.Private.Road.Ts")
RS_POLL_SEC    = float(os.environ.get("ROAD_POLL_SEC", "0.3"))
ROAD_MIN_CONF  = float(os.environ.get("ROAD_MIN_CONF", "0.4"))
# '신선도'는 시뮬레이션 시계로 보정(아래 2) 참고)
SIM_STALE_SEC  = float(os.environ.get("SIM_STALE_SEC", "1.0"))

# ---------- 유틸: CARLA ego 찾기 ----------
def find_actor_by_role(world, role_name: str) -> Optional["carla.Actor"]:
    try:
        for actor in world.get_actors().filter("*vehicle*"):
            if actor.attributes.get("role_name") == role_name:
                return actor
    except Exception:
        pass
    return None

def _read_speed_from_kuksa(kc):
    try:
        m = kc.get_current_values([SPEED_VSS_KEY]) or {}
        v = m.get(SPEED_VSS_KEY)
        val = getattr(v, "value", v)
        return float(val or 0.0)
    except Exception:
        return 0.0


# ---------- Kuksa helpers ----------
def _unwrap(x):
    try: return getattr(x, "value")
    except Exception: pass
    if isinstance(x, dict) and "value" in x: return x["value"]
    if isinstance(x, (list, tuple)) and x:  return _unwrap(x[0])
    return x

def _as_bool(x):
    if isinstance(x, bool): return x
    try:
        if isinstance(x, (int,float)): return x != 0
        if isinstance(x, str): return x.strip().lower() in ("1","true","t","yes","y","on")
    except: pass
    return False

def read_acc_sensors(kc: VSSClient):
    try:
        res = kc.get_current_values(ACC_SENS_PATHS)
        if not isinstance(res, dict):
            res = {p: kc.get_current_value(p) for p in ACC_SENS_PATHS}
    except AttributeError:
        res = {p: kc.get_current_value(p) for p in ACC_SENS_PATHS}
    return {k: _unwrap(v) for k, v in res.items()}
    
def _get_vss_value(kc: VSSClient, key: str, prefer_target: bool = False):
    """VSSClient 버전 차이를 흡수하는 안전 단일키 getter."""
    # 1) 복수형 우선
    try:
        if prefer_target and hasattr(kc, "get_target_values"):
            m = kc.get_target_values([key]) or {}
            return _unwrap(m.get(key))
        if hasattr(kc, "get_current_values"):
            m = kc.get_current_values([key]) or {}
            return _unwrap(m.get(key))
    except Exception:
        pass
    # 2) 단수형이 있다면 폴백
    for name in ("get_target_value", "get_current_value", "get_value", "get"):
        if hasattr(kc, name):
            try:
                return _unwrap(getattr(kc, name)(key))
            except Exception:
                pass
    return None

# --- 스티어 듀얼-라이트 + 마지막 값 폴백 ---
_last_steer = 0.0
def publish_steer_dual(kuksa: VSSClient, st: Optional[float]):
    global _last_steer
    if st is None:
        st = _last_steer
    else:
        _last_steer = float(st)
    st = float(st)
    try: kuksa.set_target_values({LK_STEER_PATH: Datapoint(st)})
    except Exception: pass
    try: kuksa.set_current_values({LK_STEER_PATH: Datapoint(st)})
    except Exception: pass

def write_acc_dual(kc: VSSClient, thr: float, brk: float, mode: str):
    thr = float(max(0.0, min(1.0, thr)))
    brk = float(max(0.0, min(1.0, brk)))
    try:
        kc.set_target_values({
            ACC_THR_PATH:  Datapoint(thr),
            ACC_BRK_PATH:  Datapoint(brk),
            ACC_MODE_PATH: Datapoint(mode),
        })
    except Exception: pass
    try:
        kc.set_current_values({
            ACC_THR_PATH:  Datapoint(thr),
            ACC_BRK_PATH:  Datapoint(brk),
            ACC_MODE_PATH: Datapoint(mode),
        })
    except Exception: pass

# ---------- ACC signature adapters ----------
def _acc_build_inputs(v_set_mps, **kwargs):
    try:
        return ACCInputs(**kwargs, v_set=v_set_mps)
    except TypeError:
        try:
            obj = ACCInputs(**kwargs)
            for name in ("v_set","v_set_mps","target_speed","target_speed_mps","setpoint"):
                if hasattr(obj, name):
                    setattr(obj, name, v_set_mps)
                    break
            return obj
        except Exception:
            return SimpleNamespace(**kwargs, v_set=v_set_mps)

def _acc_step(ctrl, acc_in, dt, v_set_mps):
    try:
        sig = inspect.signature(ctrl.step)
        names = [p.name for p in sig.parameters.values() if p.name != "self"]
        if names == ["sim_time", "ego_speed_mps", "acc"]:
            return ctrl.step(time.perf_counter(), 0.0, acc_in)
        kw = {}
        for name in names:
            n = name.lower()
            if n in ("inputs","inp","acc","acc_in","meas","measurement","state"):
                kw[name] = acc_in
            elif n in ("dt","delta_t","delta","delta_time","step","ts","timestep","time_step","time_dt"):
                kw[name] = dt
            elif n in ("v_set","vset","target_speed","target_speed_mps","setpoint",
                       "v_ref","v_target","v_des","v_des_mps","set_speed","cruise","cruise_speed"):
                kw[name] = v_set_mps
            elif n in ("sim_time","time_s","t_now"): kw[name] = time.perf_counter()
            elif n in ("ego_speed","ego_speed_mps","v_ego"): kw[name] = 0.0
        if kw:
            return ctrl.step(**kw)
    except Exception:
        pass

    tries = (
        lambda: ctrl.step(time.perf_counter(), 0.0, acc_in),
        lambda: ctrl.step(acc_in, dt, v_set_mps),
        lambda: ctrl.step(acc_in, v_set_mps, dt),
        lambda: ctrl.step(v_set_mps, dt, acc_in),
        lambda: ctrl.step(dt, acc_in, v_set_mps),
        lambda: ctrl.step(dt, v_set_mps, acc_in),
        lambda: ctrl.step(acc_in),
    )
    for f in tries:
        try: return f()
        except Exception: pass
    return SimpleNamespace(throttle=0.0, brake=0.0, mode="ERR_SIG")

def _acc_unpack(out):
    if isinstance(out, (tuple, list)):
        if len(out) >= 3:
            thr = float(out[0]); brk = float(out[1]); mode = str(out[2])
            return thr, brk, mode
        return 0.0, 0.0, "UNKNOWN"
    if isinstance(out, dict):
        thr = float(out.get("throttle", out.get("acc_throttle", 0.0)))
        brk = float(out.get("brake",    out.get("acc_brake",    0.0)))
        mode = str(out.get("mode",      out.get("state", "UNKNOWN")))
        return thr, brk, mode
    thr = float(getattr(out, "throttle", getattr(out, "acc_throttle", 0.0)))
    brk = float(getattr(out, "brake",    getattr(out, "acc_brake",    0.0)))
    mode = str(getattr(out, "mode",      getattr(out, "state", "UNKNOWN")))
    return thr, brk, mode

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    # Zenoh
    ap.add_argument("--zenoh_endpoint", default="tcp/127.0.0.1:7447")
    ap.add_argument("--topic_cam", default=DEF_TOPIC_CAM)
    ap.add_argument("--fps", type=int, default=20)

    # Kuksa
    ap.add_argument("--kuksa_host", default="127.0.0.1")
    ap.add_argument("--kuksa_port", type=int, default=55555)

    # CARLA (옵션)
    ap.add_argument("--carla_host", default="127.0.0.1")
    ap.add_argument("--carla_port", type=int, default=2000)
    ap.add_argument("--carla_timeout", type=float, default=5.0)
    ap.add_argument("--carla_role", default="ego")
    ap.add_argument("--use_carla_speed", type=int, default=1)

    # LK
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--canny_low", type=int, default=60)
    ap.add_argument("--canny_high", type=int, default=180)
    ap.add_argument("--hough_thresh", type=int, default=40)
    ap.add_argument("--hough_min_len", type=int, default=20)
    ap.add_argument("--hough_max_gap", type=int, default=60)
    ap.add_argument("--lane_mem_ttl", type=int, default=90)
    ap.add_argument("--anchor_speed_kmh", type=float, default=50.0)

    # ACC 출발/게이팅
    ap.add_argument("--acc_target_kmh", type=float, default=80.0)
    ap.add_argument("--rv_sign", type=float, default=1.0, help="상대속도 부호 보정(CARLA RADAR: +는 멀어짐) → -1")
    ap.add_argument("--launch_ttc_s", type=float, default=10.0, help="TTC가 이 값보다 크면 타깃 무시(출발 허용)")
    ap.add_argument("--launch_gap_m", type=float, default=15.0, help="거리 d가 이 값보다 크면 타깃 무시")
    ap.add_argument("--cruise_thr_floor", type=float, default=1.0, help="타깃 무시 상태 최소 추진력")
    
    ap.add_argument("--rv_ema", type=float, default=0.40)
    ap.add_argument("--ttc_closing_floor", type=float, default=0.4)
    ap.add_argument("--follow_gate_gap_m", type=float, default=35.0)
    ap.add_argument("--follow_gate_ttc_s", type=float, default=18.0)
    ap.add_argument("--acc_follow_enter_ttc", type=float, default=10.0)
    ap.add_argument("--acc_follow_exit_ttc",  type=float, default=12.0)

    # Viz/Log
    ap.add_argument("--display", type=int, default=1)
    ap.add_argument("--print", type=int, default=1)
    args = ap.parse_args()
    dt = 1.0 / max(1, args.fps)

    # Kuksa
    kuksa = VSSClient(args.kuksa_host, args.kuksa_port)
    kuksa.connect()
    print(f"[decision_LK+ACC] Kuksa connected @ {args.kuksa_host}:{args.kuksa_port}")

    # CARLA (옵션 연결)
    ego = None
    world = None                      # >>> 루프 재탐색에서 쓸 world 보관
    _last_seek = 0.0              # >>> 마지막 재탐색 시각

    if carla and args.use_carla_speed:
        try:
            client = carla.Client(args.carla_host, int(args.carla_port))
            client.set_timeout(float(args.carla_timeout))
            world = client.get_world()                                   # >>>
            ego = find_actor_by_role(world, args.carla_role)

            s = world.get_settings()
            print(f"[WORLD] connected. sync={s.synchronous_mode}, fixed_dt={s.fixed_delta_seconds}, ego_found={bool(ego)}")
        except Exception as e:
            print(f"[WARN] CARLA connect failed: {e}. Fallback to static anchor_speed_kmh.")

    # Zenoh
    zcfg = zenoh.Config()
    try:
        zcfg.insert_json5("mode", '"client"')
        zcfg.insert_json5("connect/endpoints", f'["{args.zenoh_endpoint}"]')
    except AttributeError:
        zcfg.insert_json("mode", '"client"')
        zcfg.insert_json("connect/endpoints", f'["{args.zenoh_endpoint}"]')
    z = zenoh.open(zcfg)

    latest = {"bgr": None, "meta": {}, "ts": 0.0}
    def _on_cam(sample):
        try:
            raw = None
            if hasattr(sample, "payload"): raw = bytes(sample.payload)
            elif hasattr(sample, "value") and hasattr(sample.value, "payload"):
                raw = bytes(sample.value.payload)
            if raw is None: return

            meta = None
            att = getattr(sample, "attachment", None)
            if att:
                try: meta = json.loads(att.decode("utf-8"))
                except Exception: meta = None

            w = int((meta or {}).get("w", args.width))
            h = int((meta or {}).get("h", args.height))

            if len(raw) != w*h*4:
                if args.print:
                    print(f"[WARN] bad payload size: got={len(raw)} expected={w*h*4} (w={w},h={h})")
                return

            arr = np.frombuffer(raw, np.uint8).reshape((h, w, 4))
            latest["bgr"] = arr[:, :, :3].copy()
            latest["meta"] = meta or {"w": w, "h": h, "frame": -1}
            latest["ts"] = time.time()
        except Exception as e:
            print("[ERR] cam callback:", repr(e))
    sub_cam = z.declare_subscriber(args.topic_cam, _on_cam)

    if args.display:
        cv2.namedWindow("LK_decision", cv2.WINDOW_NORMAL)

    lane_mem = {"left": None, "right": None, "t_left": None, "t_right": None, "center_x": None}
    acc_ctrl = ACCController()
    # Kuksa에서 도로 상태 읽기(예: 퓨전 결과)
    #road = _get_vss_value(kuksa, "Vehicle.Private.StateFused.State") 퓨전기반 
    road = _get_vss_value(kuksa, "Vehicle.Private.Road.State") #카메라 기반
    acc_ctrl.set_road_state(road or "dry")
    road_raw = road  # 마지막으로 읽은 raw 를 저장해 로그에서 보여주기 위함
    next_rs_poll = time.perf_counter()

    try:
        ACCP.FOLLOW_ENTER_TTC = float(args.acc_follow_enter_ttc)
        ACCP.FOLLOW_EXIT_TTC  = float(args.acc_follow_exit_ttc)
    except Exception:
        pass
    v_set = float(args.acc_target_kmh) / 3.6

    print("[INFO] decision_LK+ACC running. Press q to quit.")
    try:
        next_t = time.perf_counter()
        while True:
            now = time.perf_counter()
            if now < next_t: time.sleep(next_t - now)
            next_t += dt

            if latest["bgr"] is None:
                if args.display: cv2.waitKey(1)
                continue
         # 위치: while True: 내부, ACC 계산 블록 들어가기 '바로 직전'이 가장 깔끔
            if now >= next_rs_poll:
                try:
                    raw  = _get_vss_value(kuksa, ROAD_KEY,      prefer_target=False)
                    conf = float(_get_vss_value(kuksa, ROAD_CONF_KEY, prefer_target=False) or 0.0)
                    ts   = float(_get_vss_value(kuksa, ROAD_TS_KEY,   prefer_target=False) or 0.0)

        # 시뮬레이션 시계 기준 신선도(아래 2) 설명)
                    sim_now = float((latest.get("meta") or {}).get("sim_ts") or 0.0)
                    fresh = True if sim_now <= 0.0 else ((sim_now - ts) <= SIM_STALE_SEC)
                    if raw and fresh and (conf >= ROAD_MIN_CONF):
                        road_raw = raw
                        acc_ctrl.set_road_state(road_raw)  # 히스테리시스/Δ 반영
                except Exception:
                    pass
                next_rs_poll = now + RS_POLL_SEC


            # ================= LK =================
            bgr = latest["bgr"]
            h, w = bgr.shape[:2]
            frame_id = int(latest["meta"].get("frame", 0))
            lanes = detect_lanes_and_center(
                bgr, roi_vertices=None,
                canny_low=args.canny_low, canny_high=args.canny_high,
                hough_thresh=args.hough_thresh,
                hough_min_len=args.hough_min_len,
                hough_max_gap=args.hough_max_gap,
            )
            used_left, used_right, _, _ = fuse_lanes_with_memory(
                lanes, frame_id, lane_mem, ttl_frames=args.lane_mem_ttl
            )

            # --- 동적 y_anchor / 게인 ---
            v_mps = 0.0
            if ego is not None and args.use_carla_speed:
                try:
                    v_mps = float(speed_of(ego))  # speed_of: m/s
                except Exception:
                    v_mps = 0.0
            v_kmh = v_mps * 3.6
            # lookahead_ratio는 "km/h" 기준으로 사용 (친구 코드의 TODO 반영)
            y_anchor = int(lookahead_ratio(v_kmh if v_kmh > 0 else args.anchor_speed_kmh) * h)

            x_cam_mid = w // 2
            x_lane_mid = lane_mem.get("center_x")
            if used_left and used_right:
                x_lane_mid, _, _ = lane_mid_x(used_left, used_right, y_anchor)
                lane_mem["center_x"] = x_lane_mid

            st = 0.0
            if x_lane_mid is not None:
                offset_px = (x_lane_mid - x_cam_mid)
                # gains_for_speed는 m/s 입력
                kp, st_clip = gains_for_speed(v_mps)
                st = float(max(-st_clip, min(st_clip, kp * offset_px)))
            publish_steer_dual(kuksa, st)

            # ================= ACC =================
            sens = read_acc_sensors(kuksa)
            d_raw  = sens.get("Vehicle.ADAS.ACC.Distance", None)
            rv_raw = sens.get("Vehicle.ADAS.ACC.RelSpeed", 0.0)
            ht_raw = _as_bool(sens.get("Vehicle.ADAS.ACC.HasTarget", 0))
            vL     = sens.get("Vehicle.ADAS.ACC.LeadSpeedEst", None)
            ttc_env = sens.get("Vehicle.ADAS.ACC.TTC", None)

            # 상대속도 부호 보정(+면 접근)
            rv_app = float(args.rv_sign) * float(rv_raw or 0.0)

            # TTC: env가 유효값(9000 미만) 주면 우선 사용, 아니면 rv로 재계산
            ttc_from_env = None
            if ttc_env is not None:
                tval = float(ttc_env)
                if tval < 60.0:       # 너무 큰 TTC는 무시
                    ttc_from_env = tval

            # rv EMA
            if not hasattr(main, "_rv_f"):
                main._rv_f = rv_app
            rv_alpha = 0.45
            main._rv_f = rv_alpha * rv_app + (1.0 - rv_alpha) * main._rv_f

            # d_val = float(d_raw) if d_raw is not None else float("inf")
            d_val = float(d_raw) if (d_raw is not None and float(d_raw) < 9000.0) else float("inf")
            closing = max(0.0, main._rv_f)
            denom = max(closing, max(0.0, float(args.ttc_closing_floor)))
            ttc_sane = (d_val / denom) if np.isfinite(d_val) else float("inf")
            ttc_sane = (ttc_from_env if (ttc_from_env is not None) else ttc_sane)
            ttc_sane = 9999.9 if not np.isfinite(ttc_sane) else min(ttc_sane, 99.9)
            # 유효 타깃: ht_raw OR (거리/ttc 게이트)
            DIST_STRICT = 35.0
            FOLLOW_GAP_GATE = float(args.follow_gate_gap_m)
            FOLLOW_TTC_GATE = getattr(acc_ctrl, "FOLLOW_ENTER_TTC", float(args.follow_gate_ttc_s))
            # --- 신선도 게이팅: ACC.Ts가 있으면 Zenoh sim_ts 대비 STALE이면 타깃 무효 ---
            acc_ts = _get_vss_value(kuksa, "Vehicle.ADAS.ACC.Ts", prefer_target=False)
            sim_now = float((latest.get("meta") or {}).get("sim_ts") or 0.0)
            fresh_ok = True if (not sim_now or not acc_ts) else ((sim_now - float(acc_ts)) <= 1.0)  # 1s 내면 신선

            eff_ht = fresh_ok and (
                (np.isfinite(d_val) and d_val < DIST_STRICT) or
                bool(ht_raw) or
                (np.isfinite(d_val) and (d_val < FOLLOW_GAP_GATE)) or
                (np.isfinite(ttc_sane) and (ttc_sane < FOLLOW_TTC_GATE))
            )
            acc_in = _acc_build_inputs(
                v_set_mps=v_set,
                distance=(d_val if eff_ht else None),
                rel_speed=rv_app,
                ttc=(ttc_sane if eff_ht else float("inf")),
                has_target=eff_ht,
                lead_speed_est=(vL if eff_ht else None),
            )
            acc_out = acc_ctrl.step(time.perf_counter(), v_mps, acc_in)
            acc_thr, acc_brk, acc_mode = _acc_unpack(acc_out)

            # eff_ht=False → 킥 없이도 천천히 출발
            if not eff_ht:
                if acc_thr < args.cruise_thr_floor:
                    acc_thr = args.cruise_thr_floor
                if acc_mode in ("UNKNOWN", "UNK", "IDLE", "ERR_SIG"):
                    acc_mode = "CRUISE"

            write_acc_dual(kuksa, acc_thr, acc_brk, acc_mode)

            if args.print:
                ttc_print = f"{ttc_sane:.1f}" if np.isfinite(ttc_sane) else "inf"
                print(
        f"[ACC] d={None if d_raw is None else f'{d_val:.2f}'} "
        f"rv_app={rv_app:+.3f} "
        f"ttc={ttc_print} (gate_follow={getattr(acc_ctrl,'FOLLOW_ENTER_TTC',float(args.follow_gate_ttc_s)):.1f}, "
        f"aeb_enter={getattr(acc_ctrl,'AEB_ENTER_TTC',0.0):.1f}) "
        f"ht={eff_ht} v_set={v_set*3.6:.0f} v={v_kmh:4.1f}km/h -> "
        f"thr={acc_thr:.2f} brk={acc_brk:.2f} mode={acc_mode} "
        f"road={acc_ctrl.road_state}({road_raw})"
    )

            # ================= Viz =================
            if args.display:
                vis = bgr.copy()
                rv = lanes.get("roi_vertices")
                if rv is not None:
                    overlay = vis.copy()
                    cv2.fillPoly(overlay, rv, (0, 0, 255))
                    vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0.0)
                for x1, y1, x2, y2 in lanes.get("line_segs", []):
                    cv2.line(vis, (x1, y1), (x2, y2), (128, 128, 128), 1, cv2.LINE_AA)
                if used_left:
                    cv2.line(vis, (used_left[0], used_left[1]), (used_left[2], used_left[3]), (0, 0, 255), 5, cv2.LINE_AA)
                if used_right:
                    cv2.line(vis, (used_right[0], used_right[1]), (used_right[2], used_right[3]), (0, 0, 255), 5, cv2.LINE_AA)
                x_cam_mid = w // 2
                cv2.line(vis, (x_cam_mid, 0), (x_cam_mid, h - 1), (0, 255, 255), 1, cv2.LINE_AA)
                if lane_mem.get("center_x") is not None:
                    cv2.line(vis, (lane_mem["center_x"], 0), (lane_mem["center_x"], h - 1), (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(vis, f"v={v_kmh:5.1f} km/h  str={st:+.2f}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
                mode_raw = (acc_mode or "UNKNOWN").upper()
                road_disp = (acc_ctrl.road_state or str(road_raw) or "unknown").upper()
                if "AEB" in mode_raw:
                    mode_disp, col = "AEB", (0, 0, 255)            # 빨강
                elif "FOLLOW" in mode_raw:
                    mode_disp, col = "FOLLOW", (0, 165, 255)       # 주황
                elif "CRUISE" in mode_raw:
                    mode_disp, col = "CRUISE", (0, 255, 0)         # 초록
                else:
                    mode_disp, col = mode_raw, (200, 200, 200)     # 회색(기타)

# TTC 표기 (무한대/클램프 처리와 동일 규칙)
                ttc_txt = "∞" if (not np.isfinite(ttc_sane) or ttc_sane >= 99.9) else f"{ttc_sane:.1f}s"
                d_txt = "-" if not np.isfinite(d_val) else f"{d_val:.1f}m"

# 배지 그리기 (오른쪽 상단)
                pad = 25
                badge_w, badge_h = 210, 75
                x0, y0 = w - badge_w - pad-410, pad
                x1, y1 = w - pad, pad + badge_h


# 텍스트
                cv2.putText(
                    vis, f"{mode_disp} | TTC {ttc_txt} | d {d_txt}",
                    (x0 + 12, y0 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2, cv2.LINE_AA
                )

# 2행: ROAD 상태
#  - 색상 커스터마이즈: DRY/초록, WET/파랑, SNOW/하늘색, ICE/빨강 등 원하면 매핑 추가 가능
                cv2.putText(
                    vis, f"ROAD {road_disp}",
                    (x0 + 12, y0 + 30 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA
                )
                cv2.imshow("LK_decision", vis) 
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
                    

    finally:
        try: kuksa.close()
        except Exception: pass
        try: sub_cam.undeclare()
        except Exception as e: print("[WARN] sub_cam.undeclare:", e)
        try: z.close()
        except Exception as e: print("[WARN] zenoh.close:", e)
        if args.display:
            try: cv2.destroyAllWindows()
            except Exception as e: print("[WARN] destroyAllWindows:", e)
        print("[INFO] decision.py stopped.")

if __name__ == "__main__":
    main()
