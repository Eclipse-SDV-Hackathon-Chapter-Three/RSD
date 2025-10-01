#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Env_ADAS.py — CARLA 센서 → Kuksa(VSS) 직접 write + 카메라 Zenoh publish

- Front 카메라: Zenoh로 BGRA 버퍼 전송(디스플레이/디버깅용)
- 레이더: ACC 5개 센서값 (Distance, RelSpeed, TTC, HasTarget, LeadSpeedEst)을 Kuksa에 set_current_values
- Lead 차량은 Traffic Manager로 속도 고정 주행(옵션)
- 에고 속도도 VSS("Vehicle.Speed", m/s)로 같이 퍼블리시해서 decision에서 참조 가능

주의:
- CARLA RADAR의 velocity는 +가 멀어짐, -가 접근 → 여기서는 '접근=+'가 되도록 v_sign=-1.0 적용
  (decision은 --rv_sign 1.0로 맞추면 일관됨)
"""

import os, time, json, math, argparse, signal, sys
import numpy as np
import cv2
import carla
import zenoh
from kuksa_client.grpc import VSSClient, Datapoint

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# ---------------- Zenoh ----------------
cfg = zenoh.Config()
try:
    cfg.insert_json5('mode', '"client"')
    cfg.insert_json5('connect/endpoints', '["tcp/127.0.0.1:7447"]')
except AttributeError:
    cfg.insert_json('mode', '"client"')
    cfg.insert_json('connect/endpoints', '["tcp/127.0.0.1:7447"]')

sess = zenoh.open(cfg)
pub  = sess.declare_publisher('carla/cam/front')

# ------------- 기본 파라미터/유틸 -------------
IMG_W        = int(os.environ.get("IMG_W", "640"))
IMG_H        = int(os.environ.get("IMG_H", "480"))
SENSOR_TICK  = float(os.environ.get("SENSOR_TICK", "0.05"))   # 20Hz
STATUS_EVERY = float(os.environ.get("STATUS_EVERY", "5.0"))

# RADAR 처리 파라미터
RADAR_V_SIGN = -1.0   # CARLA vel(+ 멀어짐)을 '접근=+'로 바꾸려면 -1.0
AZI_MAX_DEG  = 6.0   # 수평 각도 게이팅(완화)
ALT_MIN_DEG  = -0.5   # 수직 각도 게이팅(완화)
ALT_MAX_DEG  =  2.5
EPS_APPROACH = 0.3    # m/s, TTC 유한 판단 임계
EMA_ALPHA    = 0.25   # 1차 필터 강도(0~1)
writer = None
def cleanup_and_exit(signum, frame):
    try:
        if 'writer' in globals() and writer is not None:
            writer.release()
    except Exception:
        pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup_and_exit)
def clamp(x, lo, hi): return max(lo, min(hi, x))

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--kuksa_port', type=int, default=55555)
    ap.add_argument('--fps',  type=int, default=40)
    ap.add_argument('--spawn_idx', type=int, default=328)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--fov', type=float, default=90.0)
    ap.add_argument('--display', type=int, default=0) #1이면 화면 나옴 
    ap.add_argument('--record', type=str, default='')
    ap.add_argument('--record_mode', choices=['raw','vis','both'], default='vis')
    ap.add_argument('--tm_port', type=int, default=8000)
    ap.add_argument('--lead_speed_kmh', type=float, default=20.0)
    ap.add_argument('--radar_log', type=int, default=1)
    ap.add_argument('--cam_log', type=int, default=0)
    args = ap.parse_args()

    # --- CARLA 연결/세팅 ---
    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)
    world = client.get_world()

    dt = 1.0 / max(1, args.fps)
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = dt
    settings.substepping = True
    settings.max_substep_delta_time = 0.005   # 5 ms
    settings.max_substeps = 10                # 최대 200 Hz 내부 물리
    world.apply_settings(settings)
    print(f"[WORLD] synchronous_mode=True, delta_seconds={dt:.3f}")

    # --- Kuksa 연결 ---
    kuksa = VSSClient(args.host, args.kuksa_port)
    kuksa.connect()
    print(f"[KUKSA] Connected to {args.host}:{args.kuksa_port}")

    # --- 차량 스폰 ---
    bp = world.get_blueprint_library()

    # ego
    ego_bp = (bp.filter('vehicle.*model3*') or bp.filter('vehicle.*'))[0]
    ego_bp.set_attribute('role_name', 'ego')
    sps = world.get_map().get_spawn_points()
    tf  = sps[min(max(0, args.spawn_idx), len(sps)-1)]
    ego = world.try_spawn_actor(ego_bp, tf)
    if ego is None:
        raise RuntimeError("Failed to spawn Ego. Try another spawn_idx or free the spawn point.")

    # lead
    lead_bp = (bp.filter('vehicle.audi.tt') or bp.filter('vehicle.*'))[0]
    lead_bp.set_attribute('role_name', 'lead')
    ego_wp  = world.get_map().get_waypoint(tf.location)
    lead_wp = ego_wp.next(30.0)[0]
    lead_tf = lead_wp.transform
    lead_tf.location.z = tf.location.z
    lead = world.try_spawn_actor(lead_bp, lead_tf)
    if lead is None:
        raise RuntimeError("Failed to spawn Lead. Try another spawn_idx or free the spawn point.")

    # --- Traffic Manager로 lead 주행 ---
    '''
    try:
        tm = client.get_trafficmanager(args.tm_port)
        try: tm.set_synchronous_mode(True)
        except Exception: pass

        lead.set_autopilot(True, args.tm_port)
        try: tm.auto_lane_change(lead, False)  # 차선 변경 금지
        except Exception: pass
        try: tm.set_desired_speed(lead, float(args.lead_speed_kmh))  # km/h
        except Exception: pass

        print(f"[TM] Lead ON @ {args.tm_port}, v={args.lead_speed_kmh:.1f} km/h")
    except Exception as e:
        print("[WARN] TM setup failed:", e)
        '''

    # --- 센서 부착 ---
    # Front camera
    cam_bp = bp.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(args.width))
    cam_bp.set_attribute('image_size_y', str(args.height))
    cam_bp.set_attribute('fov', str(args.fov))
    cam_bp.set_attribute('sensor_tick', str(dt))
    cam_tf = carla.Transform(carla.Location(x=1.2, z=1.4))
    cam    = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)

    # Radar
    radar_bp = bp.find('sensor.other.radar')
    radar_bp.set_attribute('range', '120')
    radar_bp.set_attribute('horizontal_fov', '20')
    radar_bp.set_attribute('vertical_fov', '10')
    radar_bp.set_attribute('sensor_tick', str(dt))
    radar_tf = carla.Transform(carla.Location(x=2.8, z=1.0))
    radar    = world.spawn_actor(radar_bp, radar_tf, attach_to=ego)

    # (옵션) chase cam
    chase = None
    latest_chase = {'bgr': None}
    try:
        chase_bp = bp.find('sensor.camera.rgb')
        chase_bp.set_attribute('image_size_x', str(args.width))
        chase_bp.set_attribute('image_size_y', str(args.height))
        chase_bp.set_attribute('fov', '70')
        chase_bp.set_attribute('sensor_tick', str(dt))
        chase_tf = carla.Transform(carla.Location(x=-6.0, z=3.0), carla.Rotation(pitch=-12.0))
        chase = world.spawn_actor(chase_bp, chase_tf, attach_to=ego)
        def _on_chase(img: carla.Image):
            arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
            latest_chase['bgr'] = arr[:, :, :3].copy()
        chase.listen(_on_chase)
    except Exception as e:
        print(f"[WARN] Failed to attach chase camera: {e}")

    # --- 카메라 → Zenoh ---
    latest_front = {'bgr': None}
    def on_cam(img: carla.Image):
        buf = memoryview(img.raw_data)
        att = json.dumps({
            "w": img.width, "h": img.height, "c": 4, "format": "bgra8",
            "stride": img.width * 4, "frame": int(img.frame),
            "sim_ts": float(img.timestamp), "pub_ts": time.time()
        }).encode("utf-8")
        pub.put(bytes(buf), attachment=att)
        # 로컬 디스플레이용 복사
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
        latest_front['bgr'] = arr[:, :, :3].copy()
        if args.cam_log:
            print("[CAM] image sending...")

    cam.listen(on_cam)

    # --- 레이더 → Kuksa(VSS) ---
    # EMA 상태
    state = {"d": None, "vr": None, "vlead": None, "ttc": None}
    def ema(name, value):
        prev = state[name]
        if value is None:
            return prev  # 이전값 유지(초기엔 None일 수 있음)
        state[name] = value if prev is None else (1.0 - EMA_ALPHA) * prev + EMA_ALPHA * value
        return state[name]

    # 타깃 메모리(잠깐 끊겨도 유지)
    TARGET_MEM_SEC = 0.4
    last = {"ts": 0.0, "d": 9999.9, "rel": 0.0, "ttc": 9999.9, "has": False, "vlead": 0.0}

    def on_radar(meas: carla.RadarMeasurement):
        try:
            # 1) 각도/거리 게이팅
            cand = []
            for d in meas:
                az, alt = math.degrees(d.azimuth), math.degrees(d.altitude)
                if abs(az) <= AZI_MAX_DEG and (ALT_MIN_DEG <= alt <= ALT_MAX_DEG) and d.depth >= 1.0:
                    cand.append(d)

            # 2) 타깃 없음 처리(메모리 유지)
            if not cand:
                now = time.time()
                if (now - last["ts"] <= TARGET_MEM_SEC) and last["has"]:
                    updates = {
                        "Vehicle.ADAS.ACC.Distance":     Datapoint(float(last["d"])),
                        "Vehicle.ADAS.ACC.RelSpeed":     Datapoint(float(last["rel"])),
                        "Vehicle.ADAS.ACC.TTC":          Datapoint(float(last["ttc"])),
                        "Vehicle.ADAS.ACC.HasTarget":    Datapoint(True),
                        "Vehicle.ADAS.ACC.LeadSpeedEst": Datapoint(float(last["vlead"])),
                    }
                    kuksa.set_current_values(updates)
                else:
                    updates = {
                        "Vehicle.ADAS.ACC.Distance":     Datapoint(9999.9),
                        "Vehicle.ADAS.ACC.RelSpeed":     Datapoint(0.0),
                        "Vehicle.ADAS.ACC.TTC":          Datapoint(9999.9),
                        "Vehicle.ADAS.ACC.HasTarget":    Datapoint(False),
                        "Vehicle.ADAS.ACC.LeadSpeedEst": Datapoint(0.0),
                    }
                    kuksa.set_current_values(updates)
                return

            # 3) 가까운 것 3개 가중 평균(1/depth)
            cand.sort(key=lambda x: x.depth)
            picks = cand[:min(3, len(cand))]
            ws    = [1.0 / max(1e-3, p.depth) for p in picks]
            wsum  = sum(ws) or 1.0

            dist = sum(p.depth * w for p, w in zip(picks, ws)) / wsum
            rel  = sum((RADAR_V_SIGN * p.velocity) * w for p, w in zip(picks, ws)) / wsum  # 접근=+
            # 평균 방위각(라디안) — 이후 yaw 보정과 동적 게이트에 사용
            az_avg_rad = sum((p.azimuth) * w for p, w in zip(picks, ws)) / wsum
            az_avg_deg = az_avg_rad * 180.0 / math.pi
 

            # 4) ego 속도
            v = ego.get_velocity()
            ego_speed = math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)

            # 5) TTC (접근일 때만 유한)
         # 4.5) ego yaw-rate 보정: 회전으로 생기는 가짜 방사속도 제거
            omega_z = float(ego.get_angular_velocity().z)  # rad/s (CARLA 기본 단위)
            # 회전 유발 방사속도 ≈ ω * d * sin(azimuth)
            rel_rot  = omega_z * dist * math.sin(az_avg_rad)
            rel_corr = rel - rel_rot

            # 5) TTC (접근일 때만 유한) — 보정된 rel 사용
            EPS_APPROACH = 0.12  # 곡선에서 과탐지 줄이려 0.12~0.2 권장
            if rel_corr > EPS_APPROACH:
                ttc = max(0.1, dist / rel_corr)
            else:
                ttc = float('inf')

            # 6) 유효 타깃 판정
             # 5.5) 커브일 때 azimuth 게이트 동적 축소(예: |ω|≥0.3 rad/s → ±6°)
            base_gate_deg = 10.0
            min_gate_deg  = 6.0
            gate_deg = base_gate_deg if abs(omega_z) < 0.3 else min_gate_deg

            # 6) 유효 타깃 판정(강화)
            HAS_TARGET_DIST_GATE = 60.0  # m
            has_target = (
                math.isfinite(ttc)
                and (rel_corr > EPS_APPROACH)
                and (dist < HAS_TARGET_DIST_GATE)
                and (abs(az_avg_deg) <= gate_deg)
            )

            # 7) 정규화 + EMA
            dist = clamp(dist, 0.0, 500.0)
            rel  = clamp(rel_corr,  -100.0, 100.0)
            lead_est = clamp(ego_speed - rel, 0.0, 100.0)
            ttc_out  = 9999.9 if not math.isfinite(ttc) else clamp(ttc, 0.0, 1e4)
            # None을 넣지 말고, 이전값이 있으면 이전값/없으면 9999.9로 대체해 숫자 보장
            prev_ttc = state["ttc"]
            ttc_in   = ttc_out if ttc_out < 9000.0 else (prev_ttc if (prev_ttc is not None) else 9999.9)
            ttc_f    = ema("ttc", ttc_in)
            ttc_send = (ttc_f if ttc_f is not None else 9999.9)   # 전송 값은 숫자로 보장 (핵심)
                        # >>> 누락된 EMA 계산 추가 <<<
            dist_f = ema("d",   dist)
            rel_f  = ema("vr",  rel)
            vle_f  = ema("vlead", lead_est)
            updates = {
                "Vehicle.ADAS.ACC.Distance":     Datapoint(float(dist_f)),
                "Vehicle.ADAS.ACC.RelSpeed":     Datapoint(float(rel_f)),
                "Vehicle.ADAS.ACC.TTC":          Datapoint(float(ttc_send)),
                "Vehicle.ADAS.ACC.HasTarget":    Datapoint(bool(has_target)),
                "Vehicle.ADAS.ACC.LeadSpeedEst": Datapoint(float(vle_f)),
            }
            kuksa.set_current_values(updates)

            # 타깃 메모리 갱신
            now = time.time()
            last.update({"ts": now, "d": dist_f, "rel": rel_f, "ttc": ttc_send, "has": bool(has_target), "vlead": vle_f})

            if args.radar_log:
                ttc_pr = "inf" if (ttc_send >= 9000.0) else f"{ttc_send:.1f}"
                print(f"[RADAR] d={dist_f:.1f} rel_corr={rel_f:+.2f} ttc={ttc_pr} "
                      f"ω={omega_z:+.2f}rad/s az={az_avg_deg:+.1f}° gate=±{gate_deg:.0f}° has={has_target}")

        except Exception:
            # 실패 시 '타깃 없음' 폴백
            updates = {
                "Vehicle.ADAS.ACC.Distance":     Datapoint(9999.9),
                "Vehicle.ADAS.ACC.RelSpeed":     Datapoint(0.0),
                "Vehicle.ADAS.ACC.TTC":          Datapoint(9999.9),
                "Vehicle.ADAS.ACC.HasTarget":    Datapoint(False),
                "Vehicle.ADAS.ACC.LeadSpeedEst": Datapoint(0.0),
            }
            try: kuksa.set_current_values(updates)
            except Exception: pass
            # 디버깅 필요 시: print("[RADAR] fallback:", repr(e))

    radar.listen(on_radar)

    # --- 표시 창 ---
    if args.display:
        cv2.namedWindow('front', cv2.WINDOW_NORMAL)
        if chase is not None:
            cv2.namedWindow('chase', cv2.WINDOW_NORMAL)

    print("[RUN] Streaming... (Ctrl+C to stop)")

    # --- 메인 루프 ---
    try:
        last_status = time.time()
        while True:
            world.tick()

            # 에고 속도 VSS 퍼블리시(결정에서 읽어 v 표시/게인에 활용)
            try:
                v = ego.get_velocity()
                ego_speed_mps = float(math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z))
                kuksa.set_current_values({"Vehicle.Speed": Datapoint(ego_speed_mps)})
            except Exception:
                pass

            now = time.time()
            if now - last_status >= STATUS_EVERY:
                v = ego.get_velocity()
                speed_kmh = 3.6 * math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)
                loc = ego.get_transform().location
                print(f"[STATUS] t={now:.0f}s ego@({loc.x:.1f},{loc.y:.1f}) {speed_kmh:.1f} km/h")
                last_status = now

            if args.display:
                if latest_front['bgr'] is not None:
                    cv2.imshow('front', latest_front['bgr'])
                if chase is not None and latest_chase['bgr'] is not None:
                    cv2.imshow('chase', latest_chase['bgr'])
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break

            time.sleep(0.003)  # CPU 여유
    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C")
    finally:
        # 정리
        try: cam.stop(); radar.stop()
        except Exception: pass
        for a in [cam, radar, lead, ego]:
            try: a.destroy()
            except Exception: pass
        try: pub.undeclare(); sess.close()
        except Exception: pass
        try: kuksa.disconnect()
        except Exception: pass
        world.apply_settings(original_settings)
        print("[CLEAN] Done.")

if __name__ == "__main__":
    main()
