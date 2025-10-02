#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test.py 
"""

import sys, time, math, threading, queue
from dataclasses import dataclass, field
from typing import List, Optional

import carla
import zenoh

# ===== Zenoh (reset) =====
cfg = zenoh.Config()
try:
    cfg.insert_json5("mode", '"client"')
    cfg.insert_json5("connect/endpoints", '["tcp/127.0.0.1:7447"]')
except AttributeError:
    cfg.insert_json("mode", '"client"')
    cfg.insert_json("connect/endpoints", '["tcp/127.0.0.1:7447"]')
z_sess = zenoh.open(cfg)
reset_pub = z_sess.declare_publisher("scenario/reset")

# =========================

# =========================
def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def deg_wrap180(a):
    return (a + 180.0) % 360.0 - 180.0

def norm2d(x, y):
    return math.sqrt(x*x + y*y)

def zero_vel(v: carla.Vehicle):
    try:
        v.set_velocity(carla.Vector3D(0,0,0))
        v.set_angular_velocity(carla.Vector3D(0,0,0))
    except:
        pass

def teleport(v: carla.Vehicle, tf: carla.Transform):
    v.set_simulate_physics(False)
    v.set_transform(tf)
    zero_vel(v)
    v.set_simulate_physics(True)

def lerp(a, b, t):
    
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    return a + (b - a) * t

def weather_lerp(base: carla.WeatherParameters, target: carla.WeatherParameters, alpha: float):
    return carla.WeatherParameters(
        cloudiness             = lerp(base.cloudiness,             target.cloudiness, alpha),
        precipitation          = lerp(base.precipitation,          target.precipitation, alpha),
        precipitation_deposits = lerp(base.precipitation_deposits, target.precipitation_deposits, alpha),
        wetness                = lerp(base.wetness,                target.wetness, alpha),
        wind_intensity         = lerp(base.wind_intensity,         target.wind_intensity, alpha),
        sun_azimuth_angle      = base.sun_azimuth_angle,
        sun_altitude_angle     = base.sun_altitude_angle,
        fog_density            = base.fog_density,
        fog_distance           = base.fog_distance,
        fog_falloff            = base.fog_falloff,
        mie_scattering_scale   = getattr(base, 'mie_scattering_scale', 0.0),
        rayleigh_scattering_scale = getattr(base, 'rayleigh_scattering_scale', 0.0),
        scattering_intensity   = getattr(base, 'scattering_intensity', 0.0),
        dust_storm             = getattr(base, 'dust_storm', 0.0),
    )

# =========================
# =========================
class TrackFrame:
    def __init__(self, start: carla.Location, end: carla.Location):
        self.set_line(start, end)
    def set_line(self, start: carla.Location, end: carla.Location):
        self.sx, self.sy = start.x, start.y
        self.ex, self.ey = end.x, end.y
        dx, dy = (self.ex - self.sx), (self.ey - self.sy)
        L = norm2d(dx, dy)
        self.L = L
        self.ux, self.uy = (1.0, 0.0) if L < 1e-6 else (dx/L, dy/L)
    def s_on_line(self, p: carla.Location) -> float:
        if self.L < 1e-6: return -1e9
        px, py = p.x - self.sx, p.y - self.sy
        return px*self.ux + py*self.uy

track_frame = TrackFrame(carla.Location(), carla.Location())

def build_s_frame_for_length(amap: carla.Map, base_tf: carla.Transform, length_m: float):
    wp = amap.get_waypoint(base_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    acc = 0.0; cur = wp
    while acc < length_m:
        nxt = cur.next(2.0)
        if not nxt: break
        acc += cur.transform.location.distance(nxt[0].transform.location)
        cur = nxt[0]
    track_frame.set_line(base_tf.location, cur.transform.location)

# =========================
# PID
# =========================
class PID:
    def __init__(self, kp, ki, kd, i_limit=2.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i = 0.0; self.i_limit = i_limit
        self.prev_e = 0.0; self.first = True
    def reset(self):
        self.i = 0.0; self.prev_e = 0.0; self.first = True
    def step(self, e, dt):
        self.i += e * dt
        self.i = clamp(self.i, -self.i_limit, self.i_limit)
        d = 0.0 if self.first else (e - self.prev_e) / max(dt, 1e-3)
        self.prev_e = e; self.first = False
        return self.kp*e + self.ki*self.i + self.kd*d

# =========================
# Lane-Lock
# =========================
@dataclass
class LaneLock:
    road_id: int
    lane_id: int

def build_lane_locked_path(amap: carla.Map, start_wp: carla.Waypoint,
                           lock: LaneLock, max_length_m=2000.0, step_m=2.0):
    path=[start_wp]; total=0.0; cur=start_wp
    while total < max_length_m:
        nxts = cur.next(step_m)
        if not nxts: break
        nxt = nxts[0]
        if (nxt.road_id != lock.road_id) or (nxt.lane_id != lock.lane_id):
            cand = [w for w in nxts if (w.road_id == lock.road_id and w.lane_id == lock.lane_id)]
            if cand: nxt = cand[0]
        path.append(nxt)
        total += step_m; cur = nxt
    return path

def nearest_index_on_path(path, loc, hint=0, window=50):
    if not path: return 0
    i = clamp(hint, 0, len(path)-2)
    lo = max(0, i-window); hi = min(len(path)-2, i+window)
    best_i = i; best_d2 = 1e18
    for k in range(lo, hi+1):
        p = path[k].transform.location
        d2 = (p.x - loc.x)**2 + (p.y - loc.y)**2
        if d2 < best_d2: best_d2 = d2; best_i = k
    return best_i

# =========================
# LKAS
# =========================
class LKAS:
    def __init__(self, k_e=0.9, k_h=1.0, k_soft=1.0, steer_rate=1.5, alpha=0.4):
        self.k_e=k_e; self.k_h=k_h; self.k_soft=k_soft
        self.steer_rate=steer_rate; self.alpha=alpha
        self.prev=0.0; self.inited=False
    def reset(self):
        self.prev=0.0; self.inited=False
    def step(self, path, idx, lock, loc, yaw_deg, speed, dt):
        idx = clamp(idx, 0, len(path)-2)
        wpa = path[idx].transform; wpb = path[idx+1].transform
        hx, hy = (wpb.location.x - wpa.location.x), (wpb.location.y - wpa.location.y)
        lane_yaw = math.atan2(hy, hx); lane_yaw_deg = math.degrees(lane_yaw)
        heading_err = deg_wrap180(lane_yaw_deg - yaw_deg)
        heading_err_rad = math.radians(heading_err)
        dx, dy = (loc.x - wpa.location.x), (loc.y - wpa.location.y)
        cross_track = -math.sin(lane_yaw)*dx + math.cos(lane_yaw)*dy
        soften = self.k_soft + max(0.1, speed)
        steer_cmd = self.k_h*heading_err_rad + math.atan2(-self.k_e * cross_track, soften)
        steer_cmd = clamp(steer_cmd, -1.2, 1.2)
        if not self.inited:
            filtered = steer_cmd; self.inited=True
        else:
            filtered = (1.0 - self.alpha) * steer_cmd + self.alpha * self.prev
            max_step = self.steer_rate * dt
            delta = clamp(filtered - self.prev, -max_step, max_step)
            filtered = self.prev + delta
        self.prev = filtered
        return float(clamp(filtered, -1.0, 1.0))

# =========================
# Scenario
# =========================
@dataclass
class Scenario:
    sid: int
    name: str
    route_type: str
    weather: Optional[carla.WeatherParameters]
    target_kmh: float
    decel_start: float
    stop_s: float
    stop_zones: List[float] = field(default_factory=list)

def get_scenario(sid: int, original_weather: carla.WeatherParameters) -> Scenario:
    if sid == 0:
        return Scenario(0, "S0", "mixed", carla.WeatherParameters.Default, 60.0, 250.0, 300.0, [])
    if sid == 1:
        return Scenario(1, "S1", "straight", carla.WeatherParameters.Default, 60.0, 250.0, 300.0, [])
    if sid == 2:
        return Scenario(2, "S2", "straight", carla.WeatherParameters.MidRainyNight, 60.0, 200.0, 250.0, [])
    if sid == 3:
        return Scenario(3, "S3", "straight", carla.WeatherParameters.WetSunset, 55.0, 200.0, 250.0, [])
    return get_scenario(0, original_weather)


def apply_weather_preset(world: carla.World, preset: carla.WeatherParameters):
    world.set_weather(preset)

def apply_slip_to_ego(ego: carla.Vehicle, scenario: Scenario, slip_k=0.9, dry_k=3.5):
    pc = ego.get_physics_control()
    try:
        pc.use_sweep_wheel_collision = True
        pc.substepping = True
        pc.max_substep_delta_time = 0.01
        pc.max_substeps = 10
    except Exception:
        pass

    wheels = pc.wheels
    if scenario.sid in (2, 3):
        for w in wheels:
            w.tire_friction = float(slip_k)
    else:
        for w in wheels:
            w.tire_friction = float(dry_k)

    pc.wheels = wheels
    ego.apply_physics_control(pc)
    print(f"[EGO] Slip applied (friction={pc.wheels[0].tire_friction}) for scenario {scenario.sid}")

# =========================

# =========================
def start_input_thread(cmd_q: "queue.Queue[str]"):
    def _run():
        while True:
            s = sys.stdin.readline()
            if not s: continue
            s = s.strip()
            if s in ("0","1","2","3","q","Q","d","D","w","W","i","I"):
                cmd_q.put(s)
    threading.Thread(target=_run, daemon=True).start()

# =========================

# =========================
def find_ego(world: carla.World):
    xs = [a for a in world.get_actors().filter("vehicle.*") if a.attributes.get("role_name")=="ego"]
    if not xs: raise RuntimeError("No ego vehicle found. Run env.py first.")
    return xs[0]

def find_lead(world: carla.World):
    xs = [a for a in world.get_actors().filter("vehicle.*") if a.attributes.get("role_name")=="lead"]
    if not xs: raise RuntimeError("No lead vehicle found. Run env.py first.")
    return xs[0]

# =========================

# =========================
def reset_positions(world, amap, ego, lead, scenario: Scenario, sps):
    if scenario.sid == 0:
        base_tf = sps[19]; build_s_frame_for_length(amap, base_tf, 700.0)
    else:
        tf53 = sps[53]
        wp = amap.get_waypoint(tf53.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        prevs = wp.previous(250.0)
        base_tf = (prevs[0] if prevs else wp).transform
        build_s_frame_for_length(amap, base_tf, 600.0)

    ego_tf = carla.Transform(carla.Location(base_tf.location.x, base_tf.location.y, base_tf.location.z+1.0),
                             base_tf.rotation)
    teleport(ego, ego_tf)

    wp_e = amap.get_waypoint(ego_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    fw = wp_e.next(30.0)
    if fw:
        lead_tf = fw[0].transform
        lead_tf.location.z = ego_tf.location.z + 1.0
        teleport(lead, lead_tf)

    zero_vel(ego); zero_vel(lead); time.sleep(0.3)

# =========================

# =========================
def quick_set_weather(world: carla.World, ego: carla.Vehicle, preset: carla.WeatherParameters, sid_for_log: int):
    apply_weather_preset(world, preset)
    # S1에서는 마찰을 바꾸지 않음(요청사항). 로그만.
    print(f"[WEATHER] set to preset ({preset}) for S{sid_for_log}")

# =========================

# =========================
def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--start_scenario", type=int, default=0)

    ap.add_argument("--auto_weather", type=int, default=1)
    ap.add_argument("--wet_start_s", type=float, default=150.0)   # DRY->WET 
    ap.add_argument("--wet_ramp_sec", type=float, default=3.0)   

    args=ap.parse_args()

    client=carla.Client(args.host, args.port); client.set_timeout(10.0)
    world=client.get_world(); amap=world.get_map()
    sps=amap.get_spawn_points(); original_weather=world.get_weather()
    bp_lib=world.get_blueprint_library()

    ego=find_ego(world); lead=find_lead(world)

    sensors=[]


    collision_bp = bp_lib.find('sensor.other.collision')
    collided = {"hit": False}
    def on_collision(ev):
        print(f"[COLLISION] lead hit {ev.other_actor.type_id}")
        collided["hit"] = True
    collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=lead)
    sensors.append(collision_sensor)
    collision_sensor.listen(on_collision)


    sid=int(args.start_scenario); scenario=get_scenario(sid, original_weather)
    reset_positions(world, amap, ego, lead, scenario, sps)
    apply_weather_preset(world, scenario.weather)
    apply_slip_to_ego(ego, scenario)  

 
    start_wp = amap.get_waypoint(lead.get_transform().location, project_to_road=True, lane_type=carla.LaneType.Driving)
    lane_lock = LaneLock(start_wp.road_id, start_wp.lane_id)
    path = build_lane_locked_path(amap, start_wp, lane_lock, 2000.0, 2.0)


    speed_pid = PID(0.60, 0.05, 0.02)
    lkas = LKAS()
    idx_hint=0

    # kmh_cmd 
    kmh_cmd_smooth = scenario.target_kmh

    print(f"[SCENARIO] {scenario.name}  (0/1/2/3 전환, q 종료)")

    # S3 stop zone 
    sz = {"visited": set(), "phase":"none", "t0":0.0, "v0":0.0, "hold_until":0.0}


    cmd_q: "queue.Queue[str]" = queue.Queue()
    start_input_thread(cmd_q)

    # S1 DRY->WET 
    aw = {"phase": 0, "t0": 0.0, "w0": world.get_weather(), "w1": None}

    last_dt=0.05
    last_print=0.0
    try:
        while True:
            snap=world.wait_for_tick()
            dt=snap.timestamp.delta_seconds if snap else last_dt; last_dt=dt
            now=time.time()


            try: cmd=cmd_q.get_nowait()
            except queue.Empty: cmd=None
            if cmd in ("q","Q"): break
            if cmd in ("0","1","2","3"):

                for s in sensors:
                    try: s.stop(); s.destroy()
                    except: pass
                sensors.clear()
                collided["hit"]=False

                speed_pid.reset(); lkas.reset(); idx_hint=0


                sid=int(cmd); scenario=get_scenario(sid, original_weather)
                reset_positions(world, amap, ego, lead, scenario, sps)
                apply_weather_preset(world, scenario.weather)
                apply_slip_to_ego(ego, scenario)
                reset_pub.put(b"reset")

                start_wp = amap.get_waypoint(lead.get_transform().location, project_to_road=True, lane_type=carla.LaneType.Driving)
                lane_lock = LaneLock(start_wp.road_id, start_wp.lane_id)
                path = build_lane_locked_path(amap, start_wp, lane_lock, 2000.0, 2.0)

                collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=lead)
                sensors.append(collision_sensor)
                collision_sensor.listen(on_collision)

                kmh_cmd_smooth = scenario.target_kmh
                print(f"\n[SCENARIO] {scenario.name}")


                aw = {"phase": 0, "t0": 0.0, "w0": world.get_weather(), "w1": None}
                continue


            if cmd in ("d","D"):  # DRY
                quick_set_weather(world, ego, carla.WeatherParameters.Default, sid_for_log=scenario.sid); continue
            if cmd in ("w","W"):  # WET
                quick_set_weather(world, ego, carla.WeatherParameters.WetCloudyNoon, sid_for_log=scenario.sid); continue
            if cmd in ("i","I"):  # ICY 
                quick_set_weather(world, ego, carla.WeatherParameters.WetSunset, sid_for_log=scenario.sid); continue


            tf=lead.get_transform(); loc=tf.location; yaw=tf.rotation.yaw
            v=lead.get_velocity(); speed=math.sqrt(v.x*v.x+v.y*v.y+v.z*v.z); speed_kmh=3.6*speed
            s=track_frame.s_on_line(loc)


            if now - last_print >= 2.0:
                print(f"[INFO] Speed = {speed_kmh:.1f} km/h, s = {s:.1f} m")
                last_print = now


            if collided["hit"]:
                lead.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=True))
                speed_pid.reset(); lkas.reset()
                continue

            kmh_target = scenario.target_kmh

            # ===== S3 stop zones: -4 m/s² → 5s hold → resume =====
            if scenario.sid == 3:
                if sz["phase"] == "none":
                    for ss in scenario.stop_zones:
                        if ss in sz["visited"]: continue
                        if abs(s - ss) <= 15.0:
                            print(f"[EVENT] stop zone {ss:.0f} m: decel -4 m/s², then 5s hold")
                            sz["visited"].add(ss)
                            sz["phase"]="decel"; sz["t0"]=now; sz["v0"]=speed
                            break
                if sz["phase"] == "decel":
                    t = max(0.0, now - sz["t0"])
                    v_target = max(0.0, sz["v0"] - 4.0*t)  # m/s
                    kmh_target = min(kmh_target, v_target*3.6)
                    if (v_target <= 0.2) and (speed <= 0.3):
                        sz["phase"]="hold"; sz["hold_until"]=now + 5.0
                        speed_pid.reset(); kmh_target=0.0
                elif sz["phase"] == "hold":
                    kmh_target = 0.0
                    lead.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=True))
                    if now >= sz["hold_until"]:
                        sz["phase"]="none"
                        speed_pid.reset()
                    continue 

            # ==========
            if scenario.decel_start <= s < scenario.stop_s:
                v_allow = scenario.target_kmh * (scenario.stop_s - s) / max((scenario.stop_s - scenario.decel_start), 1e-3)
                kmh_target = min(kmh_target, v_allow)
            elif s >= scenario.stop_s:
                lead.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=True))
                speed_pid.reset(); lkas.reset()
                continue


            try:
                wp_now = amap.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                nxt = wp_now.next(60.0)
                if nxt:
                    yaw_next = nxt[0].transform.rotation.yaw
                    yaw_diff = abs(deg_wrap180(yaw_next - wp_now.transform.rotation.yaw))
                    if yaw_diff > 30: corner_kmh = 22.0
                    elif yaw_diff > 18: corner_kmh = 28.0
                    elif yaw_diff > 12: corner_kmh = 35.0
                    else: corner_kmh = kmh_target
                    kmh_target = min(kmh_target, corner_kmh)
            except:
                pass

            max_down = 10.0 * dt
            max_up   = 15.0 * dt
            if kmh_target < kmh_cmd_smooth:
                kmh_cmd_smooth = max(kmh_target, kmh_cmd_smooth - max_down)
            else:
                kmh_cmd_smooth = min(kmh_target, kmh_cmd_smooth + max_up)

            if idx_hint > len(path) - 80:
                start_wp = amap.get_waypoint(tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
                lane_lock = LaneLock(start_wp.road_id, start_wp.lane_id)
                path = build_lane_locked_path(amap, start_wp, lane_lock, 2000.0, 2.0)
                idx_hint = 0

            idx_hint = nearest_index_on_path(path, loc, hint=idx_hint)
            steer = lkas.step(path, idx_hint, lane_lock, loc, yaw, speed, dt)

            v_ref = kmh_cmd_smooth / 3.6
            a_cmd = speed_pid.step(v_ref - speed, dt)
            throttle=0.0; brake=0.0
            if a_cmd >= 0: throttle = clamp(a_cmd, 0.0, 0.65)
            else: brake = clamp(-a_cmd, 0.0, 1.0)

            lead.apply_control(carla.VehicleControl(throttle=float(throttle),
                                                    steer=float(steer),
                                                    brake=float(brake)))

            # ===== S1 DRY->WET =====
            if args.auto_weather and scenario.sid == 1:
                if aw["phase"] == 0 and s >= args.wet_start_s:
                    aw["phase"] = 1
                    aw["t0"] = now
                    aw["w0"] = world.get_weather()
                    aw["w1"] = carla.WeatherParameters.HardRainNoon 

                if aw["phase"] == 1:
                    a = (now - aw["t0"]) / max(0.001, args.wet_ramp_sec)
                    if a >= 1.0:
                        world.set_weather(aw["w1"])
                        aw["phase"] = 2
                    else:
                        world.set_weather(weather_lerp(aw["w0"], aw["w1"], a))

    finally:
        try:
            for s in sensors:
                try: s.stop(); s.destroy()
                except: pass
        except: pass

        try: world.set_weather(original_weather)
        except: pass
        print("[CLEANUP]")

if __name__=="__main__":
    main()
