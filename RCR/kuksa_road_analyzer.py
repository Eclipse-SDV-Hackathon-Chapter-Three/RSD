# -*- coding: utf-8 -*-
# kuksa_road_analyzer.py (latest-only consumer)
#
# Zenoh subscriber → 항상 최신 프레임만 덮어쓰기 → 메인 루프에서 처리
# 큐에 쌓이는 문제 없이 최신 상태만 반영됨

import os, sys, json, time, traceback
import numpy as np
import cv2
import zenoh
from kuksa_client.grpc import VSSClient, Datapoint

# ================== 환경변수 설정 ==================
IN_KEY        = os.environ.get("IN_KEY", "carla/cam/front")
KUKSA_HOST    = os.environ.get("KUKSA_HOST", "127.0.0.1")
KUKSA_PORT    = int(os.environ.get("KUKSA_PORT", "55555"))

CAM_DT        = float(os.environ.get("CAM_DT", "0.05"))
FRAME_INTERVAL= int(os.environ.get("FRAME_INTERVAL", "1"))
FORCE_QUANT   = os.environ.get("FORCE_QUANTIZE", "1") == "1"

LINE_WRAP     = int(os.environ.get("LINE_WRAP", "8"))
SAVE_DEBUG    = os.environ.get("SAVE_DEBUG", "0") == "1"
EMA_A         = float(os.environ.get("EMA_A", "0.3"))

ROI_Y0=float(os.environ.get("ROI_Y0","0.5"))
ROI_XL=float(os.environ.get("ROI_XL","0.2"))
ROI_XR=float(os.environ.get("ROI_XR","0.8"))

TH_SRI_WET=float(os.environ.get("TH_SRI_WET","0.08"))
TH_LV_WET=float(os.environ.get("TH_LV_WET","800"))
TH_SRI_ICY=float(os.environ.get("TH_SRI_ICY","0.00")) #0.00
TH_S_MEAN_I=float(os.environ.get("TH_S_MEAN_I","0.12"))
TH_V_MEAN_I=float(os.environ.get("TH_V_MEAN_I","0.60"))
TH_DR_SNOW=float(os.environ.get("TH_DR_SNOW","0.96"))
TH_S_MEAN_S=float(os.environ.get("TH_S_MEAN_S","0.20"))

# ================== 유틸 ==================
def _to_bytes(x):
    if x is None: return b""
    if isinstance(x, (bytes, bytearray, memoryview)): return bytes(x)
    try: return bytes(x)
    except Exception: return b""

def _attachment_bytes(sample):
    att_attr = getattr(sample, "attachment", None)
    att_obj  = att_attr() if callable(att_attr) else att_attr
    return _to_bytes(att_obj)

def _payload_buffer(sample):
    pay = getattr(sample, "payload", None)
    try: return memoryview(pay)
    except TypeError: return memoryview(_to_bytes(pay))

class EMA:
    def __init__(self,a): self.a=a; self.v={}
    def update(self,d):
        out={}
        for k,v in d.items():
            p=self.v.get(k); self.v[k]=v if p is None else (1-self.a)*p+self.a*v
            out[k]=self.v[k]
        return out

def compute_metrics(bgr, roi_box):
    x0,y0,x1,y1 = roi_box
    roi = bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsv)
    Sn = S.astype(np.float32)/255.0
    Vn = V.astype(np.float32)/255.0
    highlight = (Vn > 0.70) & (Sn < 0.35)
    SRI = float(highlight.mean())
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    LV = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    ED = float((cv2.Canny(gray,80,160) > 0).mean())
    DR = float((gray < 50).mean())
    S_mean = float(Sn.mean())
    V_mean = float(Vn.mean())
    return dict(SRI=SRI, LV=LV, ED=ED, DR=DR, S_mean=S_mean, V_mean=V_mean)

_last_state = "unknown"  # 함수 바깥에 전역 변수 하나 둬야 함

def classify(m):
    import numpy as np
    if m['SRI'] > TH_SRI_ICY and m['S_mean'] < TH_S_MEAN_I and m['V_mean'] > TH_V_MEAN_I:
        st = 'wet' #icy
    # dry 조건을 살짝 더 빡세게 → wet 유지 강화
    elif (m['SRI'] > TH_SRI_WET) or (m['LV'] < TH_LV_WET*1.7 and m['V_mean'] > 0.55):
        st = 'dry'
    elif m['DR'] > TH_DR_SNOW and m['S_mean'] < TH_S_MEAN_S:
        st = 'wet' #snow
    else:
        st = 'wet'

    # confidence 계산은 기존 방식 유지
    if st == 'wet':
        conf = min(1.0, 0.6
                        + max(0,(m['SRI']-TH_SRI_WET)*6)
                        + max(0,(TH_LV_WET-m['LV'])/200)
                        + max(0,(m['V_mean']-0.5)*1.0))
    elif st == 'wet': #icy
        conf = min(1.0, 0.5
                        + max(0,(m['SRI']-TH_SRI_ICY)*5)
                        + max(0,(TH_S_MEAN_I-m['S_mean'])*2)
                        + max(0,(m['V_mean']-TH_V_MEAN_I)*1.5))
    elif st == 'wet': #snow
        conf = min(1.0, 0.5
                        + max(0,(m['DR']-TH_DR_SNOW)*1.5)
                        + max(0,(TH_S_MEAN_S-m['S_mean'])*1.5))
    else:  # dry
        conf = min(1.0, 0.5
                        + max(0,(m['LV']-70)/120)
                        + max(0,(TH_SRI_WET-m['SRI'])*4))

    return st, float(np.clip(conf,0.0,1.0))


    
def overlay(bgr, m, roi_box, st, conf):
    x0,y0,x1,y1=roi_box; vis=bgr.copy()
    cv2.rectangle(vis,(x0,y0),(x1,y1),(0,255,0),2)
    t=f"{st}({conf:.2f}) SRI:{m['SRI']:.3f} LV:{m['LV']:.1f} ED:{m['ED']:.3f}"
    cv2.putText(vis,t,(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2,cv2.LINE_AA)
    return vis

# ================== 심시간 처리 ==================
_last_ts = None
_last_state, _last_conf = "unknown", 0.40
_last_sri, _last_ed = None, None

def derive_sim_ts(meta: dict) -> float:
    global _last_ts
    sim_ts = None
    v = meta.get("sim_ts")
    if v is not None:
        try: sim_ts = float(v)
        except Exception: sim_ts = None
    if sim_ts is None:
        fr = meta.get("frame")
        if fr is not None:
            try: sim_ts = int(fr) * CAM_DT
            except Exception: sim_ts = None
    if sim_ts is None:
        sim_ts = 0.0 if _last_ts is None else (_last_ts + CAM_DT)
    if FORCE_QUANT and CAM_DT > 0:
        n = int(round(sim_ts / CAM_DT))
        sim_ts = round(n * CAM_DT, 6)
    _last_ts = sim_ts
    return sim_ts

# ================== 메인 ==================
def main():
    kc = VSSClient(KUKSA_HOST, KUKSA_PORT); kc.connect()
    print(f"[Analyzer] Kuksa connected @ {KUKSA_HOST}:{KUKSA_PORT}")

    zcfg = zenoh.Config()
    try: zcfg.insert_json5("connect/endpoints",'["tcp/127.0.0.1:7447"]')
    except AttributeError: zcfg.insert_json("connect/endpoints",'["tcp/127.0.0.1:7447"]')
    sess = zenoh.open(zcfg)
    print(f"[Analyzer] subscribing raw: {IN_KEY} (Δt={CAM_DT}s)")

    ema = EMA(EMA_A)
    frame_cnt = 0

    # === latest-only 버퍼 ===
    latest_sample = {"sample": None}

    def publish_cam_vss(sim_ts, st=None, cf=None, sri=None, ed=None):
        global _last_state, _last_conf, _last_sri, _last_ed
        if st is None: st = _last_state
        if cf is None: cf = _last_conf
        updates = {
            "Vehicle.Private.Road.State":      Datapoint(st),
            "Vehicle.Private.Road.Confidence": Datapoint(float(cf)),
            "Vehicle.Private.Road.Ts":         Datapoint(float(sim_ts)),
        }
        if sri is not None:
            updates["Vehicle.Private.Road.Metrics.SRI"] = Datapoint(float(sri))
        if ed is not None:
            updates["Vehicle.Private.Road.Metrics.ED"]  = Datapoint(float(ed))
        kc.set_current_values(updates)
        _last_state, _last_conf = st, float(cf)
        if sri is not None: _last_sri = float(sri)
        if ed  is not None: _last_ed  = float(ed)

    def on_raw(sample: zenoh.Sample):
        latest_sample["sample"] = sample   # 큐에 쌓지 않고 덮어쓰기

    sess.declare_subscriber(IN_KEY, on_raw)

    try:
        while True:
            if latest_sample["sample"] is not None:
                sample = latest_sample["sample"]
                latest_sample["sample"] = None
                try:
                    meta_b = _attachment_bytes(sample)
                    meta = json.loads(meta_b.decode("utf-8")) if meta_b else {}
                    w = int(meta.get("w", 0)); h = int(meta.get("h", 0))
                    c = int(meta.get("c", 4)); stride = int(meta.get("stride", max(0,w*c)))
                    sim_ts = derive_sim_ts(meta)

                    do_infer = (FRAME_INTERVAL <= 1) or ((frame_cnt % FRAME_INTERVAL) == 0)
                    valid_img = (w>0 and h>0 and c>=3 and stride == w*c)

                    st=cf=None
                    if do_infer and valid_img:
                        buf = _payload_buffer(sample)
                        bgra = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, c))
                        bgr = bgra[:,:,:3]
                        y0=int(h*ROI_Y0); y1=h; x0=int(w*ROI_XL); x1=int(w*ROI_XR)
                        m = compute_metrics(bgr, (x0,y0,x1,y1))
                        m = ema.update(m)
                        st, cf = classify(m)
                        publish_cam_vss(sim_ts, st, cf, sri=m["SRI"], ed=m["ED"])
                        if SAVE_DEBUG:
                            vis = overlay(bgr, m, (x0,y0,x1,y1), st, cf)
                            cv2.imwrite("debug_latest.jpg", vis)
                    else:
                        publish_cam_vss(sim_ts)

                    use_state = (st if do_infer and st is not None else _last_state)
                    use_conf  = (cf if do_infer and cf is not None else _last_conf)
                    use_sri   = (_last_sri if _last_sri is not None else 0.0)
                    use_ed    = (_last_ed  if _last_ed  is not None else 0.0)

                    #print(f"[STATE] {use_state} conf={use_conf:.2f} SRI={use_sri:.3f} ED={use_ed:.3f} ts={sim_ts:.3f}")
                    print(
                            "[METRIC]",
                            f"ts={sim_ts:.2f}",
                            f"state={st}({cf:.2f})",
                            f"SRI={m['SRI']:.3f}",
                            f"LV={m['LV']:.1f}",
                            f"ED={m['ED']:.3f}",
                            f"DR={m['DR']:.3f}",
                            f"S_mean={m['S_mean']:.3f}",
                            f"V_mean={m['V_mean']:.3f}"
                        )
                    frame_cnt += 1
                except Exception:
                    print("[Analyzer] processing error:")
                    traceback.print_exc()
            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        try: kc.disconnect()
        except: pass
        sess.close()

if __name__ == "__main__":
    main()

