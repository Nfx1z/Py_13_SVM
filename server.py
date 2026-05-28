#!/usr/bin/env python3
"""
silat_server.py  —  Flask Inference + Data Collection Server
=============================================================
Endpoints:
  POST /data      ← ESP32 sends raw sensor row here (unchanged format)
  GET  /predict   ← Returns prediction for the current window
  POST /config    ← Update window_size and none_threshold at runtime
  GET  /status    ← Buffer status (how many samples collected)
  GET  /stream    ← SSE stream: auto-pushes predictions every ~500 ms
  GET  /          ← Serves the dashboard HTML (silat_dashboard.html)

Usage:
  python silat_server.py

Then open:  http://localhost:5000
"""

import os, json, time, threading, queue, collections
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from flask import Flask, request, jsonify, Response, send_from_directory
from scipy import stats as scipy_stats

# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════
SERVER_CONFIG = {
    "model_dir"      : "./model",
    "data_log_file"  : "./data_realtime.csv",   # keeps writing like before
    "host"           : "0.0.0.0",
    "port"           : 5000,

    # ── Inference defaults (adjustable at runtime via /config) ───
    "window_size"    : 30,     # number of samples per prediction window
    "none_threshold" : 0.40,   # Model 0 confidence below this → "None"
    "buffer_maxlen"  : 500,    # rolling buffer size (keep last N samples)
}

MOVEMENT_CLASSES = [
    "Jep Kiri", "Jep Kanan",
    "Tangkisan Kiri", "Tangkisan Kanan",
    "Kombinasi Kiri", "Kombinasi Kanan",
]

# CSV column names (must match what ESP32 sends)
SENSOR_COLS = []
for i in range(4):
    for a in "xyz":
        SENSOR_COLS.append(f"a{a}{i}")
    for a in "xyz":
        SENSOR_COLS.append(f"g{a}{i}")


# ══════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION  (identical to silat_train.py)
# ══════════════════════════════════════════════════════════════════

N_SENSORS = 4
ACCEL_COLS = [[f"a{a}{i}" for a in "xyz"] for i in range(N_SENSORS)]
GYRO_COLS  = [[f"g{a}{i}" for a in "xyz"] for i in range(N_SENSORS)]


def _channel_stats(x):
    if len(x) == 0:
        return [0.0] * 12
    mean = np.mean(x); std = np.std(x)
    xmin = np.min(x);  xmax = np.max(x)
    rms  = np.sqrt(np.mean(x ** 2))
    rng  = xmax - xmin
    med  = np.median(x)
    iqr  = np.percentile(x, 75) - np.percentile(x, 25)
    skew = float(scipy_stats.skew(x))     if len(x) > 2 else 0.0
    kurt = float(scipy_stats.kurtosis(x)) if len(x) > 3 else 0.0
    fft_e = np.sum(np.abs(np.fft.rfft(x)) ** 2) / len(x)
    zcr   = np.sum(np.abs(np.diff(np.sign(x - mean)))) / (2 * max(len(x)-1, 1))
    return [mean, std, xmin, xmax, rms, rng, med, iqr, skew, kurt, fft_e, zcr]


def _safe_corr(a, b):
    if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _accel_mag(data, cols, si):
    idx = [cols.index(c) for c in ACCEL_COLS[si] if c in cols]
    return np.sqrt(np.sum(data[:, idx] ** 2, axis=1)) if idx else np.zeros(len(data))


def _gyro_mag(data, cols, si):
    idx = [cols.index(c) for c in GYRO_COLS[si] if c in cols]
    return np.sqrt(np.sum(data[:, idx] ** 2, axis=1)) if idx else np.zeros(len(data))


def extract_features(df_window: pd.DataFrame) -> np.ndarray:
    """426 topology-aware features from a window DataFrame."""
    data = df_window[SENSOR_COLS].values.astype(float)
    cols = SENSOR_COLS
    n    = len(data)
    EPS  = 1e-9
    f    = []

    for i in range(data.shape[1]):
        f.extend(_channel_stats(data[:, i]))

    s0a = _accel_mag(data, cols, 0); s1a = _accel_mag(data, cols, 1)
    s2a = _accel_mag(data, cols, 2); s3a = _accel_mag(data, cols, 3)
    s0g = _gyro_mag(data, cols, 0);  s1g = _gyro_mag(data, cols, 1)
    s2g = _gyro_mag(data, cols, 2);  s3g = _gyro_mag(data, cols, 3)
    for am, gm in [(s0a, s0g), (s1a, s1g), (s2a, s2g), (s3a, s3g)]:
        f.extend(_channel_stats(am)); f.extend(_channel_stats(gm))

    ra = (s0a + s1a) / 2; la = (s2a + s3a) / 2
    rg = (s0g + s1g) / 2; lg = (s2g + s3g) / 2
    f.extend(_channel_stats(ra - la))
    f.extend(_channel_stats(rg - lg))
    f.append(np.mean(s0a) / (np.mean(s0a) + np.mean(s1a) + EPS))
    f.append(np.mean(s3a) / (np.mean(s3a) + np.mean(s2a) + EPS))
    f.append(np.mean(s0g) / (np.mean(s0g) + np.mean(s1g) + EPS))
    f.append(np.mean(s3g) / (np.mean(s3g) + np.mean(s2g) + EPS))
    f.append(_safe_corr(s0a, s1a)); f.append(_safe_corr(s3a, s2a))
    f.append(_safe_corr(s0g, s1g)); f.append(_safe_corr(s3g, s2g))
    f.append(float(np.max(ra)) / (float(np.max(la)) + EPS))
    f.append(float(np.max(rg)) / (float(np.max(lg)) + EPS))
    f.append(float(np.argmax(s0a)) / max(n-1, 1))
    f.append(float(np.argmax(s1a)) / max(n-1, 1))
    f.append(float(np.argmax(s3a)) / max(n-1, 1))
    f.append(float(np.argmax(s2a)) / max(n-1, 1))
    f.append(float(np.mean(ra**2))); f.append(float(np.mean(la**2)))
    f.append(float(np.mean(rg**2))); f.append(float(np.mean(lg**2)))

    return np.array(f, dtype=np.float64)


# ══════════════════════════════════════════════════════════════════
#  MODEL LOADER
# ══════════════════════════════════════════════════════════════════

class ModelBank:
    """Loads and caches all 7 SVM models from disk."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.movement_model = None
        self.correctness_models: dict = {}
        self._load()

    def _load(self):
        mv_path = os.path.join(self.model_dir, "model_movement.pkl")
        if not os.path.exists(mv_path):
            raise FileNotFoundError(
                f"Movement model not found at {mv_path}. "
                "Run silat_train.py first."
            )
        self.movement_model = joblib.load(mv_path)
        print(f"  [OK] Movement model loaded")

        for mv in MOVEMENT_CLASSES:
            key  = mv.lower().replace(" ", "_")
            path = os.path.join(self.model_dir, f"model_{key}.pkl")
            if os.path.exists(path):
                self.correctness_models[mv] = joblib.load(path)
                print(f"  [OK] Correctness model: {mv}")

    def predict(self, feat: np.ndarray, none_threshold: float) -> dict:
        """
        Run 2-stage inference on a 426-feature vector.

        Returns a dict with movement, correctness, confidences.
        """
        feat_2d = feat.reshape(1, -1)

        # Stage 1 — movement
        proba   = self.movement_model.predict_proba(feat_2d)[0]
        classes = self.movement_model.classes_
        best_i  = int(np.argmax(proba))
        max_p   = float(proba[best_i])
        pred_mv = str(classes[best_i])

        mv_probs = {str(c): round(float(p), 4) for c, p in zip(classes, proba)}

        if max_p < none_threshold:
            return {
                "movement"        : "None",
                "correctness"     : None,
                "mv_confidence"   : round(max_p, 4),
                "mv_probabilities": mv_probs,
                "timestamp"       : datetime.now().isoformat(),
            }

        # Stage 2 — correctness
        correctness = None
        if pred_mv in self.correctness_models:
            c_model = self.correctness_models[pred_mv]
            c_proba = c_model.predict_proba(feat_2d)[0]
            c_pred  = int(c_model.predict(feat_2d)[0])
            correctness = {
                "label"     : "Benar" if c_pred == 1 else "Salah",
                "confidence": round(float(np.max(c_proba)), 4),
            }

        return {
            "movement"        : pred_mv,
            "correctness"     : correctness,
            "mv_confidence"   : round(max_p, 4),
            "mv_probabilities": mv_probs,
            "timestamp"       : datetime.now().isoformat(),
        }


# ══════════════════════════════════════════════════════════════════
#  FLASK APPLICATION
# ══════════════════════════════════════════════════════════════════

app = Flask(__name__)

# ── Shared state ─────────────────────────────────────────────────
_buffer      = collections.deque(maxlen=SERVER_CONFIG["buffer_maxlen"])
_buffer_lock = threading.Lock()
_sse_queue   = queue.Queue(maxsize=50)  # for SSE broadcast
_last_result = {}
_runtime_cfg = {
    "window_size"    : SERVER_CONFIG["window_size"],
    "none_threshold" : SERVER_CONFIG["none_threshold"],
}

# Load models once at startup
print("Loading SVM models…")
try:
    models = ModelBank(SERVER_CONFIG["model_dir"])
    print("All models loaded.\n")
except FileNotFoundError as e:
    print(f"[ERROR] {e}")
    models = None

# Ensure CSV header exists
if not os.path.exists(SERVER_CONFIG["data_log_file"]):
    with open(SERVER_CONFIG["data_log_file"], "w") as f:
        f.write("timestamp," + ",".join(SENSOR_COLS) + "\n")


# ── /data  —  receive ESP32 POST ─────────────────────────────────
@app.route("/data", methods=["POST"])
def receive_data():
    raw = request.form.get("data", "")
    if not raw:
        return "missing data", 400

    vals = raw.split(",")
    if len(vals) != 24:
        return f"expected 24 values, got {len(vals)}", 400

    try:
        nums = [float(v) for v in vals]
    except ValueError:
        return "non-numeric value", 400

    ts  = datetime.now()
    row = dict(zip(SENSOR_COLS, nums))
    row["timestamp"] = ts

    # Write to CSV log (keep same file as before)
    with open(SERVER_CONFIG["data_log_file"], "a") as f:
        f.write(ts.strftime("%Y-%m-%d %H:%M:%S.%f") + "," +
                ",".join(f"{v:.4f}" for v in nums) + "\n")

    # Add to rolling buffer
    with _buffer_lock:
        _buffer.append(row)

    return "OK", 200


# ── /predict  —  run inference on current window ─────────────────
@app.route("/predict", methods=["GET"])
def predict():
    if models is None:
        return jsonify({"error": "Models not loaded. Run silat_train.py first."}), 503

    win = _runtime_cfg["window_size"]
    thr = _runtime_cfg["none_threshold"]

    with _buffer_lock:
        buf_list = list(_buffer)

    if len(buf_list) < win:
        return jsonify({
            "status"     : "buffering",
            "buffer_size": len(buf_list),
            "need"       : win,
        }), 202

    # Take the most recent `window_size` samples
    window_rows = buf_list[-win:]
    df_win = pd.DataFrame(window_rows)[SENSOR_COLS]

    try:
        feat   = extract_features(df_win)
        result = models.predict(feat, thr)
        result["buffer_size"] = len(buf_list)
        result["window_size"] = win
        result["none_threshold"] = thr

        global _last_result
        _last_result = result

        # Push to SSE queue (non-blocking)
        try:
            _sse_queue.put_nowait(result)
        except queue.Full:
            pass

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── /config  —  update window_size and none_threshold ────────────
@app.route("/config", methods=["POST"])
def update_config():
    body = request.get_json(silent=True) or {}

    if "window_size" in body:
        ws = int(body["window_size"])
        if ws < 5 or ws > SERVER_CONFIG["buffer_maxlen"]:
            return jsonify({"error": f"window_size must be 5–{SERVER_CONFIG['buffer_maxlen']}"}), 400
        _runtime_cfg["window_size"] = ws

    if "none_threshold" in body:
        thr = float(body["none_threshold"])
        if not (0.0 <= thr <= 1.0):
            return jsonify({"error": "none_threshold must be 0.0–1.0"}), 400
        _runtime_cfg["none_threshold"] = thr

    return jsonify({
        "status": "updated",
        "config": _runtime_cfg,
    }), 200


# ── /status  —  buffer info ───────────────────────────────────────
@app.route("/status", methods=["GET"])
def status():
    with _buffer_lock:
        n = len(_buffer)
    return jsonify({
        "buffer_size"    : n,
        "buffer_maxlen"  : SERVER_CONFIG["buffer_maxlen"],
        "window_size"    : _runtime_cfg["window_size"],
        "none_threshold" : _runtime_cfg["none_threshold"],
        "models_loaded"  : models is not None,
        "movement_classes": MOVEMENT_CLASSES,
    }), 200


# ── /stream  —  SSE: auto-push predictions ────────────────────────
@app.route("/stream", methods=["GET"])
def stream():
    """
    Server-Sent Events endpoint.
    Every 500 ms: runs inference and pushes result to all connected clients.
    """
    def event_generator():
        while True:
            # Run inference on current window
            if models is not None:
                win = _runtime_cfg["window_size"]
                thr = _runtime_cfg["none_threshold"]
                with _buffer_lock:
                    buf_list = list(_buffer)

                if len(buf_list) >= win:
                    df_win = pd.DataFrame(buf_list[-win:])[SENSOR_COLS]
                    try:
                        feat   = extract_features(df_win)
                        result = models.predict(feat, thr)
                        result["buffer_size"]    = len(buf_list)
                        result["window_size"]    = win
                        result["none_threshold"] = thr
                    except Exception as e:
                        result = {"error": str(e)}
                else:
                    result = {
                        "status"     : "buffering",
                        "buffer_size": len(buf_list),
                        "need"       : win,
                    }
            else:
                result = {"error": "Models not loaded"}

            yield f"data: {json.dumps(result)}\n\n"
            time.sleep(0.5)

    return Response(event_generator(),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


# ── /  —  serve dashboard ─────────────────────────────────────────
@app.route("/")
def dashboard():
    html_path = os.path.join(os.path.dirname(__file__), "silat_dashboard.html")
    if os.path.exists(html_path):
        return send_from_directory(os.path.dirname(html_path), "silat_dashboard.html")
    return "<h2>Dashboard not found. Place silat_dashboard.html next to silat_server.py</h2>", 404


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\nStarting Silat Inference Server on port {SERVER_CONFIG['port']}…")
    print(f"  Dashboard  → http://localhost:{SERVER_CONFIG['port']}/")
    print(f"  Data in    → POST http://<your-ip>:{SERVER_CONFIG['port']}/data")
    print(f"  Predict    → GET  http://localhost:{SERVER_CONFIG['port']}/predict")
    print(f"  Config     → POST http://localhost:{SERVER_CONFIG['port']}/config")
    print()
    app.run(
        host=SERVER_CONFIG["host"],
        port=SERVER_CONFIG["port"],
        debug=False,
        threaded=True,
    )