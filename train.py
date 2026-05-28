#!/usr/bin/env python3
import os, joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, accuracy_score
from scipy import stats as scipy_stats

# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════
CONFIG = {
    "data_dir"      : "./data",
    "file_pattern"  : "subjek_{id}.csv",
    "model_save_dir": "./model",
    "random_state"  : 42,
}

MOVEMENT_CLASSES = [
    "Jep Kiri", "Jep Kanan",
    "Tangkisan Kiri", "Tangkisan Kanan",
    "Kombinasi Kiri", "Kombinasi Kanan",
]

# ══════════════════════════════════════════════════════════════════
#  LABEL DATA
# ══════════════════════════════════════════════════════════════════
RAW_LABELS = [
    (1, "2026-04-30", "13.44.31,069", "13.44.34,855", "Jep Kiri",      "Benar"),
    (1, "2026-04-30", "13.45.01,092", "13.45.06,498", "Jep Kanan",     "Benar"),
    (1, "2026-04-30", "13.45.30,258", "13.45.35,273", "Jep Kiri",      "Salah"),
    (1, "2026-04-30", "13.45.49,611", "13.45.54,609", "Jep Kanan",     "Salah"),
    (1, "2026-04-30", "13.46.11,124", "13.46.22,535", "Tangkisan Kiri","Benar"),
    (1, "2026-04-30", "13.46.38,449", "13.46.44,622", "Tangkisan Kanan","Benar"),
    (1, "2026-04-30", "13.47.00,623", "13.47.07,020", "Tangkisan Kiri","Salah"),
    (1, "2026-04-30", "13.47.23,820", "13.47.29,843", "Tangkisan Kanan","Salah"),
    (1, "2026-04-30", "13.47.43,737", "13.47.49,320", "Kombinasi Kiri","Benar"),
    (1, "2026-04-30", "13.48.05,540", "13.48.12,940", "Kombinasi Kanan","Benar"),
    (1, "2026-04-30", "13.48.42,263", "13.48.51,273", "Kombinasi Kiri","Salah"),
    (1, "2026-04-30", "13.49.07,247", "13.49.12,302", "Kombinasi Kanan","Salah"),

    (2, "2026-05-01", "16.23.41,027", "16.23.45,452", "Jep Kiri",      "Benar"),
    (2, "2026-05-01", "16.23.56,789", "16.24.02,048", "Jep Kanan",     "Benar"),
    (2, "2026-05-01", "16.24.31,625", "16.24.36,441", "Jep Kiri",      "Salah"),
    (2, "2026-05-01", "16.24.47,194", "16.24.51,711", "Jep Kanan",     "Salah"),
    (2, "2026-05-01", "16.28.52,247", "16.28.59,003", "Tangkisan Kiri","Benar"),
    (2, "2026-05-01", "16.29.08,117", "16.29.16,609", "Tangkisan Kanan","Benar"),
    (2, "2026-05-01", "16.29.26,549", "16.29.32,581", "Tangkisan Kiri","Salah"),
    (2, "2026-05-01", "16.29.42,077", "16.29.50,066", "Tangkisan Kanan","Salah"),
    (2, "2026-05-01", "16.26.10,141", "16.26.17,245", "Kombinasi Kiri","Benar"),
    (2, "2026-05-01", "16.26.29,392", "16.26.36,560", "Kombinasi Kanan","Benar"),
    (2, "2026-05-01", "16.26.43,831", "16.26.49,992", "Kombinasi Kiri","Salah"),
    (2, "2026-05-01", "16.26.59,629", "16.27.05,926", "Kombinasi Kanan","Salah"),

    (3, "2026-05-01", "17.17.01,800", "17.17.07,302", "Jep Kiri",      "Benar"),
    (3, "2026-05-01", "17.17.17,006", "17.17.23,470", "Jep Kanan",     "Benar"),
    (3, "2026-05-01", "17.17.31,784", "17.17.37,817", "Jep Kiri",      "Salah"),
    (3, "2026-05-01", "17.17.46,703", "17.17.53,361", "Jep Kanan",     "Salah"),
    (3, "2026-05-01", "17.18.01,379", "17.18.06,774", "Tangkisan Kiri","Benar"),
    (3, "2026-05-01", "17.18.14,200", "17.18.20,906", "Tangkisan Kanan","Benar"),
    (3, "2026-05-01", "17.18.29,508", "17.18.35,206", "Tangkisan Kiri","Salah"),
    (3, "2026-05-01", "17.18.46,609", "17.18.54,494", "Tangkisan Kanan","Salah"),
    (3, "2026-05-01", "17.20.54,570", "17.21.02,393", "Kombinasi Kiri","Benar"),
    (3, "2026-05-01", "17.21.08,890", "17.21.19,599", "Kombinasi Kanan","Benar"),
    (3, "2026-05-01", "17.21.24,715", "17.21.33,603", "Kombinasi Kiri","Salah"),
    (3, "2026-05-01", "17.21.40,304", "17.21.53,802", "Kombinasi Kanan","Salah"),

    (4, "2026-05-01", "17.40.44,126", "17.40.52,303", "Jep Kiri",      "Benar"),
    (4, "2026-05-01", "17.41.02,836", "17.41.06,965", "Jep Kanan",     "Benar"),
    (4, "2026-05-01", "17.41.17,321", "17.41.22,429", "Jep Kiri",      "Salah"),
    (4, "2026-05-01", "17.41.33,279", "17.41.40,266", "Jep Kanan",     "Salah"),
    (4, "2026-05-01", "17.41.47,422", "17.41.51,507", "Tangkisan Kiri","Benar"),
    (4, "2026-05-01", "17.41.58,492", "17.42.05,126", "Tangkisan Kanan","Benar"),
    (4, "2026-05-01", "17.42.10,565", "17.42.17,099", "Tangkisan Kiri","Salah"),
    (4, "2026-05-01", "17.42.22,958", "17.42.37,121", "Tangkisan Kanan","Salah"),
    (4, "2026-05-01", "17.42.45,240", "17.42.51,754", "Kombinasi Kiri","Benar"),
    (4, "2026-05-01", "17.42.59,279", "17.43.04,863", "Kombinasi Kanan","Benar"),
    (4, "2026-05-01", "17.43.10,735", "17.43.15,272", "Kombinasi Kiri","Salah"),
    (4, "2026-05-01", "17.43.23,259", "17.43.28,482", "Kombinasi Kanan","Salah"),

    (5, "2026-05-01", "17.55.23,658", "17.55.28,074", "Jep Kiri",      "Benar"),
    (5, "2026-05-01", "17.55.34,240", "17.55.38,454", "Jep Kanan",     "Benar"),
    (5, "2026-05-01", "17.55.44,862", "17.55.48,484", "Jep Kiri",      "Salah"),
    (5, "2026-05-01", "17.55.57,664", "17.56.01,861", "Jep Kanan",     "Salah"),
    (5, "2026-05-01", "17.56.08,730", "17.56.12,854", "Tangkisan Kiri","Benar"),
    (5, "2026-05-01", "17.56.20,040", "17.56.24,082", "Tangkisan Kanan","Benar"),
    (5, "2026-05-01", "17.56.31,174", "17.56.34,557", "Tangkisan Kiri","Salah"),
    (5, "2026-05-01", "17.56.42,756", "17.56.46,098", "Tangkisan Kanan","Salah"),
    (5, "2026-05-01", "17.56.53,266", "17.56.59,946", "Kombinasi Kiri","Benar"),
    (5, "2026-05-01", "17.57.08,442", "17.57.12,659", "Kombinasi Kanan","Benar"),
    (5, "2026-05-01", "17.57.18,764", "17.57.24,841", "Kombinasi Kiri","Salah"),
    (5, "2026-05-01", "17.57.30,969", "17.57.36,757", "Kombinasi Kanan","Salah"),

    (6, "2026-05-01", "18.05.15,534", "18.05.22,778", "Jep Kiri",      "Benar"),
    (6, "2026-05-01", "18.05.38,271", "18.05.45,147", "Jep Kanan",     "Benar"),
    (6, "2026-05-01", "18.05.53,545", "18.06.01,449", "Jep Kiri",      "Salah"),
    (6, "2026-05-01", "18.06.08,004", "18.06.14,063", "Jep Kanan",     "Salah"),
    (6, "2026-05-01", "18.06.24,099", "18.06.33,175", "Tangkisan Kiri","Benar"),
    (6, "2026-05-01", "18.06.44,029", "18.06.49,154", "Tangkisan Kanan","Benar"),
    (6, "2026-05-01", "18.06.57,116", "18.07.02,635", "Tangkisan Kiri","Salah"),
    (6, "2026-05-01", "18.07.11,948", "18.07.17,633", "Tangkisan Kanan","Salah"),
    (6, "2026-05-01", "18.07.32,450", "18.07.41,335", "Kombinasi Kiri","Benar"),
    (6, "2026-05-01", "18.07.48,953", "18.07.56,540", "Kombinasi Kanan","Benar"),
    (6, "2026-05-01", "18.08.03,204", "18.08.11,994", "Kombinasi Kiri","Salah"),
    (6, "2026-05-01", "18.08.26,568", "18.08.32,101", "Kombinasi Kanan","Salah"),

    (7, "2026-05-02", "21.49.26,693", "21.49.31,790", "Jep Kiri",      "Benar"),
    (7, "2026-05-02", "21.49.38,295", "21.49.42,976", "Jep Kanan",     "Benar"),
    (7, "2026-05-02", "21.49.49,660", "21.49.56,696", "Jep Kiri",      "Salah"),
    (7, "2026-05-02", "21.50.04,080", "21.50.10,916", "Jep Kanan",     "Salah"),
    (7, "2026-05-02", "21.50.18,605", "21.50.30,416", "Tangkisan Kiri","Benar"),
    (7, "2026-05-02", "21.50.39,496", "21.50.46,113", "Tangkisan Kanan","Benar"),
    (7, "2026-05-02", "21.50.53,422", "21.51.00,489", "Tangkisan Kiri","Salah"),
    (7, "2026-05-02", "21.51.07,530", "21.51.13,289", "Tangkisan Kanan","Salah"),
    (7, "2026-05-02", "21.51.20,412", "21.51.28,237", "Kombinasi Kiri","Benar"),
    (7, "2026-05-02", "21.51.36,222", "21.51.43,909", "Kombinasi Kanan","Benar"),
    (7, "2026-05-02", "21.51.50,661", "21.51.57,010", "Kombinasi Kiri","Salah"),
    (7, "2026-05-02", "21.52.06,006", "21.52.13,288", "Kombinasi Kanan","Salah"),
    (8, "2026-05-02", "21.31.30,657", "21.31.40,672", "Jep Kiri",      "Benar"),
    (8, "2026-05-02", "21.32.02,138", "21.32.12,127", "Jep Kanan",     "Benar"),
    (8, "2026-05-02", "21.32.23,081", "21.32.33,231", "Jep Kiri",      "Salah"),
    (8, "2026-05-02", "21.32.39,499", "21.32.45,529", "Jep Kanan",     "Salah"),
    (8, "2026-05-02", "21.32.57,288", "21.33.04,000", "Tangkisan Kiri","Benar"),
    (8, "2026-05-02", "21.33.22,583", "21.33.28,201", "Tangkisan Kanan","Benar"),
    (8, "2026-05-02", "21.33.38,662", "21.33.48,371", "Tangkisan Kiri","Salah"),
    (8, "2026-05-02", "21.34.02,712", "21.34.15,372", "Tangkisan Kanan","Salah"),
    (8, "2026-05-02", "21.34.29,429", "21.34.38,132", "Kombinasi Kiri","Benar"),
    (8, "2026-05-02", "21.34.46,735", "21.34.57,594", "Kombinasi Kanan","Benar"),
    (8, "2026-05-02", "21.35.09,191", "21.35.16,119", "Kombinasi Kiri","Salah"),
    (8, "2026-05-02", "21.35.22,388", "21.35.31,207", "Kombinasi Kanan","Salah"),
    (9, "2026-05-02", "22.02.53,580", "22.02.58,278", "Jep Kiri",      "Benar"),
    (9, "2026-05-02", "22.03.04,114", "22.03.08,005", "Jep Kanan",     "Benar"),
    (9, "2026-05-02", "22.03.14,465", "22.03.18,040", "Jep Kiri",      "Salah"),
    (9, "2026-05-02", "22.03.23,270", "22.03.30,951", "Jep Kanan",     "Salah"),
    (9, "2026-05-02", "22.03.39,948", "22.03.46,507", "Tangkisan Kiri","Benar"),
    (9, "2026-05-02", "22.03.50,295", "22.03.58,488", "Tangkisan Kanan","Benar"),
    (9, "2026-05-02", "22.04.09,403", "22.04.14,214", "Tangkisan Kiri","Salah"),
    (9, "2026-05-02", "22.04.23,530", "22.04.38,718", "Tangkisan Kanan","Salah"),
    (9, "2026-05-02", "22.04.45,293", "22.04.50,707", "Kombinasi Kiri","Benar"),
    (9, "2026-05-02", "22.04.55,854", "22.05.00,744", "Kombinasi Kanan","Benar"),
    (9, "2026-05-02", "22.05.09,483", "22.05.15,591", "Kombinasi Kiri","Salah"),
    (9, "2026-05-02", "22.05.25,320", "22.05.30,747", "Kombinasi Kanan","Salah"),
    (10, "2026-05-02", "22.15.50,249", "22.15.56,071", "Jep Kiri",      "Benar"),
    (10, "2026-05-02", "22.16.00,592", "22.16.09,259", "Jep Kanan",     "Benar"),
    (10, "2026-05-02", "22.16.14,617", "22.16.21,068", "Jep Kiri",      "Salah"),
    (10, "2026-05-02", "22.16.40,145", "22.16.43,810", "Jep Kanan",     "Salah"),
    (10, "2026-05-02", "22.16.53,239", "22.16.57,118", "Tangkisan Kiri","Benar"),
    (10, "2026-05-02", "22.17.05,309", "22.17.11,243", "Tangkisan Kanan","Benar"),
    (10, "2026-05-02", "22.18.05,617", "22.18.13,809", "Tangkisan Kiri","Salah"),
    (10, "2026-05-02", "22.18.19,135", "22.18.27,415", "Tangkisan Kanan","Salah"),
    (10, "2026-05-02", "22.18.34,493", "22.18.44,733", "Kombinasi Kiri","Benar"),
    (10, "2026-05-02", "22.18.51,212", "22.19.00,115", "Kombinasi Kanan","Benar"),
    (10, "2026-05-02", "22.19.12,380", "22.19.18,525", "Kombinasi Kiri","Salah"),
    (10, "2026-05-02", "22.19.24,399", "22.19.34,293", "Kombinasi Kanan","Salah"),
]

# ══════════════════════════════════════════════════════════════════
#  BEST PARAMETERS (from your exhaustive search)
# ══════════════════════════════════════════════════════════════════
BEST = {
    "movement": {
        "select__k": 100, "svm__C": 0.01, "svm__kernel": "linear", "svm__gamma": 0.1,
        "svm__degree": 2, "svm__coef0": 0.5, "svm__class_weight": None,
        "svm__decision_function_shape": "ovo", "svm__shrinking": False, "svm__tol": 0.001
    },
    "jep_kiri": {
        "select__k": 250, "svm__C": 100, "svm__kernel": "sigmoid", "svm__gamma": "auto",
        "svm__degree": 5, "svm__coef0": 0.5, "svm__class_weight": "balanced",
        "svm__shrinking": True, "svm__tol": 0.001
    },
    "jep_kanan": {
        "select__k": 100, "svm__C": 10, "svm__kernel": "sigmoid", "svm__gamma": 1,
        "svm__degree": 4, "svm__coef0": 2.0, "svm__class_weight": None,
        "svm__shrinking": False, "svm__tol": 0.0001
    },
    "tangkisan_kiri": {
        "select__k": 50, "svm__C": 10, "svm__kernel": "sigmoid", "svm__gamma": 1,
        "svm__degree": 4, "svm__coef0": 0.5, "svm__class_weight": None,
        "svm__shrinking": False, "svm__tol": 0.01
    },
    "tangkisan_kanan": {
        "select__k": 300, "svm__C": 1, "svm__kernel": "rbf", "svm__gamma": 1,
        "svm__degree": 2, "svm__coef0": 0.5, "svm__class_weight": None,
        "svm__shrinking": False, "svm__tol": 0.001
    },
    "kombinasi_kiri": {
        "select__k": 300, "svm__C": 100, "svm__kernel": "sigmoid", "svm__gamma": 0.0001,
        "svm__degree": 5, "svm__coef0": 1.0, "svm__class_weight": "balanced",
        "svm__shrinking": False, "svm__tol": 0.001
    },
    "kombinasi_kanan": {
        "select__k": 250, "svm__C": 100, "svm__kernel": "sigmoid", "svm__gamma": "scale",
        "svm__degree": 5, "svm__coef0": 0.5, "svm__class_weight": None,
        "svm__shrinking": True, "svm__tol": 0.001
    },
}

# ══════════════════════════════════════════════════════════════════
#  SENSOR LAYOUT (topology-aware features)
# ══════════════════════════════════════════════════════════════════
N_SENSORS = 4
ACCEL_COLS = [[f"a{a}{i}" for a in "xyz"] for i in range(N_SENSORS)]
GYRO_COLS  = [[f"g{a}{i}" for a in "xyz"] for i in range(N_SENSORS)]


# ══════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════

def parse_label_time(date_str, time_str):
    parts = time_str.split(",")
    hms   = parts[0].replace(".", ":")
    ms    = parts[1] if len(parts) > 1 else "0"
    return datetime.strptime(f"{date_str} {hms}.{ms}", "%Y-%m-%d %H:%M:%S.%f")


def load_subject_csv(subject_id):
    path = os.path.join(
        CONFIG["data_dir"],
        CONFIG["file_pattern"].replace("{id}", str(subject_id))
    )
    return pd.read_csv(path, parse_dates=["timestamp"])


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
    fft_energy = np.sum(np.abs(np.fft.rfft(x)) ** 2) / len(x)
    zcr = np.sum(np.abs(np.diff(np.sign(x - mean)))) / (2 * max(len(x) - 1, 1))
    return [mean, std, xmin, xmax, rms, rng, med, iqr, skew, kurt, fft_energy, zcr]


def _safe_corr(a, b):
    if len(a) < 3: return 0.0
    if np.std(a) < 1e-9 or np.std(b) < 1e-9: return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _accel_mag(data, cols, si):
    idx = [cols.index(c) for c in ACCEL_COLS[si] if c in cols]
    return np.sqrt(np.sum(data[:, idx] ** 2, axis=1)) if idx else np.zeros(len(data))


def _gyro_mag(data, cols, si):
    idx = [cols.index(c) for c in GYRO_COLS[si] if c in cols]
    return np.sqrt(np.sum(data[:, idx] ** 2, axis=1)) if idx else np.zeros(len(data))


def extract_features(segment):
    """426 topology-aware features — identical to server-side extraction."""
    data = segment.drop(columns=["timestamp"]).values.astype(float)
    cols = [c for c in segment.columns if c != "timestamp"]
    n    = len(data)
    EPS  = 1e-9
    f    = []

    # Part A — 24 ch × 12 stats = 288
    for i in range(data.shape[1]):
        f.extend(_channel_stats(data[:, i]))

    # Part B — 4 sensors × 2 mags × 12 = 96
    s0a = _accel_mag(data, cols, 0); s1a = _accel_mag(data, cols, 1)
    s2a = _accel_mag(data, cols, 2); s3a = _accel_mag(data, cols, 3)
    s0g = _gyro_mag(data, cols, 0);  s1g = _gyro_mag(data, cols, 1)
    s2g = _gyro_mag(data, cols, 2);  s3g = _gyro_mag(data, cols, 3)
    for am, gm in [(s0a, s0g), (s1a, s1g), (s2a, s2g), (s3a, s3g)]:
        f.extend(_channel_stats(am))
        f.extend(_channel_stats(gm))

    # Part C — topology-aware = 42
    ra = (s0a + s1a) / 2; la = (s2a + s3a) / 2
    rg = (s0g + s1g) / 2; lg = (s2g + s3g) / 2
    f.extend(_channel_stats(ra - la))          # C1 accel asymmetry (12)
    f.extend(_channel_stats(rg - lg))          # C1 gyro  asymmetry (12)
    f.append(np.mean(s0a) / (np.mean(s0a) + np.mean(s1a) + EPS))   # C2 (4)
    f.append(np.mean(s3a) / (np.mean(s3a) + np.mean(s2a) + EPS))
    f.append(np.mean(s0g) / (np.mean(s0g) + np.mean(s1g) + EPS))
    f.append(np.mean(s3g) / (np.mean(s3g) + np.mean(s2g) + EPS))
    f.append(_safe_corr(s0a, s1a)); f.append(_safe_corr(s3a, s2a))  # C3 (4)
    f.append(_safe_corr(s0g, s1g)); f.append(_safe_corr(s3g, s2g))
    f.append(float(np.max(ra)) / (float(np.max(la)) + EPS))          # C4 (2)
    f.append(float(np.max(rg)) / (float(np.max(lg)) + EPS))
    f.append(float(np.argmax(s0a)) / max(n - 1, 1))                  # C5 (4)
    f.append(float(np.argmax(s1a)) / max(n - 1, 1))
    f.append(float(np.argmax(s3a)) / max(n - 1, 1))
    f.append(float(np.argmax(s2a)) / max(n - 1, 1))
    f.append(float(np.mean(ra ** 2))); f.append(float(np.mean(la ** 2)))  # C6 (4)
    f.append(float(np.mean(rg ** 2))); f.append(float(np.mean(lg ** 2)))

    return np.array(f, dtype=np.float64)


# ══════════════════════════════════════════════════════════════════
#  DATASET BUILDER
# ══════════════════════════════════════════════════════════════════

def build_dataset():
    label_df = pd.DataFrame(RAW_LABELS, columns=[
        "subject", "date", "t_start", "t_end", "movement", "correctness"
    ])
    cache = {}
    X_mv, y_mv, subj = [], [], []
    per_class = {mv: {"X": [], "y": [], "subjects": []} for mv in MOVEMENT_CLASSES}
    skipped = 0

    for _, row in label_df.iterrows():
        sid = int(row["subject"])
        if sid not in cache:
            try:
                cache[sid] = load_subject_csv(sid)
            except FileNotFoundError as e:
                print(f"  [WARN] {e}")
                skipped += 1
                continue

        df      = cache[sid]
        t_start = parse_label_time(row["date"], row["t_start"])
        t_end   = parse_label_time(row["date"], row["t_end"])
        seg     = df[(df["timestamp"] >= t_start) & (df["timestamp"] <= t_end)]

        if len(seg) < 3:
            skipped += 1
            continue

        feat = extract_features(seg)
        mv   = row["movement"]
        corr = 1 if row["correctness"] == "Benar" else 0

        X_mv.append(feat); y_mv.append(mv); subj.append(sid)
        per_class[mv]["X"].append(feat)
        per_class[mv]["y"].append(corr)
        per_class[mv]["subjects"].append(sid)

    X_mv  = np.array(X_mv)
    y_mv  = np.array(y_mv)
    subj  = np.array(subj)
    for mv in MOVEMENT_CLASSES:
        per_class[mv]["X"]        = np.array(per_class[mv]["X"])
        per_class[mv]["y"]        = np.array(per_class[mv]["y"])
        per_class[mv]["subjects"] = np.array(per_class[mv]["subjects"])

    print(f"  Loaded: {len(X_mv)} segments | Skipped: {skipped}")
    return X_mv, y_mv, subj, per_class


# ══════════════════════════════════════════════════════════════════
#  PIPELINE BUILDER  (matches BEST params structure)
# ══════════════════════════════════════════════════════════════════

def build_pipeline(params, n_classes):
    p   = params.copy()
    k   = p.pop("select__k")
    kw  = {key[5:]: val for key, val in p.items()}
    # decision_function_shape not valid for binary classifiers
    if n_classes == 2 and "decision_function_shape" in kw:
        del kw["decision_function_shape"]
    svc = SVC(probability=True, random_state=CONFIG["random_state"], **kw)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=k)),
        ("svm",    svc),
    ])


# ══════════════════════════════════════════════════════════════════
#  LOSO EVALUATION
# ══════════════════════════════════════════════════════════════════

def loso_eval(X, y, subjects, params, label, target_names, n_classes):
    unique_subj = np.unique(subjects)
    all_true, all_pred = [], []

    for test_sid in unique_subj:
        tr = subjects != test_sid
        te = subjects == test_sid
        if len(np.unique(y[tr])) < 2:
            continue
        pipe = build_pipeline(params.copy(), n_classes)
        pipe.fit(X[tr], y[tr])
        all_true.extend(y[te])
        all_pred.extend(pipe.predict(X[te]))

    acc = accuracy_score(all_true, all_pred)
    print(f"\n{'=' * 60}")
    print(f"  LOSO: {label}   Accuracy: {acc:.4f} ({acc * 100:.1f}%)")
    print(classification_report(all_true, all_pred,
                                 target_names=target_names, zero_division=0))
    return acc


# ══════════════════════════════════════════════════════════════════
#  SAVE MODELS
# ══════════════════════════════════════════════════════════════════

def save_models(X_mv, y_mv, per_class):
    """
    Train each model on the FULL dataset (no held-out fold)
    and save as .pkl to CONFIG['model_save_dir'].
    """
    os.makedirs(CONFIG["model_save_dir"], exist_ok=True)

    # Model 0 — Movement
    print("\n  Fitting final movement model on full data…")
    pipe_mv = build_pipeline(BEST["movement"].copy(), n_classes=6)
    pipe_mv.fit(X_mv, y_mv)
    path = os.path.join(CONFIG["model_save_dir"], "model_movement.pkl")
    joblib.dump(pipe_mv, path)
    print(f"  Saved → {path}  ({os.path.getsize(path)/1024:.1f} KB)")

    # Models 1-6 — Correctness per movement
    for mv in MOVEMENT_CLASSES:
        key = mv.lower().replace(" ", "_")
        if key not in BEST:
            continue
        d = per_class[mv]
        if len(d["X"]) < 4 or len(np.unique(d["y"])) < 2:
            print(f"  [SKIP] {mv} — insufficient data")
            continue

        print(f"  Fitting final model for {mv}…")
        pipe_c = build_pipeline(BEST[key].copy(), n_classes=2)
        pipe_c.fit(d["X"], d["y"])
        path = os.path.join(CONFIG["model_save_dir"], f"model_{key}.pkl")
        joblib.dump(pipe_c, path)
        print(f"  Saved → {path}  ({os.path.getsize(path)/1024:.1f} KB)")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Silat SVM — Training + Save")
    print("=" * 60)

    # 1. Build dataset
    print("\n[1/3] Building dataset…")
    X_mv, y_mv, subj, per_class = build_dataset()

    if len(X_mv) == 0:
        print("[ERROR] No data loaded. Check data_dir in CONFIG.")
        return

    # 2. LOSO evaluation
    print("\n[2/3] LOSO evaluation (best fixed parameters)…")
    loso_eval(X_mv, y_mv, subj,
              BEST["movement"], "Movement", MOVEMENT_CLASSES, 6)

    for mv in MOVEMENT_CLASSES:
        key = mv.lower().replace(" ", "_")
        if key not in BEST:
            continue
        d = per_class[mv]
        loso_eval(d["X"], d["y"], d["subjects"],
                  BEST[key], f"Correctness – {mv}", ["Salah", "Benar"], 2)

    # 3. Save full-data models
    print("\n[3/3] Saving final models…")
    save_models(X_mv, y_mv, per_class)

    print("\n✓ All done. Models saved to:", CONFIG["model_save_dir"])
    print("  Run silat_server.py to start the inference server.")


if __name__ == "__main__":
    main()