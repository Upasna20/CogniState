import time
import threading
import pandas as pd
import numpy as np
import joblib
from evdev import InputDevice, categorize, ecodes, list_devices
from flask import Flask, jsonify
from flask_cors import CORS
from collections import deque

# ==========================================
# 1. INITIALIZATION & AI MODEL
# ==========================================
MODEL_PATH = 'stress_svm_model.pkl'
print("Loading AI Model...")
svm_model = joblib.load(MODEL_PATH)

event_buffer = []
buffer_lock = threading.Lock()
historical_features = []
CALIBRATION_WINDOWS_NEEDED = 4

# ==========================================
# SHARED STATE (read by Flask API)
# ==========================================
daemon_state = {
    "running": False,
    "status": "idle",           # idle | calibrating | normal | stress
    "status_text": "Daemon not running",
    "cal_count": 0,
    "window_count": 0,
    "normal_count": 0,
    "stress_count": 0,
    "last_result": None,        # "normal" | "stress" | "calibrating"
    "last_result_time": None,
    "last_features": {},
    "history": deque(maxlen=20),  # list of {window, result, time, features}
    "log": deque(maxlen=100),     # list of {time, msg, cls}
}
state_lock = threading.Lock()


def add_log(msg, cls="info"):
    with state_lock:
        daemon_state["log"].appendleft({
            "time": time.strftime("%H:%M:%S"),
            "msg": msg,
            "cls": cls
        })
    print(msg)


def record_event(event_type, xpos=0, ypos=0, key=0):
    with buffer_lock:
        event_buffer.append({
            'timestamp': time.time(),
            'event': event_type,
            'xpos': xpos,
            'ypos': ypos,
            'key': str(key)
        })


# ==========================================
# 2. HARDWARE LISTENERS (evdev - Wayland Safe)
# ==========================================
def find_devices():
    keyboards, mice = [], []
    for path in list_devices():
        dev = InputDevice(path)
        caps = dev.capabilities()
        if ecodes.EV_KEY in caps:
            keys = caps[ecodes.EV_KEY]
            if ecodes.KEY_A in keys:
                keyboards.append(dev)
            elif ecodes.BTN_LEFT in keys:
                mice.append(dev)
    return keyboards, mice


def listen_keyboard(dev):
    try:
        for event in dev.read_loop():
            if event.type == ecodes.EV_KEY:
                key_event = categorize(event)
                if key_event.keystate == key_event.key_down:
                    record_event('keydown', key=key_event.keycode)
                elif key_event.keystate == key_event.key_up:
                    record_event('keyup', key=key_event.keycode)
    except OSError:
        pass


def listen_mouse(dev):
    x, y = 0, 0
    try:
        for event in dev.read_loop():
            if event.type == ecodes.EV_REL:
                if event.code == ecodes.REL_X:
                    x += event.value
                elif event.code == ecodes.REL_Y:
                    y += event.value
                record_event('mousemove', xpos=x, ypos=y)
    except OSError:
        pass


# ==========================================
# 3. FEATURE EXTRACTION
# ==========================================
def extract_features(df):
    if df.empty:
        return pd.DataFrame([{col: 0.0 for col in [
            'mouse_vel_mean', 'mouse_vel_std', 'mouse_total_dist',
            'key_press_count', 'backspace_count', 'key_dwell_mean',
            'key_dwell_std', 'key_flight_mean', 'key_flight_std', 'error_rate'
        ]}])

    df = df.sort_values(by='timestamp').reset_index(drop=True)

    mouse_df = df[df['event'] == 'mousemove'].copy()
    if not mouse_df.empty:
        mouse_df['dx'] = mouse_df['xpos'].diff()
        mouse_df['dy'] = mouse_df['ypos'].diff()
        mouse_df['dt'] = mouse_df['timestamp'].diff().dt.total_seconds()
        valid_move = (mouse_df['dt'] > 0) & (mouse_df['dt'] < 2.0)
        mouse_df.loc[valid_move, 'dist'] = np.sqrt(
            mouse_df.loc[valid_move, 'dx']**2 + mouse_df.loc[valid_move, 'dy']**2)
        mouse_df.loc[valid_move, 'velocity'] = (
            mouse_df.loc[valid_move, 'dist'] / mouse_df.loc[valid_move, 'dt'])
        mouse_vel_mean = mouse_df['velocity'].mean()
        mouse_vel_std = mouse_df['velocity'].std()
        mouse_total_dist = mouse_df['dist'].sum()
    else:
        mouse_vel_mean = mouse_vel_std = mouse_total_dist = 0.0

    key_df = df[df['event'].isin(['keydown', 'keyup'])].copy()
    key_state, dwell_times = {}, []

    for _, row in key_df.iterrows():
        key = row['key']
        event = row['event']
        if key not in key_state:
            key_state[key] = None
        if event == 'keydown':
            key_state[key] = row['timestamp']
            dwell_times.append(np.nan)
        elif event == 'keyup':
            if key_state[key] is not None:
                dwell_times.append((row['timestamp'] - key_state[key]).total_seconds())
                key_state[key] = None
            else:
                dwell_times.append(np.nan)

    key_df['dwell_time'] = dwell_times
    key_down_df = key_df[key_df['event'] == 'keydown'].copy()

    if not key_down_df.empty:
        key_press_count = len(key_down_df)
        backspace_count = key_down_df['key'].apply(
            lambda x: 1 if 'BACKSPACE' in str(x).upper() else 0).sum()
        key_down_df['flight_time'] = key_down_df['timestamp'].diff().dt.total_seconds()
        valid_flight = (key_down_df['flight_time'] > 0) & (key_down_df['flight_time'] < 2.0)
        key_flight_mean = key_down_df.loc[valid_flight, 'flight_time'].mean()
        key_flight_std = key_down_df.loc[valid_flight, 'flight_time'].std()
    else:
        key_press_count = backspace_count = 0
        key_flight_mean = key_flight_std = 0.0

    error_rate = backspace_count / (key_press_count + 1)

    features_df = pd.DataFrame([{
        'mouse_vel_mean': mouse_vel_mean,
        'mouse_vel_std': mouse_vel_std,
        'mouse_total_dist': mouse_total_dist,
        'key_press_count': key_press_count,
        'backspace_count': backspace_count,
        'key_dwell_mean': key_df['dwell_time'].mean() if not key_df.empty else 0.0,
        'key_dwell_std': key_df['dwell_time'].std() if not key_df.empty else 0.0,
        'key_flight_mean': key_flight_mean,
        'key_flight_std': key_flight_std,
        'error_rate': error_rate
    }]).fillna(0.0)

    return features_df


# ==========================================
# 4. THE 30-SECOND PROCESSOR
# ==========================================
def analyze_window():
    global event_buffer, historical_features

    while True:
        time.sleep(30)

        with state_lock:
            if not daemon_state["running"]:
                continue

        with buffer_lock:
            if len(event_buffer) == 0:
                with state_lock:
                    daemon_state["status"] = "idle"
                    daemon_state["status_text"] = "No activity detected in current window"

                add_log("[Idle] No activity detected.", "info")
                continue
                
            current_data = list(event_buffer)
            event_buffer.clear()

        df = pd.DataFrame(current_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        features_df = extract_features(df)
        features_dict = features_df.iloc[0].to_dict()

        with state_lock:
            daemon_state["window_count"] += 1
            daemon_state["last_features"] = {
                k: round(float(v), 4) for k, v in features_dict.items()
            }
            win_num = daemon_state["window_count"]

        if len(historical_features) < CALIBRATION_WINDOWS_NEEDED:
            historical_features.append(features_df.iloc[0])
            cal = len(historical_features)
            with state_lock:
                daemon_state["cal_count"] = cal
                daemon_state["status"] = "calibrating"
                daemon_state["status_text"] = f"Calibrating... ({cal}/{CALIBRATION_WINDOWS_NEEDED})"
                daemon_state["last_result"] = "calibrating"
                daemon_state["last_result_time"] = time.strftime("%H:%M:%S")
                daemon_state["history"].append({
                    "window": win_num,
                    "result": "calibrating",
                    "time": time.strftime("%H:%M:%S"),
                    "features": daemon_state["last_features"].copy()
                })
            add_log(f"[Calibrating] Gathering baseline... ({cal}/{CALIBRATION_WINDOWS_NEEDED})", "calibrating")
            if cal == CALIBRATION_WINDOWS_NEEDED:
                add_log("[System] Calibration complete. Live predictions now active.", "info")
            continue

        history_df = pd.DataFrame(historical_features)
        live_z_scores = (features_df - history_df.mean()) / (history_df.std() + 1e-6)

        feature_cols = [
            'mouse_vel_mean', 'mouse_vel_std', 'mouse_total_dist',
            'key_press_count', 'backspace_count', 'key_dwell_mean',
            'key_dwell_std', 'key_flight_mean', 'key_flight_std', 'error_rate'
        ]
        live_z_scores = live_z_scores[feature_cols].fillna(0)
        prediction = svm_model.predict(live_z_scores)[0]

        now_str = time.strftime("%H:%M:%S")
        if prediction == 1:
            result = "normal"
            with state_lock:
                daemon_state["normal_count"] += 1
                daemon_state["status"] = "normal"
                daemon_state["status_text"] = "Normal — behavior matches your baseline"
                daemon_state["last_result"] = "normal"
                daemon_state["last_result_time"] = now_str
                daemon_state["history"].append({
                    "window": win_num, "result": "normal",
                    "time": now_str, "features": daemon_state["last_features"].copy()
                })
            add_log("[✅ NORMAL] - Behavior is matching your baseline.", "normal")
        else:
            result = "stress"
            with state_lock:
                daemon_state["stress_count"] += 1
                daemon_state["status"] = "stress"
                daemon_state["status_text"] = "Stress detected — erratic KMD behavior"
                daemon_state["last_result"] = "stress"
                daemon_state["last_result_time"] = now_str
                daemon_state["history"].append({
                    "window": win_num, "result": "stress",
                    "time": now_str, "features": daemon_state["last_features"].copy()
                })
            add_log("[🚨 STRESS DETECTED] - Erratic KMD behavior identified!", "stress")


# ==========================================
# 5. FLASK API
# ==========================================
app = Flask(__name__)
CORS(app)  # allow requests from the dashboard HTML


@app.route('/status')
def status():
    with state_lock:
        return jsonify({
            "running":          daemon_state["running"],
            "status":           daemon_state["status"],
            "status_text":      daemon_state["status_text"],
            "cal_count":        daemon_state["cal_count"],
            "window_count":     daemon_state["window_count"],
            "normal_count":     daemon_state["normal_count"],
            "stress_count":     daemon_state["stress_count"],
            "last_result":      daemon_state["last_result"],
            "last_result_time": daemon_state["last_result_time"],
            "last_features":    daemon_state["last_features"],
            "history":          list(daemon_state["history"]),
            "log":              list(daemon_state["log"]),
        })


@app.route('/start', methods=['POST'])
def start():
    global historical_features, event_buffer
    with state_lock:
        if daemon_state["running"]:
            return jsonify({"ok": False, "msg": "Already running"})
        daemon_state["running"] = True
        daemon_state["status"] = "calibrating"
        daemon_state["status_text"] = "Starting..."
        daemon_state["cal_count"] = 0
        daemon_state["window_count"] = 0
        daemon_state["normal_count"] = 0
        daemon_state["stress_count"] = 0
        daemon_state["last_result"] = None
        daemon_state["last_result_time"] = None
        daemon_state["last_features"] = {}
        daemon_state["history"].clear()
        daemon_state["log"].clear()
    with buffer_lock:
        event_buffer.clear()
    historical_features = []
    add_log("[System] Daemon started.", "info")
    return jsonify({"ok": True})


@app.route('/stop', methods=['POST'])
def stop():
    with state_lock:
        daemon_state["running"] = False
        daemon_state["status"] = "idle"
        daemon_state["status_text"] = "Daemon stopped"
    add_log("[System] Daemon stopped.", "info")
    return jsonify({"ok": True})


# ==========================================
# 6. ENTRY POINT
# ==========================================
if __name__ == "__main__":
    print("Starting KMD Background Daemon (Wayland-Native)...")

    keyboards, mice = find_devices()

    if not keyboards and not mice:
        print("ERROR: No input devices found! Run 'sudo usermod -a -G input $USER' and log out/in.")
        exit()

    for dev in keyboards:
        threading.Thread(target=listen_keyboard, args=(dev,), daemon=True).start()
        add_log(f"[System] Listening to Keyboard: {dev.name}", "info")

    for dev in mice:
        threading.Thread(target=listen_mouse, args=(dev,), daemon=True).start()
        add_log(f"[System] Listening to Mouse: {dev.name}", "info")

    # Mark as running on startup (or leave False and use /start from the UI)
    with state_lock:
        daemon_state["running"] = True
        daemon_state["status"] = "calibrating"
        daemon_state["status_text"] = "Calibrating..."

    analyzer_thread = threading.Thread(target=analyze_window, daemon=True)
    analyzer_thread.start()

    # Flask runs on port 5050, accessible from the dashboard
    print("API running at http://localhost:5050")
    app.run(host='0.0.0.0', port=5050, debug=False, use_reloader=False)
