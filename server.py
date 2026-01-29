import os
import csv
import base64
import io
import pickle
import shutil
import json
import socket
from datetime import datetime
from functools import wraps

from flask import (
    Flask, request, jsonify, render_template,
    redirect, session, send_file
)

import face_recognition
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

from zeroconf import Zeroconf, ServiceInfo
from flask_socketio import SocketIO

# ================= CONFIG =================
ENC_FILE = "known_faces_encodings.pkl"
KNOWN_DIR = "known_faces"
ATTEND_DIR = "attendance"
ENROLL_SAMPLES = 10
MIN_GAP_SECONDS = 20

ESP32_CAM_IP = None
USERS_FILE = "users.json"

# ================= FOLDERS =================
os.makedirs(KNOWN_DIR, exist_ok=True)
os.makedirs(ATTEND_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)

# ================= FLASK =================
app = Flask(__name__, template_folder="templates")
app.secret_key = "attendance_secret_key"

# ✅ SOCKETIO (WebSocket)
socketio = SocketIO(app, cors_allowed_origins="*")

# ================= GLOBAL STATE =================
MODE = {"type": "idle", "name": None}
ENROLL_COUNT = 0

LAST_RESULT = {
    "status": "IDLE",
    "name": "",
    "entry": "",
    "confidence": 0
}

# ================= HELPERS =================
def ws_push_all():
    payload = {
        "mode": MODE["type"],
        "name": MODE["name"],
        "enroll_count": ENROLL_COUNT,
        "last": LAST_RESULT,
        "stream_url": (f"http://{ESP32_CAM_IP}:81/stream" if ESP32_CAM_IP else "")
    }
    socketio.emit("state_update", payload)

# ================= USERS =================
def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

# ================= LOAD ENCODINGS =================
if os.path.exists(ENC_FILE):
    with open(ENC_FILE, "rb") as f:
        data = pickle.load(f)
        known_encodings = data.get("encodings", [])
        known_names = data.get("names", [])
else:
    known_encodings = []
    known_names = []

# ================= LOGIN REQUIRED =================
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if "user" not in session:
            return redirect("/login")
        return f(*args, **kwargs)
    return wrap

# ================= mDNS attendance.local =================
zeroconf = None

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

def register_mdns_service():
    global zeroconf
    hostname = "attendance"
    port = 5000
    ip = get_local_ip()

    info = ServiceInfo(
        type_="_http._tcp.local.",
        name=f"{hostname}._http._tcp.local.",
        addresses=[socket.inet_aton(ip)],
        port=port,
        properties={},
        server=f"{hostname}.local."
    )

    zeroconf = Zeroconf()
    zeroconf.register_service(info)

    print("============================================")
    print(f"✅ mDNS Registered : http://{hostname}.local:{port}")
    print(f"✅ Server IP       : http://{ip}:{port}")
    print("============================================")

# ================= FILTER =================
@app.template_filter("ddmmyyyy")
def ddmmyyyy(date_str):
    try:
        y, m, d = date_str.split("-")
        return f"{d}-{m}-{y}"
    except:
        return date_str

# ================= SOCKETIO EVENTS =================
@socketio.on("connect")
def ws_connect():
    ws_push_all()

# ================= ESP32-CAM REGISTER =================
@app.route("/esp32_register", methods=["POST"])
def esp32_register():
    global ESP32_CAM_IP
    data = request.get_json(force=True, silent=True) or {}
    ESP32_CAM_IP = data.get("ip")
    print("✅ ESP32-CAM Registered IP:", ESP32_CAM_IP)
    ws_push_all()
    return jsonify({"status": "ok", "ip": ESP32_CAM_IP})

# ================= AUTH =================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        users = load_users()
        if username in users and users[username]["password"] == password:
            session["user"] = username
            session["role"] = users[username]["role"]
            return redirect("/dashboard")

        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

# ================= DASHBOARD =================
@app.route("/dashboard")
@login_required
def dashboard():
    stream_url = f"http://{ESP32_CAM_IP}:81/stream" if ESP32_CAM_IP else ""
    return render_template("dashboard.html", stream_url=stream_url)

# ================= USERS =================
@app.route("/users", methods=["GET", "POST"])
@login_required
def manage_users():
    if session.get("role") != "admin":
        return "Access denied"

    users = load_users()

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if username and password:
            users[username] = {"password": password, "role": "user"}
            save_users(users)

    return render_template("users.html", users=users)

@app.route("/delete_user/<username>")
@login_required
def delete_user(username):
    if session.get("role") != "admin":
        return "Access denied"

    users = load_users()

    if username == session.get("user"):
        return "Cannot delete current admin"

    if username in users:
        del users[username]
        save_users(users)

    return redirect("/users")

# ================= MODE (from ESP32 Nextion) =================
@app.route("/mode", methods=["POST"])
def set_mode():
    global MODE, ENROLL_COUNT, LAST_RESULT

    data = request.get_json(force=True, silent=True) or {}
    print("✅ /mode received:", data)

    new_mode = data.get("mode") or data.get("type")

    if new_mode in ["idle", "enroll", "attend"]:
        MODE["type"] = new_mode

    MODE["name"] = data.get("name", None)
    ENROLL_COUNT = 0

    LAST_RESULT = {
        "status": MODE["type"].upper(),
        "name": MODE["name"] or "",
        "entry": "",
        "confidence": 0
    }

    ws_push_all()
    return jsonify({"status": "ok", "mode": MODE})

@app.route("/status")
def status():
    return jsonify({
        "mode": MODE["type"],
        "name": MODE["name"],
        "enroll_count": ENROLL_COUNT,
        "last": LAST_RESULT
    })

@app.route("/last_recognition")
def last_recognition():
    return jsonify(LAST_RESULT)

# ================= CAPTURE (from ESP32-CAM) =================
@app.route("/capture", methods=["POST"])
def capture():
    global ENROLL_COUNT, LAST_RESULT
    global known_encodings, known_names

    data = request.get_json(force=True, silent=True) or {}
    if "image" not in data:
        return jsonify({"status": "NO_IMAGE"})

    try:
        img_bytes = base64.b64decode(data["image"])
        img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except:
        return jsonify({"status": "BAD_IMAGE"})

    frame = np.array(img_pil)

    encodings = face_recognition.face_encodings(frame)
    if not encodings:
        LAST_RESULT = {"status": "NO_FACE", "name": "", "entry": "", "confidence": 0}
        ws_push_all()
        return jsonify(LAST_RESULT)

    face = encodings[0]

    # -------- ENROLL --------
    if MODE["type"] == "enroll":
        name = MODE["name"]
        if not name:
            return jsonify({"status": "NO_NAME"})

        person_dir = os.path.join(KNOWN_DIR, name)
        os.makedirs(person_dir, exist_ok=True)

        # ✅ Save JPG samples
        img_path = os.path.join(person_dir, f"img_{ENROLL_COUNT + 1}.jpg")
        img_pil.save(img_path, "JPEG")

        known_encodings.append(face)
        known_names.append(name)
        ENROLL_COUNT += 1

        if ENROLL_COUNT >= ENROLL_SAMPLES:
            with open(ENC_FILE, "wb") as f:
                pickle.dump({"encodings": known_encodings, "names": known_names}, f)

            MODE["type"] = "idle"
            MODE["name"] = None

            LAST_RESULT = {"status": "ENROLL_COMPLETE", "name": name, "entry": "", "confidence": 100}
            ws_push_all()
            return jsonify(LAST_RESULT)

        LAST_RESULT = {"status": "ENROLLING", "name": name, "entry": "", "confidence": 0}
        ws_push_all()
        return jsonify({"status": "ENROLLING", "count": ENROLL_COUNT})

    # -------- ATTEND --------
    matches = face_recognition.compare_faces(known_encodings, face, tolerance=0.5)
    if True not in matches:
        LAST_RESULT = {"status": "UNKNOWN", "name": "", "entry": "", "confidence": 0}
        ws_push_all()
        return jsonify(LAST_RESULT)

    idx = matches.index(True)
    name = known_names[idx]
    confidence = 85

    file_path = os.path.join(ATTEND_DIR, f"{name}.csv")
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")

    # ✅ first entry per day must be IN
    entry = "IN"

    if os.path.exists(file_path):
        with open(file_path) as f:
            rows = list(csv.reader(f))

        if len(rows) > 1:
            today_rows = [r for r in rows[1:] if r[0] == today]

            if len(today_rows) > 0:
                last = today_rows[-1]
                last_status = last[3]

                last_time = datetime.strptime(f"{last[0]} {last[1]}", "%Y-%m-%d %H:%M:%S")
                diff = (now - last_time).total_seconds()

                if diff < MIN_GAP_SECONDS:
                    LAST_RESULT = {"status": "WAIT", "name": name, "entry": "", "confidence": 0}
                    ws_push_all()
                    return jsonify({"status": "WAIT", "seconds": int(MIN_GAP_SECONDS - diff)})

                entry = "OUT" if last_status == "IN" else "IN"

    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        if os.stat(file_path).st_size == 0:
            writer.writerow(["Date", "Time", "Name", "Status"])
        writer.writerow([now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), name, entry])

    LAST_RESULT = {"status": "ATTENDANCE_MARKED", "name": name, "entry": entry, "confidence": confidence}
    ws_push_all()
    return jsonify(LAST_RESULT)

# ================= FACES =================
@app.route("/faces")
@login_required
def faces():
    persons = [
        d for d in os.listdir(KNOWN_DIR)
        if os.path.isdir(os.path.join(KNOWN_DIR, d))
    ]
    return render_template("faces.html", users=persons)

@app.route("/delete/<name>")
@login_required
def delete_face(name):
    global known_encodings, known_names

    idxs = [i for i, n in enumerate(known_names) if n == name]
    for i in sorted(idxs, reverse=True):
        known_encodings.pop(i)
        known_names.pop(i)

    with open(ENC_FILE, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)

    shutil.rmtree(os.path.join(KNOWN_DIR, name), ignore_errors=True)

    att_file = os.path.join(ATTEND_DIR, f"{name}.csv")
    if os.path.exists(att_file):
        os.remove(att_file)

    ws_push_all()
    return redirect("/faces")

# ================= ADD FACE (single) =================
@app.route("/add_face", methods=["GET", "POST"])
@login_required
def add_face():
    global known_encodings, known_names

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        file = request.files.get("image")

        if not name or not file:
            return "Name or image missing"

        img_pil = Image.open(file.stream).convert("RGB")
        img_pil.thumbnail((640, 480))

        enc = face_recognition.face_encodings(np.array(img_pil))
        if not enc:
            return "No face detected. Use a clear photo."

        person_dir = os.path.join(KNOWN_DIR, name)
        os.makedirs(person_dir, exist_ok=True)
        img_pil.save(os.path.join(person_dir, "profile.jpg"), "JPEG")

        known_encodings.append(enc[0])
        known_names.append(name)

        with open(ENC_FILE, "wb") as f:
            pickle.dump({"encodings": known_encodings, "names": known_names}, f)

        ws_push_all()
        return redirect("/faces")

    return render_template("add_face.html")

# ================= ADD FACE UPLOAD (auto 10) =================
@app.route("/add_face_upload", methods=["GET", "POST"])
@login_required
def add_face_upload():
    global known_encodings, known_names

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        file = request.files.get("image")

        if not name or not file:
            return "Name or image missing"

        person_dir = os.path.join(KNOWN_DIR, name)
        os.makedirs(person_dir, exist_ok=True)

        img_pil = Image.open(file.stream).convert("RGB")
        img_pil.thumbnail((640, 480))

        for i in range(1, 11):
            img_path = os.path.join(person_dir, f"img_{i}.jpg")
            img_pil.save(img_path, "JPEG")
            enc = face_recognition.face_encodings(np.array(img_pil))
            if enc:
                known_encodings.append(enc[0])
                known_names.append(name)

        with open(ENC_FILE, "wb") as f:
            pickle.dump({"encodings": known_encodings, "names": known_names}, f)

        ws_push_all()
        return redirect("/faces")

    return render_template("add_face_upload.html")

# ================= ADD FACE CAPTURE (10 files upload) =================
@app.route("/add_face_capture", methods=["GET", "POST"])
@login_required
def add_face_capture():
    global known_encodings, known_names

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        files = request.files.getlist("images")

        if not name or len(files) != 10:
            return "10 images required"

        person_dir = os.path.join(KNOWN_DIR, name)
        os.makedirs(person_dir, exist_ok=True)

        for i, file in enumerate(files, start=1):
            img_pil = Image.open(file.stream).convert("RGB")
            img_pil.thumbnail((640, 480))

            img_path = os.path.join(person_dir, f"img_{i}.jpg")
            img_pil.save(img_path, "JPEG")

            enc = face_recognition.face_encodings(np.array(img_pil))
            if enc:
                known_encodings.append(enc[0])
                known_names.append(name)

        with open(ENC_FILE, "wb") as f:
            pickle.dump({"encodings": known_encodings, "names": known_names}, f)

        ws_push_all()
        return redirect("/faces")

    return render_template("add_face_capture.html")

# ================= ATTENDANCE =================
@app.route("/attendance")
@login_required
def attendance():
    persons = []
    for file in os.listdir(ATTEND_DIR):
        if file.endswith(".csv"):
            persons.append(file.replace(".csv", ""))
    return render_template("attendance_list.html", persons=persons)

@app.route("/attendance/<name>")
@login_required
def attendance_person(name):
    file_path = os.path.join(ATTEND_DIR, f"{name}.csv")
    records = []

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

        for date, group in df.groupby("Date"):
            in_time, out_time = None, None
            for _, row in group.iterrows():
                if row["Status"] == "IN" and not in_time:
                    in_time = row["Time"]
                if row["Status"] == "OUT":
                    out_time = row["Time"]

            work_duration = "--"
            if in_time and out_time:
                t1 = datetime.strptime(in_time, "%H:%M:%S")
                t2 = datetime.strptime(out_time, "%H:%M:%S")
                diff = t2 - t1
                hours = diff.seconds // 3600
                minutes = (diff.seconds % 3600) // 60
                work_duration = f"{hours:02d}:{minutes:02d}"

            records.append({"Date": date, "In": in_time or "--", "Out": out_time or "--", "Work": work_duration})

    return render_template("attendance_person.html", name=name, records=records)

# ================= DELETE BY DATE =================
@app.route("/attendance/delete_by_date", methods=["POST"])
@login_required
def delete_attendance_by_date():
    name = request.form.get("name")
    from_date = request.form.get("from_date")
    to_date = request.form.get("to_date")

    file_path = os.path.join(ATTEND_DIR, f"{name}.csv")
    if not os.path.exists(file_path):
        return redirect("/attendance")

    rows = []
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if from_date <= row["Date"] <= to_date:
                continue
            rows.append(row)

    with open(file_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Date", "Time", "Name", "Status"])
        writer.writeheader()
        writer.writerows(rows)

    ws_push_all()
    return redirect(f"/attendance/{name}")

# ================= DELETE PERSON CSV =================
@app.route("/attendance/delete_person/<name>")
@login_required
def delete_person_attendance(name):
    file_path = os.path.join(ATTEND_DIR, f"{name}.csv")
    if os.path.exists(file_path):
        os.remove(file_path)

    ws_push_all()
    return redirect("/attendance")

# ================= EXPORT =================
@app.route("/export/<name>")
@login_required
def export_person(name):
    file_path = os.path.join(ATTEND_DIR, f"{name}.csv")
    if not os.path.exists(file_path):
        return "No attendance data found"

    return send_file(
        file_path,
        as_attachment=True,
        download_name=f"{name}_attendance.csv"
    )

# ================= CHARTS =================
@app.route("/charts")
@login_required
def charts():
    labels = []
    values = []

    for file in os.listdir(ATTEND_DIR):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(ATTEND_DIR, file))
            labels.append(file.replace(".csv", ""))
            values.append(df["Status"].value_counts().get("IN", 0))

    plt.clf()
    plt.bar(labels, values)
    plt.title("Daily IN Count")
    plt.savefig("static/chart.png")

    return render_template("charts.html")

# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True)
