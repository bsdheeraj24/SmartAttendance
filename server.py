import os
import csv
import base64
import io
import shutil
import json
import binascii
import smtplib
import ssl
import threading
import time
from collections import OrderedDict
from datetime import datetime, time as dt_time, timedelta, timezone
from email.message import EmailMessage
from functools import wraps

from flask import (
    Flask, request, jsonify, render_template,
    redirect, session, send_file
)

import firebase_admin
from firebase_admin import credentials, firestore
from werkzeug.security import generate_password_hash, check_password_hash

import face_recognition
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================= CONFIG =================
KNOWN_DIR = "known_faces"
FACE_SAMPLES_COLLECTION = "face_samples"
ENROLL_SAMPLES = 10
MIN_GAP_SECONDS = int(os.environ.get("MIN_GAP_SECONDS", "60"))
POST_ENROLL_GAP_SECONDS = int(os.environ.get("POST_ENROLL_GAP_SECONDS", "60"))
MATCH_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", "0.58"))
NO_FACE_GRACE_SECONDS = float(os.environ.get("NO_FACE_GRACE_SECONDS", "2.0"))
STATUS_HOLD_SECONDS = float(os.environ.get("STATUS_HOLD_SECONDS", "2.5"))
IST = timezone(timedelta(hours=5, minutes=30), name="IST")


def now_ist():
    return datetime.now(timezone.utc).astimezone(IST).replace(tzinfo=None)


def _to_ist_naive(value):
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(IST).replace(tzinfo=None)


def _attendance_local_datetime(payload):
    local_dt = _to_ist_naive(payload.get("timestamp"))
    if local_dt is not None:
        return local_dt

    date_value = (payload.get("date") or "").strip()
    time_value = (payload.get("time") or "").strip()
    if not date_value or not time_value:
        return None

    try:
        return datetime.strptime(f"{date_value} {time_value}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _attendance_local_date_time(payload):
    local_dt = _attendance_local_datetime(payload)
    if local_dt is not None:
        return local_dt.strftime("%Y-%m-%d"), local_dt.strftime("%H:%M:%S")
    return payload.get("date", ""), payload.get("time", "")


def _is_true(value):
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _safe_doc_token(value):
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in (value or "").strip())
    return cleaned.strip("_") or "unknown"


def _parse_cutoff_time(value):
    text = (value or "09:45").strip()
    try:
        return datetime.strptime(text, "%H:%M").time()
    except ValueError:
        return datetime.strptime("09:45", "%H:%M").time()


LATE_ALERT_ENABLED = _is_true(os.environ.get("LATE_ALERT_ENABLED", "0"))
LATE_ALERT_PERSON = os.environ.get("LATE_ALERT_PERSON", "").strip()
LATE_ALERT_TO_EMAIL = os.environ.get("LATE_ALERT_TO_EMAIL", "").strip()
LATE_ALERT_CUTOFF = _parse_cutoff_time(os.environ.get("LATE_ALERT_CUTOFF", "09:45"))

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com").strip()
SMTP_PORT = int(os.environ.get("SMTP_PORT", "465"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "").strip()
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "").strip()
SMTP_FROM_EMAIL = os.environ.get("SMTP_FROM_EMAIL", SMTP_USERNAME).strip()

ESP32_CAM_IP = None

# ================= FOLDERS =================
os.makedirs(KNOWN_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)

# ================= FIREBASE =================
def _normalize_firebase_credentials(data):
    if isinstance(data, dict) and "private_key" in data and isinstance(data["private_key"], str):
        data["private_key"] = data["private_key"].replace("\\n", "\n")
    return data


def _load_firebase_credentials_from_raw(raw):
    """Load Firebase credentials from JSON or base64(JSON) text."""
    raw = raw.strip()
    if not raw:
        raise RuntimeError("Firebase credentials are empty")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            decoded = base64.b64decode(raw).decode("utf-8")
            data = json.loads(decoded)
        except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "FIREBASE_CREDENTIALS must be valid JSON or base64-encoded JSON"
            ) from exc

    return _normalize_firebase_credentials(data)


def _load_firebase_credentials_from_file(path, source_name):
    if not os.path.isfile(path):
        raise RuntimeError(f"{source_name} points to a missing file: {path}")

    try:
        with open(path, "r", encoding="utf-8") as handle:
            return _normalize_firebase_credentials(json.load(handle))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{source_name} must point to a valid JSON file: {path}") from exc


def _load_firebase_credentials():
    raw = os.environ.get("FIREBASE_CREDENTIALS", "").strip()
    if raw:
        return _load_firebase_credentials_from_raw(raw)

    for env_name in ("FIREBASE_CREDENTIALS_PATH", "GOOGLE_APPLICATION_CREDENTIALS"):
        path = os.environ.get(env_name, "").strip()
        if path:
            return _load_firebase_credentials_from_file(path, env_name)

    for candidate in ("serviceAccountKey.json", "firebase-service-account.json", "firebase_credentials.json"):
        if os.path.isfile(candidate):
            return _load_firebase_credentials_from_file(candidate, candidate)

    raise RuntimeError(
        "Firebase credentials not found. Set FIREBASE_CREDENTIALS, FIREBASE_CREDENTIALS_PATH, "
        "or GOOGLE_APPLICATION_CREDENTIALS, or add serviceAccountKey.json in the project root"
    )


FIREBASE_INIT_ERROR = None
db = None

try:
    firebase_creds = _load_firebase_credentials()
    cred = credentials.Certificate(firebase_creds)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as exc:
    FIREBASE_INIT_ERROR = str(exc)
    print(f"Firebase init failed: {FIREBASE_INIT_ERROR}")

# ================= FLASK =================
app = Flask(__name__, template_folder="templates")
app.secret_key = os.environ.get("SECRET_KEY", "attendance_secret_key_fallback")


_DEVICE_ENDPOINTS = {"healthz", "esp32_register", "set_mode", "status", "last_recognition"}

@app.before_request
def ensure_backend_ready():
    # Device endpoints (ESP32/ESP32-CAM) and health check don't require Firebase
    if request.endpoint in _DEVICE_ENDPOINTS:
        return None

    if db is None:
        return jsonify({
            "status": "CONFIG_ERROR",
            "message": "Firebase is not configured correctly.",
            "details": FIREBASE_INIT_ERROR,
        }), 503


@app.route("/healthz")
def healthz():
    return jsonify({
        "status": "ok" if db is not None else "config_error",
        "firebase_ready": db is not None,
        "details": FIREBASE_INIT_ERROR,
    }), 200


@app.route("/")
def index():
    return redirect("/login")

# ================= GLOBAL STATE =================
MODE = {"type": "idle", "name": None}
ENROLL_COUNT = 0

LAST_RESULT = {
    "status": "IDLE",
    "name": "",
    "entry": "",
    "confidence": 0
}
LAST_FACE_SEEN_AT = None
LAST_RESULT_SET_AT = now_ist()
LAST_ENROLL_COMPLETED_AT = {}

# ================= HELPERS =================
def _remove_person_from_meta(name):
    meta_ref = db.collection("metadata").document("attendance_persons")
    meta_doc = meta_ref.get()
    if meta_doc.exists:
        names = meta_doc.to_dict().get("names", [])
        if name in names:
            names.remove(name)
            meta_ref.update({"names": names})


def _replace_name_in_meta_list(doc_id, old_name, new_name):
    old_key = _face_name_key(old_name)
    new_normalized = _normalize_face_name(new_name)

    meta_ref = db.collection("metadata").document(doc_id)
    meta_doc = meta_ref.get()
    names = meta_doc.to_dict().get("names", []) if meta_doc.exists else []

    updated = []
    replaced = False
    for existing in names:
        if _face_name_key(existing) == old_key:
            if not replaced:
                updated.append(new_normalized)
                replaced = True
            continue
        updated.append(existing)

    if not replaced:
        updated.append(new_normalized)

    meta_ref.set({"names": _dedupe_face_names(updated)}, merge=True)


def _face_name_doc_id(name):
    return _safe_doc_token(_face_name_key(name))


def _image_to_base64_jpeg(img_pil, max_size=(480, 480), quality=75):
    sample = img_pil.copy()
    sample.thumbnail(max_size)
    output = io.BytesIO()
    sample.save(output, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(output.getvalue()).decode("utf-8")


def _store_face_sample(name, encoding_vec, img_pil, source):
    normalized_name = _normalize_face_name(name)
    if not _is_plausible_face_name(normalized_name):
        return

    db.collection(FACE_SAMPLES_COLLECTION).add({
        "name": normalized_name,
        "name_key": _face_name_key(normalized_name),
        "encoding": np.asarray(encoding_vec, dtype=np.float64).tolist(),
        "image_b64": _image_to_base64_jpeg(img_pil),
        "source": source,
        "created_at": firestore.SERVER_TIMESTAMP,
    })


def _delete_face_samples(name):
    normalized_name = _normalize_face_name(name)
    target_key = _face_name_key(normalized_name)

    docs = db.collection(FACE_SAMPLES_COLLECTION).where("name_key", "==", target_key).get()
    if not docs:
        docs = db.collection(FACE_SAMPLES_COLLECTION).where("name", "==", normalized_name).get()

    for i in range(0, len(docs), 500):
        batch = db.batch()
        for doc in docs[i:i + 500]:
            batch.delete(doc.reference)
        batch.commit()


def _get_face_sample_docs_by_name(name):
    normalized_name = _normalize_face_name(name)
    target_key = _face_name_key(normalized_name)

    by_id = {}
    for doc in db.collection(FACE_SAMPLES_COLLECTION).where("name_key", "==", target_key).stream():
        by_id[doc.id] = doc
    for doc in db.collection(FACE_SAMPLES_COLLECTION).where("name", "==", normalized_name).stream():
        by_id[doc.id] = doc
    return list(by_id.values())


def _rename_face_samples(old_name, new_name):
    docs = _get_face_sample_docs_by_name(old_name)
    if not docs:
        return

    new_normalized = _normalize_face_name(new_name)
    new_key = _face_name_key(new_normalized)

    for i in range(0, len(docs), 500):
        batch = db.batch()
        for doc in docs[i:i + 500]:
            batch.update(doc.reference, {
                "name": new_normalized,
                "name_key": new_key,
                "updated_at": firestore.SERVER_TIMESTAMP,
            })
        batch.commit()


def _rename_attendance_records(old_name, new_name):
    docs = db.collection("attendance").where("name", "==", old_name).get()
    for i in range(0, len(docs), 500):
        batch = db.batch()
        for doc in docs[i:i + 500]:
            batch.update(doc.reference, {"name": new_name})
        batch.commit()


def _rename_face_everywhere(old_name, new_name):
    old_normalized = _normalize_face_name(old_name)
    new_normalized = _normalize_face_name(new_name)

    old_key = _face_name_key(old_normalized)
    new_key = _face_name_key(new_normalized)

    if old_key == new_key:
        return

    _rename_face_samples(old_normalized, new_normalized)

    for i, existing in enumerate(known_names):
        if _face_name_key(existing) == old_key:
            known_names[i] = new_normalized

    _replace_name_in_meta_list("enrolled_faces", old_normalized, new_normalized)
    _replace_name_in_meta_list("attendance_persons", old_normalized, new_normalized)

    db.collection("enrolled_faces").document(_face_name_doc_id(new_normalized)).set({
        "name": new_normalized,
        "name_key": new_key,
        "updated_at": firestore.SERVER_TIMESTAMP,
    }, merge=True)
    db.collection("enrolled_faces").document(_face_name_doc_id(old_normalized)).delete()

    _rename_attendance_records(old_normalized, new_normalized)


def _load_known_faces_from_firestore():
    global known_encodings, known_names

    encodings = []
    names = []

    if db is None:
        known_encodings = encodings
        known_names = names
        return

    docs = db.collection(FACE_SAMPLES_COLLECTION).stream()
    for doc in docs:
        payload = doc.to_dict() or {}
        name = _normalize_face_name(payload.get("name", ""))
        encoding = payload.get("encoding")

        if not _is_plausible_face_name(name):
            continue
        if not isinstance(encoding, list) or len(encoding) == 0:
            continue

        try:
            vector = np.asarray(encoding, dtype=np.float64)
        except (ValueError, TypeError):
            continue

        if vector.ndim != 1:
            continue

        encodings.append(vector)
        names.append(name)

    known_encodings = encodings
    known_names = names


def _get_faces_meta_names():
    meta_ref = db.collection("metadata").document("enrolled_faces")
    meta_doc = meta_ref.get()
    if not meta_doc.exists:
        return []
    return meta_doc.to_dict().get("names", [])


def _normalize_face_name(name):
    if not isinstance(name, str):
        return ""

    normalized = " ".join(name.strip().split())

    # Some device payloads wrap enroll names as p<Name>t. Strip this wrapper
    # conservatively to avoid altering legitimate names.
    if len(normalized) > 2 and normalized[:1].lower() == "p" and normalized[-1:].lower() == "t":
        inner = normalized[1:-1].strip()
        if inner and inner[0].isupper():
            normalized = inner

    return normalized


def _face_name_key(name):
    return _normalize_face_name(name).casefold()


def _is_plausible_face_name(name):
    normalized = _normalize_face_name(name)
    if len(normalized) < 2:
        return False

    lowered = normalized.casefold()
    blocked = {
        "p", "profile", "profiles", "person", "persons", "people",
        "default", "metadata", "config", "settings", "unknown", "test",
    }
    if lowered in blocked:
        return False

    return any(ch.isalpha() for ch in normalized)


def _dedupe_face_names(names):
    by_key = {}
    for raw in names:
        normalized = _normalize_face_name(raw)
        if not _is_plausible_face_name(normalized):
            continue
        key = _face_name_key(normalized)
        if key not in by_key:
            by_key[key] = normalized
    return sorted(by_key.values(), key=str.lower)


def _extract_name_candidates(payload, fallback_id=""):
    if not isinstance(payload, dict):
        return []

    candidates = []
    for key in ("name", "full_name", "person_name", "person", "user", "username", "id"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            normalized = _normalize_face_name(value)
            if _is_plausible_face_name(normalized):
                candidates.append(normalized)

    if not candidates and fallback_id:
        normalized_id = _normalize_face_name(str(fallback_id))
        if _is_plausible_face_name(normalized_id):
            candidates.append(normalized_id)

    return candidates


def _get_faces_from_collection(collection_name):
    names = set()
    try:
        docs = db.collection(collection_name).stream()
        for doc in docs:
            payload = doc.to_dict() or {}
            for candidate in _extract_name_candidates(payload, fallback_id=doc.id):
                if candidate:
                    names.add(candidate)
    except Exception:
        return []
    return sorted(names, key=str.lower)


def _get_faces_from_firestore():
    names = set(_dedupe_face_names(_get_faces_meta_names()))

    # Canonical source for enrolled faces is face_samples.
    # Using many legacy collections can re-introduce old names after rename.
    try:
        for doc in db.collection(FACE_SAMPLES_COLLECTION).stream():
            payload = doc.to_dict() or {}
            normalized = _normalize_face_name(payload.get("name", ""))
            if _is_plausible_face_name(normalized):
                names.add(normalized)
    except Exception:
        pass

    try:
        attendance_meta = db.collection("metadata").document("attendance_persons").get()
        if attendance_meta.exists:
            for name in attendance_meta.to_dict().get("names", []):
                normalized = _normalize_face_name(name)
                if normalized:
                    names.add(normalized)
    except Exception:
        pass

    return _dedupe_face_names(names)


def _add_face_to_meta(name):
    normalized_name = _normalize_face_name(name)
    if not normalized_name:
        return

    names = _dedupe_face_names(_get_faces_meta_names())
    existing_keys = {_face_name_key(n) for n in names}
    if _face_name_key(normalized_name) in existing_keys:
        return

    names.append(normalized_name)
    db.collection("metadata").document("enrolled_faces").set({"names": _dedupe_face_names(names)}, merge=True)
    db.collection("enrolled_faces").document(_face_name_doc_id(normalized_name)).set({
        "name": normalized_name,
        "name_key": _face_name_key(normalized_name),
        "updated_at": firestore.SERVER_TIMESTAMP,
    }, merge=True)


def _remove_face_from_meta(name):
    normalized_name = _normalize_face_name(name)
    if not normalized_name:
        return

    names = _dedupe_face_names(_get_faces_meta_names())
    target_key = _face_name_key(normalized_name)
    kept = [n for n in names if _face_name_key(n) != target_key]
    if len(kept) == len(names):
        return

    db.collection("metadata").document("enrolled_faces").set({"names": kept}, merge=True)
    db.collection("enrolled_faces").document(_face_name_doc_id(normalized_name)).delete()


def _recover_esp32_cam_ip():
    global ESP32_CAM_IP
    if ESP32_CAM_IP or db is None:
        return ESP32_CAM_IP

    try:
        doc = db.collection("metadata").document("device_state").get()
        if doc.exists:
            ESP32_CAM_IP = doc.to_dict().get("esp32_cam_ip")
    except Exception:
        pass

    return ESP32_CAM_IP


def _current_stream_url():
    ip = _recover_esp32_cam_ip()
    return f"http://{ip}:81/stream" if ip else ""


def _set_last_result(data):
    global LAST_RESULT, LAST_RESULT_SET_AT
    LAST_RESULT = data
    LAST_RESULT_SET_AT = now_ist()


def _hold_last_result_for_esp32():
    holdable_statuses = {"ATTENDANCE_MARKED", "WAIT", "UNKNOWN", "NO_KNOWN_FACES"}
    if MODE.get("type") != "attend" or LAST_RESULT.get("status") not in holdable_statuses:
        return None

    age = (now_ist() - LAST_RESULT_SET_AT).total_seconds()
    if age <= STATUS_HOLD_SECONDS:
        return LAST_RESULT

    return None


def _hold_previous_result_on_no_face():
    if MODE.get("type") != "attend" or LAST_FACE_SEEN_AT is None:
        return None

    age = (now_ist() - LAST_FACE_SEEN_AT).total_seconds()
    if age <= NO_FACE_GRACE_SECONDS and LAST_RESULT.get("status") in {
        "ATTENDANCE_MARKED", "WAIT", "UNKNOWN", "NO_KNOWN_FACES"
    }:
        return LAST_RESULT

    return None


def _get_first_in_time_for_date(person_name, date_value):
    docs = (
        db.collection("attendance")
        .where("name", "==", person_name)
        .where("status", "==", "IN")
        .get()
    )

    if not docs:
        return None

    local_times = []
    for doc in docs:
        local_dt = _attendance_local_datetime(doc.to_dict())
        if local_dt is not None and local_dt.strftime("%Y-%m-%d") == date_value:
            local_times.append(local_dt)

    if not local_times:
        return None

    return min(local_times).time()


def _check_late_status(person_name, now_dt):
    date_value = now_dt.strftime("%Y-%m-%d")
    first_in_time = _get_first_in_time_for_date(person_name, date_value)

    if first_in_time is None:
        return True, "No IN attendance marked before cutoff", None

    if first_in_time > LATE_ALERT_CUTOFF:
        return True, f"First IN time was {first_in_time.strftime('%H:%M:%S')}", first_in_time

    return False, "On time", first_in_time


def _late_alert_doc_ref(person_name, date_value):
    doc_id = f"{date_value}_{_safe_doc_token(person_name)}"
    return db.collection("late_alerts").document(doc_id)


def _acquire_daily_late_alert_lock(person_name, date_value):
    doc_ref = _late_alert_doc_ref(person_name, date_value)
    doc_ref.create({
        "name": person_name,
        "date": date_value,
        "status": "processing",
        "created_at": firestore.SERVER_TIMESTAMP,
    })
    return doc_ref


def _send_late_alert_email(person_name, date_value, reason_text):
    if not all([SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SMTP_FROM_EMAIL, LATE_ALERT_TO_EMAIL]):
        print("Late alert email skipped: SMTP settings or recipient are incomplete")
        return False

    msg = EmailMessage()
    msg["Subject"] = f"Late Attendance Alert - {person_name} ({date_value})"
    msg["From"] = SMTP_FROM_EMAIL
    msg["To"] = LATE_ALERT_TO_EMAIL
    msg.set_content(
        f"Attendance late alert for {person_name} on {date_value}.\n"
        f"Cutoff time: {LATE_ALERT_CUTOFF.strftime('%H:%M')}\n"
        f"Reason: {reason_text}\n"
    )

    if SMTP_PORT == 465:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context, timeout=20) as server:
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
    else:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
            server.starttls(context=ssl.create_default_context())
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)

    return True


def _late_alert_worker():
    print("Late alert worker started")
    while True:
        try:
            if db is None or not LATE_ALERT_PERSON:
                time.sleep(60)
                continue

            now_dt = now_ist()
            cutoff_dt = now_dt.replace(
                hour=LATE_ALERT_CUTOFF.hour,
                minute=LATE_ALERT_CUTOFF.minute,
                second=0,
                microsecond=0,
            )

            if now_dt < cutoff_dt:
                time.sleep(60)
                continue

            date_value = now_dt.strftime("%Y-%m-%d")
            is_late, reason_text, first_in_time = _check_late_status(LATE_ALERT_PERSON, now_dt)

            if not is_late:
                time.sleep(60)
                continue

            try:
                doc_ref = _acquire_daily_late_alert_lock(LATE_ALERT_PERSON, date_value)
            except Exception as exc:
                # Firestore raises an "already exists" error if another worker created
                # this day's lock first. Skip in that case to avoid duplicate emails.
                if "already exists" in str(exc).lower():
                    time.sleep(60)
                    continue
                raise

            try:
                email_sent = _send_late_alert_email(LATE_ALERT_PERSON, date_value, reason_text)
                if email_sent:
                    payload = {
                        "status": "sent",
                        "reason": reason_text,
                        "sent_to": LATE_ALERT_TO_EMAIL,
                        "updated_at": firestore.SERVER_TIMESTAMP,
                    }
                    if first_in_time is not None:
                        payload["first_in"] = first_in_time.strftime("%H:%M:%S")
                    doc_ref.set(payload, merge=True)
                    print(f"Late alert sent for {LATE_ALERT_PERSON} on {date_value}")
                else:
                    doc_ref.delete()
            except Exception as exc:
                print(f"Late alert send failed: {exc}")
                doc_ref.delete()

        except Exception as exc:
            print(f"Late alert worker error: {exc}")

        time.sleep(60)


def _start_late_alert_worker():
    if not LATE_ALERT_ENABLED:
        print("Late alert worker disabled")
        return

    if not LATE_ALERT_PERSON:
        print("Late alert worker disabled: LATE_ALERT_PERSON is not set")
        return

    worker = threading.Thread(target=_late_alert_worker, daemon=True)
    worker.start()

# ================= USERS =================
def load_users():
    users = {}
    docs = db.collection("users").stream()
    for doc in docs:
        users[doc.id] = doc.to_dict()
    return users

# ================= LOAD ENCODINGS =================
known_encodings = []
known_names = []
_load_known_faces_from_firestore()

# ================= LOGIN REQUIRED =================
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if "user" not in session:
            return redirect("/login")
        return f(*args, **kwargs)
    return wrap

# ================= FILTER =================
@app.template_filter("ddmmyyyy")
def ddmmyyyy(date_str):
    try:
        y, m, d = date_str.split("-")
        return f"{d}-{m}-{y}"
    except:
        return date_str

# ================= ESP32-CAM REGISTER =================
@app.route("/esp32_register", methods=["POST"])
def esp32_register():
    global ESP32_CAM_IP
    data = request.get_json(force=True, silent=True) or {}
    ESP32_CAM_IP = (data.get("ip") or "").strip()

    if db is not None and ESP32_CAM_IP:
        db.collection("metadata").document("device_state").set({
            "esp32_cam_ip": ESP32_CAM_IP,
            "updated_at": firestore.SERVER_TIMESTAMP,
        }, merge=True)

    print("ESP32-CAM Registered IP:", ESP32_CAM_IP)
    return jsonify({"status": "ok", "ip": ESP32_CAM_IP})

# ================= AUTH =================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        doc = db.collection("users").document(username).get()
        if doc.exists:
            user = doc.to_dict()
            if check_password_hash(user["password_hash"], password):
                session["user"] = username
                session["role"] = user.get("role", "user")
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
    stream_url = _current_stream_url()
    is_admin = session.get("role") == "admin"
    # Show setup button only if no admin exists yet
    has_admin = any(
        u.to_dict().get("role") == "admin"
        for u in db.collection("users").stream()
    ) if db is not None else True
    return render_template("dashboard.html", stream_url=stream_url, is_admin=is_admin, no_admin=not has_admin)

# ================= USERS =================
@app.route("/setup_admin", methods=["POST"])
@login_required
def setup_admin():
    """One-time bootstrap: promotes the logged-in user to admin when no admin exists yet."""
    admins = [
        u for u in db.collection("users").stream()
        if u.to_dict().get("role") == "admin"
    ]
    if admins:
        return redirect("/dashboard")

    current_user = session.get("user")
    db.collection("users").document(current_user).update({"role": "admin"})
    session["role"] = "admin"
    return redirect("/dashboard")

@app.route("/users", methods=["GET", "POST"])
@login_required
def manage_users():
    if session.get("role") != "admin":
        return "Access denied"

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if username and password:
            db.collection("users").document(username).set({
                "password_hash": generate_password_hash(password),
                "role": "user",
            })

    users = load_users()
    return render_template("users.html", users=users)

@app.route("/delete_user/<username>")
@login_required
def delete_user(username):
    if session.get("role") != "admin":
        return "Access denied"

    if username == session.get("user"):
        return "Cannot delete current admin"

    db.collection("users").document(username).delete()
    return redirect("/users")

@app.route("/change_credentials", methods=["POST"])
@login_required
def change_credentials():
    if session.get("role") != "admin":
        return "Access denied"

    current_password = request.form.get("current_password", "").strip()
    new_username     = request.form.get("new_username", "").strip()
    new_password     = request.form.get("new_password", "").strip()
    confirm_password = request.form.get("confirm_password", "").strip()

    current_user = session.get("user")
    doc = db.collection("users").document(current_user).get()
    if not doc.exists:
        users = load_users()
        return render_template("users.html", users=users, cred_error="Current user not found.")

    user_data = doc.to_dict()
    if not check_password_hash(user_data["password_hash"], current_password):
        users = load_users()
        return render_template("users.html", users=users, cred_error="Current password is incorrect.")

    if new_password and new_password != confirm_password:
        users = load_users()
        return render_template("users.html", users=users, cred_error="New passwords do not match.")

    if not new_username and not new_password:
        users = load_users()
        return render_template("users.html", users=users, cred_error="Provide a new username or password.")

    target_username = new_username if new_username else current_user
    updated_hash    = generate_password_hash(new_password) if new_password else user_data["password_hash"]

    if new_username and new_username != current_user:
        # Create new doc, delete old one
        db.collection("users").document(new_username).set({
            "password_hash": updated_hash,
            "role": user_data.get("role", "admin"),
        })
        db.collection("users").document(current_user).delete()
        session["user"] = new_username
    else:
        db.collection("users").document(current_user).update({"password_hash": updated_hash})

    users = load_users()
    return render_template("users.html", users=users, cred_success="Credentials updated successfully.")

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

    _set_last_result({
        "status": MODE["type"].upper(),
        "name": MODE["name"] or "",
        "entry": "",
        "confidence": 0
    })

    return jsonify({"status": "ok", "mode": MODE})

@app.route("/status")
def status():
    return jsonify({
        "mode": MODE["type"],
        "name": MODE["name"],
        "enroll_count": ENROLL_COUNT,
        "last": LAST_RESULT,
        "stream_url": _current_stream_url(),
    })

@app.route("/last_recognition")
def last_recognition():
    return jsonify(LAST_RESULT)

# ================= CAPTURE (from ESP32-CAM) =================
@app.route("/capture", methods=["POST"])
def capture():
    global ENROLL_COUNT, LAST_RESULT, LAST_FACE_SEEN_AT, LAST_ENROLL_COMPLETED_AT
    global known_encodings, known_names

    data = request.get_json(force=True, silent=True) or {}
    if "image" not in data:
        return jsonify({"status": "NO_IMAGE"})

    try:
        img_bytes = base64.b64decode(data["image"])
        img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except:
        return jsonify({"status": "BAD_IMAGE"})

    held = _hold_last_result_for_esp32()
    if held is not None:
        return jsonify(held)

    frame = np.array(img_pil)

    # Detect faces explicitly and choose the largest one for better stability.
    locations = face_recognition.face_locations(frame, model="hog")
    if not locations:
        held = _hold_previous_result_on_no_face()
        if held is not None:
            return jsonify(held)
        _set_last_result({"status": "NO_FACE", "name": "", "entry": "", "confidence": 0})
        return jsonify(LAST_RESULT)

    best_loc = max(locations, key=lambda loc: max(0, loc[2] - loc[0]) * max(0, loc[1] - loc[3]))
    encodings = face_recognition.face_encodings(
        frame,
        known_face_locations=[best_loc],
        num_jitters=1,
    )
    if not encodings:
        held = _hold_previous_result_on_no_face()
        if held is not None:
            return jsonify(held)
        _set_last_result({"status": "NO_FACE", "name": "", "entry": "", "confidence": 0})
        return jsonify(LAST_RESULT)

    face = encodings[0]
    LAST_FACE_SEEN_AT = now_ist()

    # -------- ENROLL --------
    if MODE["type"] == "enroll":
        name = _normalize_face_name(MODE["name"])
        if not _is_plausible_face_name(name):
            return jsonify({"status": "NO_NAME", "error": "Invalid or missing name"})

        _store_face_sample(name, face, img_pil, source="esp32_enroll")

        known_encodings.append(face)
        known_names.append(name)
        ENROLL_COUNT += 1

        if ENROLL_COUNT >= ENROLL_SAMPLES:
            _add_face_to_meta(name)
            LAST_ENROLL_COMPLETED_AT[_face_name_key(name)] = now_ist()

            MODE["type"] = "idle"
            MODE["name"] = None

            _set_last_result({"status": "ENROLL_COMPLETE", "name": name, "entry": "", "confidence": 100})
            return jsonify(LAST_RESULT)

        _set_last_result({"status": "ENROLLING", "name": name, "entry": "", "confidence": 0})
        return jsonify({"status": "ENROLLING", "count": ENROLL_COUNT})

    # -------- ATTEND --------
    if not known_encodings:
        _set_last_result({"status": "NO_KNOWN_FACES", "name": "", "entry": "", "confidence": 0})
        return jsonify(LAST_RESULT)

    distances = face_recognition.face_distance(known_encodings, face)
    best_idx = int(np.argmin(distances))
    best_distance = float(distances[best_idx])

    # Lower distance means a better match. Keep threshold conservative.
    if best_distance > MATCH_THRESHOLD:
        _set_last_result({"status": "UNKNOWN", "name": "", "entry": "", "confidence": 0})
        return jsonify(LAST_RESULT)

    name = known_names[best_idx]
    confidence = max(0, min(99, int((1.0 - best_distance) * 100)))

    now = now_ist()
    enroll_done_at = LAST_ENROLL_COMPLETED_AT.get(_face_name_key(name))
    if enroll_done_at is not None:
        enroll_gap = (now - enroll_done_at).total_seconds()
        if enroll_gap < POST_ENROLL_GAP_SECONDS:
            remaining = int(POST_ENROLL_GAP_SECONDS - enroll_gap)
            _set_last_result({"status": "WAIT", "name": name, "entry": "", "confidence": 0})
            return jsonify({"status": "WAIT", "seconds": max(1, remaining)})

    today = now.strftime("%Y-%m-%d")
    entry = "IN"

    # Query today's entries for this person
    today_docs = db.collection("attendance").where("name", "==", name).get()
    today_docs = [
        doc.to_dict()
        for doc in today_docs
        if _attendance_local_date_time(doc.to_dict())[0] == today
    ]

    # Sort in Python to avoid requiring a Firestore composite index.
    today_docs = sorted(
        today_docs,
        key=lambda d: _attendance_local_datetime(d) or datetime.min
    )

    if today_docs:
        last_doc = today_docs[-1]
        last_status = last_doc["status"]
        last_time = _attendance_local_datetime(last_doc)
        if last_time is None:
            last_time = now
        diff = (now - last_time).total_seconds()

        if diff < MIN_GAP_SECONDS:
            _set_last_result({"status": "WAIT", "name": name, "entry": "", "confidence": 0})
            return jsonify({"status": "WAIT", "seconds": int(MIN_GAP_SECONDS - diff)})

        entry = "OUT" if last_status == "IN" else "IN"

    # Write new attendance record
    db.collection("attendance").add({
        "name": name,
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "status": entry,
        "timestamp": firestore.SERVER_TIMESTAMP,
    })

    # Update metadata (add person if not already tracked)
    meta_ref = db.collection("metadata").document("attendance_persons")
    meta_doc = meta_ref.get()
    if meta_doc.exists:
        names_list = meta_doc.to_dict().get("names", [])
        if name not in names_list:
            names_list.append(name)
            meta_ref.update({"names": names_list})
    else:
        meta_ref.set({"names": [name]})

    _set_last_result({"status": "ATTENDANCE_MARKED", "name": name, "entry": entry, "confidence": confidence})
    return jsonify(LAST_RESULT)

# ================= FACES =================
@app.route("/faces")
@login_required
def faces():
    persons = _get_faces_from_firestore()

    return render_template("faces.html", users=persons)


@app.route("/rename_face", methods=["POST"])
@login_required
def rename_face():
    old_name = _normalize_face_name(request.form.get("old_name", ""))
    new_name = _normalize_face_name(request.form.get("new_name", ""))

    if not _is_plausible_face_name(old_name):
        return "Original name is invalid", 400

    if not _is_plausible_face_name(new_name):
        return "New name is invalid", 400

    existing = {_face_name_key(name) for name in _get_faces_from_firestore()}
    if _face_name_key(new_name) != _face_name_key(old_name) and _face_name_key(new_name) in existing:
        return "New name already exists", 400

    _rename_face_everywhere(old_name, new_name)
    return redirect("/faces")

@app.route("/delete/<name>")
@login_required
def delete_face(name):
    global known_encodings, known_names

    normalized_name = _normalize_face_name(name)
    target_key = _face_name_key(normalized_name)
    idxs = [i for i, n in enumerate(known_names) if _face_name_key(n) == target_key]
    for i in sorted(idxs, reverse=True):
        known_encodings.pop(i)
        known_names.pop(i)

    _delete_face_samples(normalized_name)
    shutil.rmtree(os.path.join(KNOWN_DIR, normalized_name), ignore_errors=True)

    _remove_face_from_meta(name)

    # Delete attendance records from Firestore
    att_docs = db.collection("attendance").where("name", "==", name).get()
    for i in range(0, len(att_docs), 500):
        batch = db.batch()
        for doc in att_docs[i:i + 500]:
            batch.delete(doc.reference)
        batch.commit()
    _remove_person_from_meta(name)

    return redirect("/faces")

# ================= ADD FACE (single) =================
@app.route("/add_face", methods=["GET", "POST"])
@login_required
def add_face():
    global known_encodings, known_names

    if request.method == "POST":
        name = _normalize_face_name(request.form.get("name", ""))
        file = request.files.get("image")

        if not _is_plausible_face_name(name) or not file:
            return "Name or image missing"

        img_pil = Image.open(file.stream).convert("RGB")
        img_pil.thumbnail((640, 480))

        enc = face_recognition.face_encodings(np.array(img_pil))
        if not enc:
            return "No face detected. Use a clear photo."

        _store_face_sample(name, enc[0], img_pil, source="web_single")
        known_encodings.append(enc[0])
        known_names.append(name)

        _add_face_to_meta(name)

        return redirect("/faces")

    return render_template("add_face.html")

# ================= ADD FACE UPLOAD (auto 10) =================
@app.route("/add_face_upload", methods=["GET", "POST"])
@login_required
def add_face_upload():
    global known_encodings, known_names

    if request.method == "POST":
        name = _normalize_face_name(request.form.get("name", ""))
        file = request.files.get("image")

        if not _is_plausible_face_name(name) or not file:
            return "Name or image missing"

        img_pil = Image.open(file.stream).convert("RGB")
        img_pil.thumbnail((640, 480))

        for i in range(1, 11):
            enc = face_recognition.face_encodings(np.array(img_pil))
            if enc:
                _store_face_sample(name, enc[0], img_pil, source="web_upload")
                known_encodings.append(enc[0])
                known_names.append(name)

        _add_face_to_meta(name)

        return redirect("/faces")

    return render_template("add_face_upload.html")

# ================= ADD FACE CAPTURE (10 files upload) =================
@app.route("/add_face_capture", methods=["GET", "POST"])
@login_required
def add_face_capture():
    global known_encodings, known_names

    if request.method == "POST":
        name = _normalize_face_name(request.form.get("name", ""))
        files = request.files.getlist("images")

        if not _is_plausible_face_name(name) or len(files) != 10:
            return "10 images required"

        for i, file in enumerate(files, start=1):
            img_pil = Image.open(file.stream).convert("RGB")
            img_pil.thumbnail((640, 480))

            enc = face_recognition.face_encodings(np.array(img_pil))
            if enc:
                _store_face_sample(name, enc[0], img_pil, source="web_capture")
                known_encodings.append(enc[0])
                known_names.append(name)

        _add_face_to_meta(name)

        return redirect("/faces")

    return render_template("add_face_capture.html")

# ================= ATTENDANCE =================
@app.route("/attendance")
@login_required
def attendance():
    meta_doc = db.collection("metadata").document("attendance_persons").get()
    persons = meta_doc.to_dict().get("names", []) if meta_doc.exists else []
    return render_template("attendance_list.html", persons=persons)

@app.route("/attendance/<name>")
@login_required
def attendance_person(name):
    docs = (
        db.collection("attendance")
        .where("name", "==", name)
        .get()
    )
    rows = []
    for doc in docs:
        payload = doc.to_dict()
        local_date, local_time = _attendance_local_date_time(payload)
        rows.append({
            "date": local_date,
            "time": local_time,
            "status": payload.get("status", ""),
            "sort_dt": _attendance_local_datetime(payload),
        })

    # Sort in Python to avoid requiring a Firestore composite index
    rows = sorted(rows, key=lambda d: (d["sort_dt"] or datetime.min, d["status"]))

    grouped = OrderedDict()
    for row in rows:
        date = row["date"]
        if date not in grouped:
            grouped[date] = []
        grouped[date].append(row)

    records = []
    for date, entries in grouped.items():
        in_time, out_time = None, None
        for e in entries:
            if e["status"] == "IN" and not in_time:
                in_time = e["time"]
            if e["status"] == "OUT":
                out_time = e["time"]

        work_duration = "--"
        if in_time and out_time:
            t1 = datetime.strptime(in_time, "%H:%M:%S")
            t2 = datetime.strptime(out_time, "%H:%M:%S")
            diff = t2 - t1
            hours = diff.seconds // 3600
            minutes = (diff.seconds % 3600) // 60
            work_duration = f"{hours:02d}:{minutes:02d}"

        records.append({
            "Date": date,
            "In": in_time or "--",
            "Out": out_time or "--",
            "Work": work_duration,
        })

    return render_template("attendance_person.html", name=name, records=records)

# ================= DELETE BY DATE =================
@app.route("/attendance/delete_by_date", methods=["POST"])
@login_required
def delete_attendance_by_date():
    name = (request.form.get("name") or "").strip()
    from_date = (request.form.get("from_date") or "").strip()
    to_date = (request.form.get("to_date") or "").strip()

    if not name or not from_date or not to_date:
        return "Name and date range are required", 400

    try:
        from_dt = datetime.strptime(from_date, "%Y-%m-%d")
        to_dt = datetime.strptime(to_date, "%Y-%m-%d")
    except ValueError:
        return "Invalid date format", 400

    if from_dt > to_dt:
        return "From date cannot be later than To date", 400

    # Query by name only and filter date range in Python to avoid composite index errors.
    docs = db.collection("attendance").where("name", "==", name).get()
    docs = [
        doc for doc in docs
        if from_date <= _attendance_local_date_time(doc.to_dict())[0] <= to_date
    ]

    for i in range(0, len(docs), 500):
        batch = db.batch()
        for doc in docs[i:i + 500]:
            batch.delete(doc.reference)
        batch.commit()

    # Remove person from metadata if no records remain
    remaining = db.collection("attendance").where("name", "==", name).limit(1).get()
    if not remaining:
        _remove_person_from_meta(name)

    return redirect(f"/attendance/{name}")

# ================= DELETE PERSON ATTENDANCE =================
@app.route("/attendance/delete_person/<name>")
@login_required
def delete_person_attendance(name):
    docs = db.collection("attendance").where("name", "==", name).get()
    for i in range(0, len(docs), 500):
        batch = db.batch()
        for doc in docs[i:i + 500]:
            batch.delete(doc.reference)
        batch.commit()

    _remove_person_from_meta(name)
    return redirect("/attendance")

# ================= EXPORT =================
@app.route("/export/<name>")
@login_required
def export_person(name):
    docs = db.collection("attendance").where("name", "==", name).get()

    if not docs:
        return "No attendance data found"

    rows = []
    for doc in docs:
        payload = doc.to_dict()
        local_date, local_time = _attendance_local_date_time(payload)
        rows.append({
            "date": local_date,
            "time": local_time,
            "name": payload.get("name", name),
            "status": payload.get("status", ""),
            "sort_dt": _attendance_local_datetime(payload),
        })

    rows = sorted(rows, key=lambda d: (d["sort_dt"] or datetime.min, d["status"]))

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Date", "Time", "Name", "Status"])
    for row in rows:
        writer.writerow([
            row["date"],
            row["time"],
            row["name"],
            row["status"],
        ])

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8")),
        as_attachment=True,
        download_name=f"{name}_attendance.csv",
        mimetype="text/csv",
    )

# ================= CHARTS =================
@app.route("/charts")
@login_required
def charts():
    selected_date = (request.args.get("date") or "").strip()
    if not selected_date:
        selected_date = now_ist().strftime("%Y-%m-%d")
    else:
        try:
            datetime.strptime(selected_date, "%Y-%m-%d")
        except ValueError:
            selected_date = now_ist().strftime("%Y-%m-%d")

    docs = db.collection("attendance").where("status", "==", "IN").get()

    counts = {}
    for doc in docs:
        payload = doc.to_dict()
        local_date, _ = _attendance_local_date_time(payload)
        if local_date != selected_date:
            continue

        n = payload.get("name", "Unknown")
        counts[n] = counts.get(n, 0) + 1

    labels = sorted(counts.keys())
    values = list(counts.values())
    pretty_date = datetime.strptime(selected_date, "%Y-%m-%d").strftime("%d-%m-%Y")

    plt.clf()
    if labels:
        plt.bar(labels, values)
    plt.title(f"Daily IN Count ({pretty_date})")
    plt.savefig("static/chart.png")

    return render_template("charts.html", chart_date=selected_date, chart_pretty_date=pretty_date)

# ================= RUN =================
_start_late_alert_worker()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
