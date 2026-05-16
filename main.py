import asyncio
import os
import sqlite3
import secrets
import smtplib
import base64
import json
import re
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from email_validator import validate_email, EmailNotValidError
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import bcrypt
import jwt
import numpy as np
import cv2
from typing import Optional
from openai import OpenAI
_openai_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=_openai_key) if _openai_key else None
print(f"OpenAI key loaded: {bool(_openai_key)}")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


JWT_SECRET = os.environ.get("JWT_SECRET", "changeme-use-env-var")
SMTP_EMAIL = os.environ.get("SMTP_EMAIL")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
DB = "scores.db"
UPLOAD_DIR = "uploads"
Path(UPLOAD_DIR).mkdir(exist_ok=True)


# ── DB SETUP ──────────────────────────────────────────────────────────────────


def get_db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            email TEXT UNIQUE,
            password_hash TEXT,
            first_name TEXT,
            last_name TEXT,
            photo_url TEXT,
            reset_token TEXT,
            reset_token_expires TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            distance_cm REAL,
            description TEXT,
            bar_name TEXT,
            bar_rating INTEGER,
            fresh_photo_uri TEXT,
            lat REAL,
            lng REAL,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS follows (
            follower TEXT,
            following TEXT,
            PRIMARY KEY (follower, following)
        )
    """)
    # Migrations
    for col in [
        "ALTER TABLE users ADD COLUMN first_name TEXT",
        "ALTER TABLE users ADD COLUMN last_name TEXT",
        "ALTER TABLE users ADD COLUMN photo_url TEXT",
        "ALTER TABLE users ADD COLUMN reset_token TEXT",
        "ALTER TABLE users ADD COLUMN reset_token_expires TEXT",
    ]:
        try:
            c.execute(col)
        except Exception:
            pass
    conn.commit()
    conn.close()


init_db()


# ── AUTH HELPERS ──────────────────────────────────────────────────────────────


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def create_token(username: str) -> str:
    payload = {
        "username": username,
        "exp": datetime.utcnow() + timedelta(days=30)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def validate_email_address(email: str) -> bool:
    try:
        validate_email(email, check_deliverability=False)
        return True
    except EmailNotValidError:
        return False


def send_reset_email(to_email: str, username: str, reset_token: str):
    reset_link = f"https://splittheg.app/reset?token={reset_token}"
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Reset your Split the G password"
    msg["From"] = SMTP_EMAIL
    msg["To"] = to_email
    html = f"""
    <html>
    <body style="font-family:sans-serif;background:#0a0a0a;color:#fff;padding:32px;max-width:480px;margin:0 auto;">
      <h2 style="color:#FDB913;margin-bottom:8px;">Split the G</h2>
      <p style="color:#aaa;margin-bottom:24px;">Hey {username}, you requested a password reset.</p>
      <a href="{reset_link}"
         style="display:inline-block;background:#FDB913;color:#000;
                padding:14px 28px;border-radius:8px;font-weight:bold;
                text-decoration:none;font-size:16px;margin-bottom:24px;">
        Reset Password
      </a>
      <p style="color:#555;font-size:13px;margin-top:24px;">
        This link expires in 1 hour.<br>
        If you didn't request this, you can safely ignore this email.
      </p>
    </body>
    </html>
    """
    msg.attach(MIMEText(html, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.sendmail(SMTP_EMAIL, to_email, msg.as_string())
        return True
    except Exception as e:
        print(f"Email send failed: {e}")
        return False


# ── AUTH ROUTES ───────────────────────────────────────────────────────────────


@app.post("/auth/signup")
async def signup(body: dict):
    email = body.get("email", "").strip().lower()
    username = body.get("username", "").strip()
    password = body.get("password", "")
    first_name = body.get("first_name", "").strip()
    last_name = body.get("last_name", "").strip()

    if not first_name or not last_name:
        return {"error": "First and last name are required"}
    if not username:
        return {"error": "Username is required"}
    if not validate_email_address(email):
        return {"error": "Invalid email address"}
    if len(password) < 8:
        return {"error": "Password must be at least 8 characters"}

    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT username FROM users WHERE email = ?", (email,))
    if c.fetchone():
        conn.close()
        return {"error": "Email already registered"}
    c.execute("SELECT username FROM users WHERE username = ?", (username,))
    if c.fetchone():
        conn.close()
        return {"error": "Username already taken"}

    c.execute(
        "INSERT INTO users (username, email, password_hash, first_name, last_name) VALUES (?, ?, ?, ?, ?)",
        (username, email, hash_password(password), first_name, last_name)
    )
    conn.commit()
    conn.close()

    token = create_token(username)
    return {
        "status": "created",
        "token": token,
        "username": username,
        "first_name": first_name,
        "last_name": last_name,
        "photo_url": None,
        "message": f"Welcome, {first_name}!"
    }


@app.post("/auth/login")
async def login(body: dict):
    username = body.get("username", "").strip()
    password = body.get("password", "")

    conn = get_db()
    c = conn.cursor()
    c.execute(
        "SELECT username, password_hash, first_name, last_name, photo_url FROM users WHERE username = ?",
        (username,)
    )
    row = c.fetchone()
    conn.close()

    if not row:
        return {"error": "No account found with that username"}
    if not verify_password(password, row["password_hash"]):
        return {"error": "Incorrect password"}

    token = create_token(row["username"])
    return {
        "status": "ok",
        "token": token,
        "username": row["username"],
        "first_name": row["first_name"],
        "last_name": row["last_name"],
        "photo_url": row["photo_url"],
    }


@app.post("/auth/forgot-password")
async def forgot_password(body: dict):
    email = body.get("email", "").strip().lower()
    if not email:
        return {"error": "Email is required"}

    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT username, email FROM users WHERE email = ?", (email,))
    row = c.fetchone()

    if not row:
        conn.close()
        return {"status": "sent", "message": "If that email exists, a reset link has been sent."}

    reset_token = secrets.token_urlsafe(32)
    expires = (datetime.utcnow() + timedelta(hours=1)).isoformat()
    c.execute(
        "UPDATE users SET reset_token = ?, reset_token_expires = ? WHERE email = ?",
        (reset_token, expires, email)
    )
    conn.commit()
    conn.close()

    send_reset_email(email, row["username"], reset_token)
    return {"status": "sent", "message": "If that email exists, a reset link has been sent."}


@app.post("/auth/reset-password")
async def reset_password(body: dict):
    token = body.get("token", "").strip()
    new_password = body.get("new_password", "")

    if len(new_password) < 8:
        return {"error": "Password must be at least 8 characters"}

    conn = get_db()
    c = conn.cursor()
    c.execute(
        "SELECT username, reset_token_expires FROM users WHERE reset_token = ?",
        (token,)
    )
    row = c.fetchone()
    if not row:
        conn.close()
        return {"error": "Invalid or expired reset link"}

    if datetime.utcnow() > datetime.fromisoformat(row["reset_token_expires"]):
        conn.close()
        return {"error": "Reset link has expired. Please request a new one."}

    c.execute(
        "UPDATE users SET password_hash = ?, reset_token = NULL, reset_token_expires = NULL WHERE reset_token = ?",
        (hash_password(new_password), token)
    )
    conn.commit()
    conn.close()
    return {"status": "ok", "message": "Password reset successfully. You can now log in."}


# ── PROFILE ROUTES ────────────────────────────────────────────────────────────


@app.post("/profile")
async def create_profile(body: dict):
    username = body.get("username", "").strip()
    if not username:
        return {"error": "Username required"}
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT username, created_at FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    if row:
        conn.close()
        return {"status": "exists", "username": row["username"],
                "message": f"Welcome back, {username}!"}
    c.execute("INSERT INTO users (username) VALUES (?)", (username,))
    conn.commit()
    conn.close()
    return {"status": "created", "username": username, "message": f"Welcome, {username}!"}


@app.get("/profile/{username}")
async def get_profile(username: str):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "SELECT username, first_name, last_name, photo_url, created_at FROM users WHERE username = ?",
        (username,)
    )
    user = c.fetchone()
    if not user:
        conn.close()
        return {"error": "User not found"}
    c.execute("""
        SELECT COUNT(*) as total,
               ROUND(AVG(distance_cm), 1) as avg_cm,
               MAX(bar_rating) as best_rating,
               MIN(CASE WHEN bar_rating > 0 THEN bar_rating END) as worst_rating
        FROM scores WHERE username = ?
    """, (username,))
    stats = c.fetchone()
    conn.close()
    return {
        "username": user["username"],
        "first_name": user["first_name"],
        "last_name": user["last_name"],
        "photo_url": user["photo_url"],
        "created_at": user["created_at"],
        "total_pours": stats["total"] or 0,
        "avg_cm": stats["avg_cm"],
        "best_rating": stats["best_rating"],
        "worst_rating": stats["worst_rating"],
    }


@app.post("/profile/{username}/edit")
async def edit_profile(username: str, body: dict):
    first_name = body.get("first_name", "").strip()
    last_name = body.get("last_name", "").strip()
    if not first_name or not last_name:
        return {"error": "First and last name are required"}
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "UPDATE users SET first_name = ?, last_name = ? WHERE username = ?",
        (first_name, last_name, username)
    )
    conn.commit()
    conn.close()
    return {"status": "ok", "first_name": first_name, "last_name": last_name}


@app.post("/profile/{username}/photo")
async def upload_profile_photo(username: str, file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1].lower()
    if ext not in ["jpg", "jpeg", "png"]:
        return {"error": "Only jpg/png allowed"}
    img_bytes = await file.read()
    filename = f"{username}_avatar.{ext}"
    path = os.path.join(UPLOAD_DIR, filename)
    with open(path, "wb") as f:
        f.write(img_bytes)
    photo_url = f"/uploads/{filename}"
    conn = get_db()
    c = conn.cursor()
    c.execute("UPDATE users SET photo_url = ? WHERE username = ?", (photo_url, username))
    conn.commit()
    conn.close()
    return {"status": "ok", "photo_url": photo_url}


@app.get("/uploads/{filename}")
async def serve_upload(filename: str):
    path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


@app.get("/profile/{username}/pours")
async def get_profile_pours(username: str):
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT id, distance_cm, description, bar_name, bar_rating,
               fresh_photo_uri, timestamp
        FROM scores WHERE username = ?
        ORDER BY
            CASE WHEN bar_rating IS NOT NULL THEN bar_rating ELSE 0 END DESC,
            CASE WHEN distance_cm IS NOT NULL THEN distance_cm ELSE 999 END ASC
    """, (username,))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── SCORES ROUTES ─────────────────────────────────────────────────────────────


@app.post("/scores")
async def submit_score(body: dict):
    username = body.get("username")
    distance_cm = body.get("distance_cm")
    description = body.get("description", "")
    bar_name = body.get("bar_name", "Unknown Bar")
    bar_rating = body.get("bar_rating", 0)
    fresh_photo_uri = body.get("fresh_photo_uri")
    lat = body.get("lat")
    lng = body.get("lng")

    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT username FROM users WHERE username = ?", (username,))
    if not c.fetchone():
        c.execute("INSERT INTO users (username) VALUES (?)", (username,))

    c.execute("""
        INSERT INTO scores (username, distance_cm, description, bar_name,
                            bar_rating, fresh_photo_uri, lat, lng)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (username, distance_cm, description, bar_name,
          bar_rating, fresh_photo_uri, lat, lng))
    conn.commit()
    conn.close()
    return {"status": "saved"}


@app.delete("/scores/{score_id}")
async def delete_score(score_id: int, body: dict):
    username = body.get("username")
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT username FROM scores WHERE id = ?", (score_id,))
    row = c.fetchone()
    if not row or row["username"] != username:
        conn.close()
        raise HTTPException(status_code=403, detail="Not authorized")
    c.execute("DELETE FROM scores WHERE id = ?", (score_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted"}


# ── LEADERBOARD ───────────────────────────────────────────────────────────────


@app.get("/leaderboard")
async def leaderboard():
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT username,
               COUNT(*) as total_pours,
               ROUND(AVG(distance_cm), 2) as avg_cm,
               ROUND(MIN(distance_cm), 2) as best_pour
        FROM scores
        WHERE distance_cm IS NOT NULL
        GROUP BY username
        HAVING COUNT(*) >= 1
        ORDER BY avg_cm ASC
        LIMIT 50
    """)
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── BARS ──────────────────────────────────────────────────────────────────────


@app.get("/bars")
async def get_bars():
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT bar_name,
               ROUND(AVG(bar_rating), 1) as avg_rating,
               COUNT(*) as total_pours,
               COUNT(DISTINCT username) as unique_visitors,
               ROUND(AVG(distance_cm), 1) as avg_cm,
               AVG(lat) as lat,
               AVG(lng) as lng
        FROM scores
        WHERE bar_name IS NOT NULL
          AND bar_name != 'Unknown Bar'
          AND bar_rating > 0
        GROUP BY bar_name
        ORDER BY avg_rating DESC, total_pours DESC
    """)
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/bars/search")
async def search_bars(q: str = ""):
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT DISTINCT bar_name FROM scores
        WHERE bar_name LIKE ? AND bar_name != 'Unknown Bar'
        LIMIT 5
    """, (f"%{q}%",))
    rows = c.fetchall()
    conn.close()
    return [r["bar_name"] for r in rows]


# ── FRIENDS ───────────────────────────────────────────────────────────────────


@app.get("/friends/{username}")
async def get_friends(username: str):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT following FROM follows WHERE follower = ?", (username,))
    following = [r["following"] for r in c.fetchall()]
    c.execute("SELECT follower FROM follows WHERE following = ?", (username,))
    followers = [r["follower"] for r in c.fetchall()]
    conn.close()
    return {"following": following, "followers": followers}


@app.get("/friends/{username}/feed")
async def friend_feed(username: str):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT following FROM follows WHERE follower = ?", (username,))
    following = [r["following"] for r in c.fetchall()]
    if not following:
        conn.close()
        return []
    placeholders = ",".join("?" * len(following))
    c.execute(f"""
        SELECT username, distance_cm, bar_name, bar_rating,
               fresh_photo_uri, timestamp
        FROM scores
        WHERE username IN ({placeholders})
        ORDER BY timestamp DESC
        LIMIT 50
    """, following)
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/friends/{username}/search")
async def search_users(username: str, q: str = ""):
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT u.username,
               EXISTS(
                   SELECT 1 FROM follows f
                   WHERE f.follower = ? AND f.following = u.username
               ) as is_following
        FROM users u
        WHERE u.username LIKE ? AND u.username != ?
        LIMIT 10
    """, (username, f"%{q}%", username))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.post("/friends/follow")
async def follow(body: dict):
    follower = body.get("follower")
    following = body.get("following")
    if follower == following:
        return {"error": "Cannot follow yourself"}
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "INSERT OR IGNORE INTO follows (follower, following) VALUES (?, ?)",
        (follower, following)
    )
    conn.commit()
    conn.close()
    return {"status": "followed"}


@app.post("/friends/unfollow")
async def unfollow(body: dict):
    follower = body.get("follower")
    following = body.get("following")
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "DELETE FROM follows WHERE follower = ? AND following = ?",
        (follower, following)
    )
    conn.commit()
    conn.close()
    return {"status": "unfollowed"}


# ── IMAGE ANALYSIS HELPERS ────────────────────────────────────────────────────


def find_glass_roi(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    h, w = img.shape[:2]
    best = None
    best_score = 0
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect = ch / max(cw, 1)
        if aspect > 1.5 and area > (h * w * 0.05) and cw > w * 0.1:
            score = area * aspect
            if score > best_score:
                best_score = score
                best = (x, y, cw, ch)
    return best


def detect_beer_line(img, roi=None, g_bbox=None):
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 80))

    scan_x1 = int(w * 0.25)
    scan_x2 = int(w * 0.75)

    # Start scanning BELOW the G text to skip the letters
    # If we know where the G is, start 20% below it
    if g_bbox:
        g_bottom = g_bbox["y"] + g_bbox["h"]
        scan_start = g_bottom + int(h * 0.05)  # 5% padding below G
    else:
        scan_start = int(h * 0.3)  # fallback

    scan_end = int(h * 0.95)

    best_row = None
    best_transition = 0

    for y in range(scan_start, scan_end):
        row = dark_mask[y, scan_x1:scan_x2]
        dark_ratio = np.sum(row > 0) / len(row)

        row_above = dark_mask[max(0, y - 10), scan_x1:scan_x2]
        dark_above = np.sum(row_above > 0) / len(row_above)

        transition = dark_ratio - dark_above
        if transition > best_transition and dark_ratio > 0.4:
            best_transition = transition
            best_row = y

    if best_row is None:
        for y in range(scan_start, scan_end):
            row = dark_mask[y, scan_x1:scan_x2]
            if np.sum(row > 0) / len(row) > 0.5:
                best_row = y
                break

    if best_row is None:
        return None, None, None

    beer_line_pct = (1 - best_row / h) * 100
    return beer_line_pct, best_row, None

def calculate_distance_cm(beer_line_pct, g_midpoint_pct):
    GLASS_HEIGHT_CM = 16.0
    diff_pct = abs(beer_line_pct - g_midpoint_pct)
    return round((diff_pct / 100) * GLASS_HEIGHT_CM, 2)


def get_beer_line_position(beer_line_pct, g_midpoint_pct):
    if abs(beer_line_pct - g_midpoint_pct) < 1.0:
        return "at_g"
    elif beer_line_pct > g_midpoint_pct:
        return "above_g"
    else:
        return "below_g"


def build_description(distance_cm, beer_line_position, g_detected):
    if not g_detected:
        return "G logo not clearly detected — show the Guinness label for best results."
    if distance_cm == 0:
        return "Perfect split! The beer line bisects the G logo perfectly."
    pos = beer_line_position.replace("_", " ")
    if distance_cm <= 0.3:
        return f"Nearly perfect — the beer line is just {distance_cm}cm {pos}."
    elif distance_cm <= 1.0:
        return f"Good pour — {distance_cm}cm off, beer line is {pos}."
    elif distance_cm <= 2.5:
        return f"Not bad — {distance_cm}cm off, beer line is {pos}. Keep practicing!"
    else:
        return f"Needs work — {distance_cm}cm off, beer line is {pos}."

# ── G Template Matching ───────────────────────────────────────────────────────
TEMPLATE_PATH = "assets/g_template.png"
_g_template_cache = None

def get_g_template():
    global _g_template_cache
    if _g_template_cache is None:
        tmpl = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
        if tmpl is None:
            return None
        _g_template_cache = tmpl
    return _g_template_cache
    
    # Sort left-to-right, take the leftmost candidate (G is first letter)
    candidates.sort(key=lambda c: c[0])
    x, y, cw, ch = candidates[0]
    
    # Adjust y back to full image coordinates
    abs_y = search_top + y + ch // 2
    abs_x = x + cw // 2
    
    return {"x": abs_x, "y": abs_y}

def detect_guinness_g(image_bgr: np.ndarray, img_bytes: bytes) -> Optional[dict]:
    if client is None:
        return None

    h, w = image_bgr.shape[:2]

    try:
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": (
                    f"This image is {w}x{h}px. Using percentages where 0%=top-left and 100%=bottom-right: "
                    f"1. The glass has the word GUINNESS in large bold white text on its dark lower section. "
                    f"Find that text and return the center of the G (first letter, leftmost) as g_x_pct and g_y_pct. "
                    f"This G is well below the foam — it is on the dark stout, not in the white foam zone. "
                    f"Do NOT return coordinates in the foam/white area. "
                    f'Return ONLY JSON: {{"g_x_pct": <float>, "g_y_pct": <float>, "line_y_pct": <float>}}'
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]}],
        max_tokens=80,
        )
        text = response.choices[0].message.content
        print(f"GPT raw response: {text}")
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if not match:
            return None

        ai = json.loads(match.group())
        g_center_x = int(ai["g_x_pct"] / 100 * w)
        g_center_y = int(ai["g_y_pct"] / 100 * h)
        beer_line_y = int(ai["line_y_pct"] / 100 * h)

        return {
            "bbox": {"x": int(ai["g_x_pct"] / 100 * w) - 60, "y": g_center_y - 50, "w": 80, "h": 100},
            "center": {"x": g_center_x, "y": g_center_y},
            "beer_line_y": beer_line_y,
            "confidence": 1.0,
        }

    except Exception as e:
        print(f"AI G detection failed: {e}")
        return None

# ── IMAGE ANALYSIS ENDPOINT ───────────────────────────────────────────────────


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"glass_detected": False, "beer_present": False, "g_detected": False,
                "distance_cm": None, "description": "Could not read image.",
                "beer_line_position": None, "g_midpoint_pct": 35, "beer_line_pct": 50,
                "measurement_method": "opencv", "g_detection": None,
                "image_width": None, "image_height": None}

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 75))
    cream_mask = cv2.inRange(hsv, (15, 10, 160), (45, 130, 255))
    dark_ratio = np.sum(dark_mask > 0) / (h * w)
    cream_ratio = np.sum(cream_mask > 0) / (h * w)
    glass_detected = dark_ratio > 0.04
    beer_present = dark_ratio > 0.06 and cream_ratio > 0.008

    if not glass_detected:
        return {"glass_detected": False, "beer_present": False, "g_detected": False,
                "distance_cm": None, "description": "No Guinness glass detected. Make sure the glass fills most of the frame.",
                "beer_line_position": None, "g_midpoint_pct": 35, "beer_line_pct": 50,
                "measurement_method": "opencv", "g_detection": None,
                "image_width": w, "image_height": h}

    if not beer_present:
        return {"glass_detected": True, "beer_present": False, "g_detected": False,
                "distance_cm": None, "description": "Glass detected but no beer visible. Is it empty?",
                "beer_line_position": None, "g_midpoint_pct": 35, "beer_line_pct": 50,
                "measurement_method": "opencv", "g_detection": None,
                "image_width": w, "image_height": h}

    # ── Detect G via template matching ──
    g_result = detect_guinness_g(img, img_bytes)

    if g_result is None:
        return {"glass_detected": True, "beer_present": True, "g_detected": False,
                "distance_cm": None, "description": "G logo not detected. Try better lighting or a closer angle.",
                "beer_line_position": None, "g_midpoint_pct": None, "beer_line_pct": None,
                "measurement_method": "opencv", "g_detection": None,
                "image_width": w, "image_height": h}

    # ── Detect beer line ──
    beer_line_y = g_result.get("beer_line_y", int(h * 0.5))
    beer_line_pct = (1 - beer_line_y / h) * 100

    # ── Calculate split using G center from template matching ──
    g_center_y = g_result["center"]["y"]
    g_midpoint_pct = (1 - g_center_y / h) * 100

    distance_cm = calculate_distance_cm(beer_line_pct, g_midpoint_pct)
    beer_line_position = get_beer_line_position(beer_line_pct, g_midpoint_pct)
    description = build_description(distance_cm, beer_line_position, True)

    return {
        "glass_detected": True,
        "beer_present": True,
        "g_detected": True,
        "distance_cm": round(distance_cm, 2),
        "description": description,
        "beer_line_position": beer_line_position,
        "g_midpoint_pct": round(g_midpoint_pct, 1),
        "beer_line_pct": round(beer_line_pct, 1),
        "measurement_method": "opencv-template",
        "g_detection": g_result,
        "image_width": w,
        "image_height": h,
    }


# ── GLOBAL STATS ──────────────────────────────────────────────────────────────


@app.get("/stats/global")
async def global_stats():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) as total FROM scores WHERE distance_cm IS NOT NULL")
    total = c.fetchone()["total"]
    conn.close()
    return {"total_pours": total}


# ── HEALTH ────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok"}

# ── CHECK FILTER ────────────────────────────────────────────────────────────────────

@app.post("/debug-image")
async def debug_image(file: UploadFile = File(...)):
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Could not read image"}

    h, w = img.shape[:2]

    g_result = await asyncio.get_event_loop().run_in_executor(
        None, lambda: detect_guinness_g(img, img_bytes)
    )

    beer_row = int(g_result["beer_line_y"]) if g_result else int(h * 0.5)
    beer_line_pct = beer_row / h * 100

    cv2.line(img, (0, beer_row), (w, beer_row), (255, 0, 0), 2)
    cv2.putText(img, f"Beer {beer_line_pct:.1f}%", (10, beer_row - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    if g_result:
        gx = g_result["bbox"]["x"]
        gy = g_result["bbox"]["y"]
        gw = g_result["bbox"]["w"]
        gh = g_result["bbox"]["h"]
        cv2.rectangle(img, (gx, gy), (gx + gw, gy + gh), (0, 165, 255), 2)

    import time
    ts = int(time.time())
    ann_path = f"uploads/debug_annotated_{ts}.jpg"
    mask_path = f"uploads/debug_dark_mask_{ts}.jpg"
    cv2.imwrite(ann_path, img)

    gray = cv2.cvtColor(cv2.imdecode(nparr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
    _, dark_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(mask_path, dark_mask)

    return {
        "beer_line_pct": beer_line_pct,
        "g_detected": g_result is not None,
        "annotated": f"/uploads/debug_annotated_{ts}.jpg",
        "dark_mask": f"/uploads/debug_dark_mask_{ts}.jpg",
    }