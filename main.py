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
from email_validator import validate_email, EmailNotValidError
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import bcrypt
import jwt
import numpy as np
import cv2
from openai import OpenAI

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
JWT_SECRET = os.environ.get("JWT_SECRET", "changeme-use-env-var")
SMTP_EMAIL = os.environ.get("SMTP_EMAIL")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
DB = "scores.db"

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
    # Migrate existing users table if columns missing
    try:
        c.execute("ALTER TABLE users ADD COLUMN first_name TEXT")
    except Exception:
        pass
    try:
        c.execute("ALTER TABLE users ADD COLUMN last_name TEXT")
    except Exception:
        pass
    try:
        c.execute("ALTER TABLE users ADD COLUMN reset_token TEXT")
    except Exception:
        pass
    try:
        c.execute("ALTER TABLE users ADD COLUMN reset_token_expires TEXT")
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
        "message": f"Welcome, {first_name}!"
    }

@app.post("/auth/login")
async def login(body: dict):
    username = body.get("username", "").strip()
    password = body.get("password", "")

    conn = get_db()
    c = conn.cursor()
    c.execute(
        "SELECT username, password_hash, first_name, last_name FROM users WHERE username = ?",
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
        "last_name": row["last_name"]
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
        "SELECT username, first_name, last_name, created_at FROM users WHERE username = ?",
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
        "created_at": user["created_at"],
        "total_pours": stats["total"] or 0,
        "avg_cm": stats["avg_cm"],
        "best_rating": stats["best_rating"],
        "worst_rating": stats["worst_rating"],
    }

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

# ── IMAGE ANALYSIS ────────────────────────────────────────────────────────────

def find_glass_roi(img):
    """Detect the Guinness glass region using edge detection and contours."""
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
        # Glass is tall and narrow, occupies significant portion of image
        if aspect > 1.5 and area > (h * w * 0.05) and cw > w * 0.1:
            score = area * aspect
            if score > best_score:
                best_score = score
                best = (x, y, cw, ch)
    return best

def detect_beer_line(img, roi=None):
    """
    Detect the beer/foam boundary line.
    Returns the beer line Y position as a percentage from bottom (0-100).
    """
    h, w = img.shape[:2]
    if roi:
        x, y, rw, rh = roi
        region = img[y:y+rh, x:x+rw]
    else:
        region = img
        y = 0

    rh, rw = region.shape[:2]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    # Guinness dark stout mask
    dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 75))
    # Creamy foam mask
    cream_mask = cv2.inRange(hsv, (15, 10, 160), (45, 130, 255))

    dark_ratio = np.sum(dark_mask > 0) / (rh * rw)
    cream_ratio = np.sum(cream_mask > 0) / (rh * rw)

    if dark_ratio < 0.03:
        return None, dark_ratio, cream_ratio

    # Scan rows from top to bottom to find transition from foam to dark
    # Smooth the masks vertically
    dark_cols = np.sum(dark_mask > 0, axis=1).astype(float)
    cream_cols = np.sum(cream_mask > 0, axis=1).astype(float)

    # Normalize per row
    dark_pct = dark_cols / rw
    cream_pct = cream_cols / rw

    # Smooth
    kernel_size = max(3, rh // 30)
    dark_smooth = np.convolve(dark_pct, np.ones(kernel_size)/kernel_size, mode='same')
    cream_smooth = np.convolve(cream_pct, np.ones(kernel_size)/kernel_size, mode='same')

    # Find the transition: look for where foam ends and dark begins (top-down)
    beer_line_y = None
    for row in range(rh // 4, int(rh * 0.85)):
        if dark_smooth[row] > 0.25 and cream_smooth[max(0, row-kernel_size):row].mean() > 0.05:
            beer_line_y = row
            break

    # Fallback: find row with maximum cream-to-dark gradient
    if beer_line_y is None:
        diff = np.gradient(dark_smooth - cream_smooth)
        beer_line_y = int(np.argmax(diff[rh//4:int(rh*0.85)]) + rh//4)

    # Convert to absolute image coords
    abs_y = beer_line_y + y
    beer_line_pct = ((h - abs_y) / h) * 100
    return beer_line_pct, dark_ratio, cream_ratio

def detect_g_logo(img, roi=None):
    """
    Detect the Guinness G logo position using template-style color segmentation.
    The G is a golden/harp logo typically in the lower-middle of the glass.
    Returns G midpoint as percentage from bottom (0-100).
    """
    h, w = img.shape[:2]
    if roi:
        x, y, rw, rh = roi
        region = img[y:y+rh, x:x+rw]
        offset_y = y
    else:
        region = img
        offset_y = 0

    rh, rw = region.shape[:2]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    # Gold/amber harp color range
    gold_mask = cv2.inRange(hsv, (15, 80, 120), (35, 255, 255))
    # White label area
    white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))

    gold_ratio = np.sum(gold_mask > 0) / (rh * rw)
    white_ratio = np.sum(white_mask > 0) / (rh * rw)

    # Try to find the label patch (white rectangle with gold G)
    label_mask = cv2.bitwise_or(gold_mask, white_mask)
    label_mask = cv2.morphologyEx(label_mask, cv2.MORPH_CLOSE,
                                   np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    best_cnt = None
    best_score = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x2, y2, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / max(ch, 1)
        # Label is roughly square-ish, not too large, not tiny
        if 0.4 < aspect < 2.5 and (rh * rw * 0.005) < area < (rh * rw * 0.35):
            score = area
            if score > best_score:
                best_score = score
                best_cnt = cnt

    if best_cnt is not None:
        x2, y2, cw, ch = cv2.boundingRect(best_cnt)
        center_y = y2 + ch // 2 + offset_y
        g_midpoint_pct = ((h - center_y) / h) * 100
        return g_midpoint_pct, True, gold_ratio

    # Fallback: use the gold pixels centroid
    if gold_ratio > 0.005:
        gold_ys = np.where(gold_mask > 0)[0]
        if len(gold_ys) > 0:
            center_y = int(np.median(gold_ys)) + offset_y
            g_midpoint_pct = ((h - center_y) / h) * 100
            return g_midpoint_pct, True, gold_ratio

    # Default: assume G is at ~35% from bottom (standard Guinness glass)
    return 35.0, False, gold_ratio

def calculate_distance_cm(beer_line_pct, g_midpoint_pct, glass_height_px=None):
    """
    Convert the percentage gap to centimeters.
    A standard Guinness pint glass is ~16cm tall.
    1% of glass height ≈ 0.16cm.
    """
    GLASS_HEIGHT_CM = 16.0
    diff_pct = abs(beer_line_pct - g_midpoint_pct)
    distance_cm = round((diff_pct / 100) * GLASS_HEIGHT_CM, 2)
    return distance_cm

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

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {
            "glass_detected": False, "beer_present": False,
            "g_detected": False, "distance_cm": None,
            "description": "Could not read image.",
            "beer_line_position": None,
            "g_midpoint_pct": 35, "beer_line_pct": 50,
            "measurement_method": "opencv",
        }

    h, w = img.shape[:2]

    # ── Step 1: Basic glass/beer detection ──
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 75))
    cream_mask = cv2.inRange(hsv, (15, 10, 160), (45, 130, 255))
    dark_ratio = np.sum(dark_mask > 0) / (h * w)
    cream_ratio = np.sum(cream_mask > 0) / (h * w)

    glass_detected = dark_ratio > 0.04
    beer_present = dark_ratio > 0.06 and cream_ratio > 0.008

    if not glass_detected:
        return {
            "glass_detected": False, "beer_present": False,
            "g_detected": False, "distance_cm": None,
            "description": "No Guinness glass detected. Make sure the glass fills most of the frame.",
            "beer_line_position": None,
            "g_midpoint_pct": 35, "beer_line_pct": 50,
            "measurement_method": "opencv",
        }

    if not beer_present:
        return {
            "glass_detected": True, "beer_present": False,
            "g_detected": False, "distance_cm": None,
            "description": "Glass detected but no beer visible. Is it empty?",
            "beer_line_position": None,
            "g_midpoint_pct": 35, "beer_line_pct": 50,
            "measurement_method": "opencv",
        }

    # ── Step 2: Find glass ROI ──
    roi = find_glass_roi(img)

    # ── Step 3: Detect beer/foam line ──
    beer_line_pct, _, _ = detect_beer_line(img, roi)
    if beer_line_pct is None:
        beer_line_pct = 70.0  # fallback: assume beer is near top

    # ── Step 4: Detect G logo ──
    g_midpoint_pct, g_detected, gold_ratio = detect_g_logo(img, roi)

    # ── Step 5: Calculate distance ──
    distance_cm = calculate_distance_cm(beer_line_pct, g_midpoint_pct)
    beer_line_position = get_beer_line_position(beer_line_pct, g_midpoint_pct)
    description = build_description(distance_cm, beer_line_position, g_detected)

    # ── Step 6: AI fallback only if OpenCV confidence is low ──
    # Trigger AI if: G not detected AND gold ratio is very low (label not visible)
    USE_AI_FALLBACK = gold_ratio < 0.003 and not g_detected

    if USE_AI_FALLBACK:
        try:
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            prompt = """Analyze this Guinness pint glass image.
Return ONLY valid JSON:
{
  "g_detected": bool,
  "g_midpoint_pct": float,
  "beer_line_pct": float,
  "distance_cm": float,
  "beer_line_position": string
}
g_midpoint_pct and beer_line_pct are 0-100 from bottom of glass.
distance_cm is cm between beer line and G center (0 = perfect split)."""
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]
                }],
                max_tokens=150,
            )
            text = response.choices[0].message.content
            match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
            if match:
                ai = json.loads(match.group())
                g_midpoint_pct = ai.get("g_midpoint_pct", g_midpoint_pct)
                beer_line_pct = ai.get("beer_line_pct", beer_line_pct)
                distance_cm = ai.get("distance_cm", distance_cm)
                beer_line_position = ai.get("beer_line_position", beer_line_position)
                g_detected = ai.get("g_detected", g_detected)
                description = build_description(distance_cm, beer_line_position, g_detected)
                return {
                    "glass_detected": True, "beer_present": True,
                    "g_detected": g_detected, "distance_cm": distance_cm,
                    "description": description,
                    "beer_line_position": beer_line_position,
                    "g_midpoint_pct": g_midpoint_pct,
                    "beer_line_pct": beer_line_pct,
                    "measurement_method": "opencv+ai",
                }
        except Exception as e:
            print(f"AI fallback failed: {e}")

    return {
        "glass_detected": True,
        "beer_present": True,
        "g_detected": g_detected,
        "distance_cm": distance_cm,
        "description": description,
        "beer_line_position": beer_line_position,
        "g_midpoint_pct": round(g_midpoint_pct, 1),
        "beer_line_pct": round(beer_line_pct, 1),
        "measurement_method": "opencv",
    }

# ── HEALTH ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}