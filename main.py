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

def analyze_with_opencv(img_bytes: bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 80))
    cream_mask = cv2.inRange(hsv, (15, 20, 180), (40, 120, 255))
    dark_ratio = np.sum(dark_mask > 0) / (h * w)
    cream_ratio = np.sum(cream_mask > 0) / (h * w)
    glass_detected = dark_ratio > 0.03
    beer_present = dark_ratio > 0.05 and cream_ratio > 0.01
    return {
        "glass_detected": glass_detected,
        "beer_present": beer_present,
        "dark_ratio": float(dark_ratio),
        "cream_ratio": float(cream_ratio),
    }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    img_bytes = await file.read()

    opencv_result = analyze_with_opencv(img_bytes)
    if opencv_result and not opencv_result["glass_detected"]:
        return {
            "glass_detected": False,
            "beer_present": False,
            "g_detected": False,
            "distance_cm": None,
            "description": "No Guinness glass detected.",
            "beer_line_position": None,
            "g_midpoint_pct": 50,
            "beer_line_pct": 50,
            "measurement_method": "opencv",
        }

    b64 = base64.b64encode(img_bytes).decode("utf-8")

    prompt = """Analyze this image of a Guinness pint glass.

Return ONLY valid JSON with these exact fields:
{
  "glass_detected": bool,
  "beer_present": bool,
  "g_detected": bool,
  "g_midpoint_pct": float (0-100, vertical % from bottom of glass where center of G logo is),
  "beer_line_pct": float (0-100, vertical % from bottom of glass where beer/foam line is),
  "distance_cm": float (estimated cm between beer line and G midpoint, 0 if perfect),
  "beer_line_position": string (one of: "above_g", "at_g", "below_g"),
  "description": string (one sentence describing the pour quality)
}

The "Split the G" challenge: the beer/foam line should bisect the G logo perfectly.
If the beer line is exactly at the G midpoint, distance_cm = 0 (perfect split).
Be precise with measurements."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]
            }],
            max_tokens=300,
        )
        text = response.choices[0].message.content
        match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
        if match:
            result = json.loads(match.group())
            result["measurement_method"] = "gpt-4o"
            return result
    except Exception as e:
        print(f"OpenAI error: {e}")

    return {
        "glass_detected": True,
        "beer_present": True,
        "g_detected": False,
        "distance_cm": None,
        "description": "Could not analyze image. Try again with better lighting.",
        "beer_line_position": None,
        "g_midpoint_pct": 50,
        "beer_line_pct": 50,
        "measurement_method": "fallback",
    }

# ── HEALTH ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}