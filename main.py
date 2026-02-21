from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import openai
import base64
import json
import os
import sqlite3
from datetime import datetime

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Database Setup ---
def get_db():
    conn = sqlite3.connect("leaderboard.db")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            distance_cm REAL,
            description TEXT,
            bar_name TEXT,
            bar_rating INTEGER,
            fresh_photo_uri TEXT,
            timestamp TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    for col in ["bar_name TEXT", "bar_rating INTEGER", "fresh_photo_uri TEXT"]:
        try:
            conn.execute(f"ALTER TABLE scores ADD COLUMN {col}")
        except:
            pass
    conn.commit()
    conn.close()

init_db()

# --- Analyze Prompt ---
ANALYZE_PROMPT = (
    "You are analyzing a photo of a Guinness pint glass. "
    "FIRST check all three of these conditions: "
    "1. Is there a Guinness pint glass visible? "
    "2. Is there dark stout with a white foam head present? "
    "3. Is the GUINNESS logo visible with the letter G? "
    "If ANY of these are missing, set the appropriate field to false and return a helpful description. "
    "IF all conditions are met: "
    "1. Estimate the total visible height of the pint glass in pixels. "
    "2. A standard Guinness pint glass is 16.5cm tall. "
    "3. Use this ratio to convert pixel distances to centimeters. "
    "4. Find the letter G in the GUINNESS logo. "
    "5. Find the exact vertical midpoint of the G. "
    "6. Find where the stout/foam boundary line is. "
    "7. Calculate the real-world distance in cm between the boundary and the G midpoint. "
    "8. Also express this as a fraction of the G height. "
    "distance_cm rules: 0.0 means beer line perfectly bisects the middle of the G. "
    "beer_line_position rules: above_center means beer line is above the G midpoint, "
    "below_center means beer line is below the G midpoint, "
    "perfect means beer line is exactly at the G midpoint. "
    "g_midpoint_pct and beer_line_pct: expressed as percentage from the BOTTOM of the glass. "
    "0 = bottom of glass, 100 = top of glass. "
    "Respond with JSON only, no extra text: "
    '{"glass_detected": true or false, '
    '"g_detected": true or false, '
    '"beer_present": true or false, '
    '"distance_cm": 0.0, '
    '"distance_in_g_heights": 0.0, '
    '"g_midpoint_pct": 0.0, '
    '"beer_line_pct": 0.0, '
    '"beer_line_position": "above_center or below_center or perfect", '
    '"description": "brief human readable result"}'
)

# --- Root ---
@app.get("/")
def root():
    return {"status": "Split the G API is running"}

# --- Analyze ---
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_b64 = base64.b64encode(image_bytes).decode()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": ANALYZE_PROMPT},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}"
                }}
            ]
        }],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# --- Save Score ---
@app.post("/scores")
async def save_score(payload: dict):
    username = payload.get("username", "Anonymous")
    distance_cm = payload.get("distance_cm")
    description = payload.get("description", "")
    bar_name = payload.get("bar_name", "Unknown Bar")
    bar_rating = payload.get("bar_rating", None)
    fresh_photo_uri = payload.get("fresh_photo_uri", None)

    conn = get_db()
    conn.execute(
        """INSERT INTO scores
           (username, distance_cm, description, bar_name,
            bar_rating, fresh_photo_uri, timestamp)
           VALUES (?,?,?,?,?,?,?)""",
        (username, distance_cm, description, bar_name,
         bar_rating, fresh_photo_uri, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    return {"status": "saved"}

# --- Leaderboard ---
@app.get("/leaderboard")
def get_leaderboard():
    conn = get_db()
    rows = conn.execute("""
        SELECT
            username,
            ROUND(AVG(distance_cm), 2) as avg_cm,
            COUNT(*) as total_pours,
            ROUND(MIN(distance_cm), 2) as best_pour
        FROM scores
        WHERE distance_cm IS NOT NULL
        GROUP BY username
        ORDER BY avg_cm ASC
        LIMIT 20
    """).fetchall()
    conn.close()
    return [dict(row) for row in rows]

# --- Bars ---
@app.get("/bars")
def get_bars():
    conn = get_db()
    rows = conn.execute("""
        SELECT
            bar_name,
            ROUND(AVG(distance_cm), 2) as avg_cm,
            ROUND(AVG(bar_rating), 1) as avg_rating,
            COUNT(*) as total_pours,
            COUNT(DISTINCT username) as unique_visitors
        FROM scores
        WHERE bar_name IS NOT NULL AND bar_name != 'Unknown Bar'
        GROUP BY bar_name
        ORDER BY avg_rating DESC, avg_cm ASC
        LIMIT 50
    """).fetchall()
    conn.close()
    return [dict(row) for row in rows]

@app.get("/bars/{bar_name}")
def get_bar_detail(bar_name: str):
    conn = get_db()
    rows = conn.execute("""
        SELECT username, distance_cm, bar_rating, description, timestamp
        FROM scores WHERE bar_name=?
        ORDER BY timestamp DESC LIMIT 20
    """, (bar_name,)).fetchall()
    conn.close()
    return [dict(row) for row in rows]

# --- Create or Get Profile ---
@app.post("/profile")
async def create_profile(payload: dict):
    username = payload.get("username", "").strip()
    if not username:
        return {"error": "Username is required"}

    conn = get_db()
    existing = conn.execute(
        "SELECT * FROM profiles WHERE username=?", (username,)
    ).fetchone()

    if existing:
        conn.close()
        return {"status": "exists", "username": username,
                "message": f"Welcome back, {username}!"}

    conn.execute(
        "INSERT INTO profiles (username, created_at) VALUES (?,?)",
        (username, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    return {"status": "created", "username": username,
            "message": f"Profile created for {username}!"}

# --- Get Profile Stats ---
@app.get("/profile/{username}")
def get_profile(username: str):
    conn = get_db()
    profile = conn.execute(
        "SELECT * FROM profiles WHERE username=?", (username,)
    ).fetchone()

    if not profile:
        conn.close()
        return {"e
