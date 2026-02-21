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
    # Migrations for existing DBs
    for col in ['bar_name TEXT', 'bar_rating INTEGER', 'fresh_photo_uri TEXT']:
        try:
            conn.execute(f"ALTER TABLE scores ADD COLUMN {col}")
        except:
            pass
    conn.commit()
    conn.close()

init_db()

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
                {"type": "text", "text": """
                    You are analyzing a photo of a Guinness pint glass.

                    FIRST check all three of these conditions:
                    1. Is there a Guinness pint glass visible?
                    2. Is there dark stout with a white foam head present?
                    3. Is the GUINNESS logo visible with the letter G?

                    If ANY of these are missing, set the appropriate field
                    to false and return a helpful description.

                    IF all conditions are met:
                    1. Estimate the total visible height of the pint glass
                       in the image in pixels
                    2. A standard Guinness pint glass is 16.5cm tall
                    3. Use this ratio to convert pixel distances to centimeters
                    4. Find the letter G in the GUINNESS logo
                    5. Find the exact vertical midpoint of the G
                    6. Find where the stout/foam boundary line is
                    7. Calculate the real-world distance in cm between the
                       boundary and the G midpoint using the glass height ratio
                    8. Also express this as a fraction of the G height

                    distance_cm rules:
                    - 0.0 = beer line perfectly bisects the middle of the G
                    - positive value = beer line is off center by that many cm

                    beer_line_position rules:
                    - "above_center" = beer line is above the G midpoint
                    - "below_center" = beer line is below the G midpoint
                    - "perfect" = beer line is exactly at the G midpoint

                    g_midpoint_pct and beer_line_pct:
                    - expressed as percentage from the BOTTOM of the glass
                    - 0 = bottom, 100 =