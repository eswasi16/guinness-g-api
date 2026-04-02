from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from vision import analyze_image as opencv_analyze
import json
import base64
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

ANALYZE_PROMPT = """
Analyze this image of a Guinness pint glass and return a JSON object with these fields:
- glass_detected (bool): is there a Guinness pint glass in the image?
- g_detected (bool): is the Guinness G logo visible on the glass?
- beer_present (bool): is there beer/stout in the glass?
- g_midpoint_pct (float): vertical position of the center of the G logo as a percentage from the bottom of the glass (0=bottom, 100=top)
- beer_line_pct (float): vertical position of the beer/foam line as a percentage from the bottom of the glass
- distance_cm (float): estimated distance in cm between the beer line and the G midpoint
- distance_in_g_heights (float): distance expressed in multiples of the G logo height
- beer_line_position (str): one of "above_center", "below_center", or "perfect"
- description (str): a one sentence description of the pour quality
Return only valid JSON.
"""

@app.get("/")
def root():
    return {"status": "ok", "message": "Guinness G API is running"}

@app.get("/leaderboard")
async def leaderboard():
    try:
        import sqlite3
        conn = sqlite3.connect("scores.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT username,
                   COUNT(*) as total_pours,
                   ROUND(AVG(distance_cm), 2) as avg_cm,
                   ROUND(MIN(distance_cm), 2) as best_pour
            FROM scores
            WHERE distance_cm IS NOT NULL
            GROUP BY username
            ORDER BY avg_cm ASC
            LIMIT 50
        """)
        rows = cursor.fetchall()
        conn.close()
        return [
            {"username": r[0], "total_pours": r[1], "avg_cm": r[2], "best_pour": r[3]}
            for r in rows
        ]
    except Exception as e:
        return []

@app.get("/bars")
async def get_bars():
    try:
        import sqlite3
        conn = sqlite3.connect("scores.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT bar_name,
                   ROUND(AVG(bar_rating), 2) as avg_rating,
                   COUNT(*) as total_pours,
                   COUNT(DISTINCT username) as unique_visitors,
                   ROUND(AVG(distance_cm), 2) as avg_cm,
                   AVG(lat) as lat,
                   AVG(lng) as lng
            FROM scores
            WHERE bar_name IS NOT NULL AND bar_name != 'Unknown Bar' AND bar_rating > 0
            GROUP BY bar_name
            ORDER BY avg_rating DESC
        """)
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "bar_name": r[0], "avg_rating": r[1], "total_pours": r[2],
                "unique_visitors": r[3], "avg_cm": r[4], "lat": r[5], "lng": r[6]
            }
            for r in rows
        ]
    except Exception as e:
        return []

@app.get("/bars/search")
async def search_bars(q: str = ""):
    try:
        import sqlite3
        conn = sqlite3.connect("scores.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT bar_name FROM scores
            WHERE bar_name LIKE ? AND bar_name != 'Unknown Bar'
            LIMIT 5
        """, (f"%{q}%",))
        rows = cursor.fetchall()
        conn.close()
        return [r[0] for r in rows]
    except Exception as e:
        return []

@app.post("/profile")
async def create_or_get_profile(body: dict):
    try:
        import sqlite3
        username = body.get("username", "").strip()
        if not username:
            return {"error": "Username is required"}
        conn = sqlite3.connect("scores.db")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
        existing = cursor.fetchone()
        if existing:
            conn.close()
            return {"username": username, "status": "existing", "message": f"Welcome back, {username}!"}
        cursor.execute("INSERT INTO users (username) VALUES (?)", (username,))
        conn.commit()
        conn.close()
        return {"username": username, "status": "created", "message": f"Account created for {username}!"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/profile/{username}")
async def get_profile(username: str):
    try:
        import sqlite3
        conn = sqlite3.connect("scores.db")
        cursor = conn.cursor()
        cursor.execute("SELECT created_at FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        if not user:
            conn.close()
            return {"error": "User not found"}
        cursor.execute("""
            SELECT COUNT(*) as total_pours,
                   ROUND(AVG(distance_cm), 2) as avg_cm,
                   MAX(bar_rating) as best_rating,
                   MIN(CASE WHEN bar_rating > 0 THEN bar_rating END) as worst_rating
            FROM scores WHERE username = ?
        """, (username,))
        stats = cursor.fetchone()
        conn.close()
        return {
            "username": username,
            "created_at": user[0],
            "total_pours": stats[0],
            "avg_cm": stats[1],
            "best_rating": stats[2],
            "worst_rating": stats[3]
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/profile/{username}/pours")
async def get_profile_pours(username: str):
    try:
        import sqlite3
        conn = sqlite3.connect("scores.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, distance_cm, bar_name, bar_rating, fresh_photo_uri, timestamp
            FROM scores
            WHERE username = ?
            ORDER BY
                CASE WHEN bar_rating IS NULL OR bar_rating = 0 THEN 1 ELSE 0 END ASC,
                bar_rating DESC,
                distance_cm ASC
        """, (username,))
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "id": r[0], "distance_cm": r[1], "bar_name": r[2],
                "bar_rating": r[3], "fresh_photo_uri": r[4], "timestamp": r[5]
            }
            for r in rows
        ]
    except Exception as e:
        return []

@app.post("/scores")
async def submit_score(body: dict):
    try:
        import sqlite3
        conn = sqlite3.connect("scores.db")
        cursor = conn.cursor()
        cursor.execute("""
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
        cursor.execute("""
            INSERT INTO scores (username, distance_cm, description, bar_name, bar_rating, fresh_photo_uri, lat, lng)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            body.get("username"), body.get("distance_cm"), body.get("description"),
            body.get("bar_name"), body.get("bar_rating"), body.get("fresh_photo_uri"),
            body.get("lat"), body.get("lng")
        ))
        conn.commit()
        conn.close()
        return {"status": "ok"}
    except Exception as e:
        return {"error": str(e)}

@app.delete("/scores/{score_id}")
async def delete_score(score_id: int, body: dict):
    try:
        import sqlite3
        username = body.get("username")
        conn = sqlite3.connect("scores.db")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM scores WHERE id = ? AND username = ?", (score_id, username))
        conn.commit()
        conn.close()
        return {"status": "ok"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/friends/{username}")
async def get_friends(username: str):
    try:
        import sqlite3
        conn = sqlite3.connect("scores.db")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS follows (
                follower TEXT, following TEXT,
                PRIMARY KEY (follower, following)
            )
        """)
        cursor.execute("SELECT following FROM follows WHERE follower = ?", (username,))
        following = [r[0] for r in cursor.fetchall()]
        cursor.execute("SELECT follower FROM follows WHERE following = ?", (username,))
        followers = [r[0] for r in cursor.fetchall()]
        conn.close()
        return {"following": following, "followers": followers}
    except Exception as e:
        return {"following": [], "followers": []}

@app.get("/friends/{username}/feed")
async def get_friend_feed(username: str):
    try:
        import sqlite3
        conn = sqlite3.connect("scores.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.username, s.distance_cm, s.bar_name, s.bar_rating, s.fresh_photo_uri, s.timestamp
            FROM scores s
            JOIN follows f ON s.username = f.following
            WHERE f.follower = ?
            ORDER BY s.timestamp DESC
            LIMIT 50
        """, (username,))
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "username": r[0], "distance_cm": r[1], "bar_name": r[2],
                "bar_rating": r[3], "fresh_photo_uri": r[4], "timestamp": r[5]
            }
            for r in rows
        ]
    except Exception as e:
        return []

@app.get("/friends/{username}/search")
async def search_users(username: str, q: str = ""):
    try:
        import sqlite3
        conn = sqlite3.connect("scores.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT u.username,
                   CASE WHEN f.following IS NOT NULL THEN 1 ELSE 0 END as is_following
            FROM users u
            LEFT JOIN follows f ON f.follower = ? AND f.following = u.username
            WHERE u.username LIKE ? AND u.username != ?
            LIMIT 10
        """, (username, f"%{q}%", username))
        rows = cursor.fetchall()
        conn.close()
        return [{"username": r[0], "is_following": r[1]} for r in rows]
    except Exception as e:
        return []

@app.post("/friends/follow")
async def follow_user(body: dict):
    try:
        import sqlite3
        follower = body.get("follower")
        following = body.get("following")
        conn = sqlite3.connect("scores.db")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS follows (
                follower TEXT, following TEXT,
                PRIMARY KEY (follower, following)
            )
        """)
        cursor.execute(
            "INSERT OR IGNORE INTO follows (follower, following) VALUES (?, ?)",
            (follower, following)
        )
        conn.commit()
        conn.close()
        return {"status": "ok"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/friends/unfollow")
async def unfollow_user(body: dict):
    try:
        import sqlite3
        follower = body.get("follower")
        following = body.get("following")
        conn = sqlite3.connect("scores.db")
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM follows WHERE follower = ? AND following = ?",
            (follower, following)
        )
        conn.commit()
        conn.close()
        return {"status": "ok"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    image_bytes = await file.read()

    # Step 1: Run OpenCV pipeline
    cv_result = opencv_analyze(image_bytes)

    # Step 2: Run GPT-4o for detection
    img_b64 = base64.b64encode(image_bytes).decode()
    gpt_response = client.chat.completions.create(
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
    gpt_result = json.loads(gpt_response.choices[0].message.content)

    # Step 3: Use OpenCV if confident, else GPT-4o
    use_opencv = (
        cv_result.get("glass_detected") and
        cv_result.get("g_confidence", 0) > 0.5
    )

    return {
        "glass_detected": gpt_result.get("glass_detected", False),
        "g_detected": gpt_result.get("g_detected", False),
        "beer_present": gpt_result.get("beer_present", False),
        "distance_cm": cv_result["distance_cm"] if use_opencv else gpt_result.get("distance_cm", 0),
        "distance_in_g_heights": gpt_result.get("distance_in_g_heights", 0),
        "g_midpoint_pct": cv_result["g_midpoint_pct"] if use_opencv else gpt_result.get("g_midpoint_pct", 50),
        "beer_line_pct": cv_result["beer_line_pct"] if use_opencv else gpt_result.get("beer_line_pct", 50),
        "beer_line_position": cv_result["beer_line_position"] if use_opencv else gpt_result.get("beer_line_position", "unknown"),
        "description": gpt_result.get("description", ""),
        "measurement_method": "opencv+homography" if use_opencv else "gpt4o",
        "g_confidence": cv_result.get("g_confidence", 0),
        "homography_applied": cv_result.get("homography_applied", False),
    }