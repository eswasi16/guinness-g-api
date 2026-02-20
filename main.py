from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import openai, base64, json, os

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/")
def root():
    return {"status": "Guinness G API is running"}

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
                    Analyze this Guinness pint glass photo.
                    Find the 'G' in GUINNESS on the glass and measure
                    how close the stout/foam boundary is to the exact
                    middle of the G.
                    Respond with JSON only:
                    {
                      "glass_detected": true,
                      "g_detected": true,
                      "distance_from_center_pct": 0,
                      "beer_line_position": "above_center or below_center or perfect",
                      "description": "brief result here"
                    }
                """},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}"
                }}
            ]
        }],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)
