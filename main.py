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
    You are analyzing a photo of a Guinness pint glass.
    
    FIRST, determine if the glass actually contains Guinness stout.
    Guinness has a very distinct look: dark black/brown beer with a 
    thick creamy white foam head on top. There must be a clear visible 
    boundary between the dark stout and white foam.
    
    If the glass is empty, not a Guinness, or you cannot clearly see 
    a stout/foam boundary, set glass_detected or g_detected to false.
    
    ONLY if you can clearly see:
    1. A Guinness pint glass with the GUINNESS logo
    2. Dark stout in the glass with a white foam head
    3. A clear boundary between the stout and foam
    
    THEN find the letter 'G' in the GUINNESS logo and measure how close 
    the stout/foam boundary is to the exact vertical midpoint of that 'G'.
    
    distance_from_center_pct rules:
    - 0 = beer line perfectly bisects the middle of the G
    - 1-15 = very close to center of G
    - 15-50 = somewhat off
    - 50-100 = far off or G not visible
    
    Respond with JSON only:
    {
      "glass_detected": true/false,
      "g_detected": true/false,
      "beer_present": true/false,
      "distance_from_center_pct": <0-100>,
      "beer_line_position": "above_center or below_center or perfect",
      "description": "<brief human-readable result>"
    }
    
    If the glass is empty, return beer_present: false and 
    glass_detected: false with a helpful description.
"""},

                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}"
                }}
            ]
        }],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)
