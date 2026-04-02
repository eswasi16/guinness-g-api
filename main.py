from vision import analyze_image as opencv_analyze

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    image_bytes = await file.read()

    # --- Step 1: Run OpenCV pipeline ---
    cv_result = opencv_analyze(image_bytes)

    # --- Step 2: Always run GPT-4o for glass/beer/G detection ---
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

    # --- Step 3: Decide which measurement to use ---
    # Use OpenCV measurements if glass was detected with good confidence
    # Use GPT-4o as fallback or for detection flags
    use_opencv = (
        cv_result.get("glass_detected") and
        cv_result.get("g_confidence", 0) > 0.5
    )

    return {
        # Detection from GPT-4o (more reliable for these)
        "glass_detected": gpt_result.get("glass_detected", False),
        "g_detected": gpt_result.get("g_detected", False),
        "beer_present": gpt_result.get("beer_present", False),
        # Measurements: OpenCV if confident, else GPT-4o
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