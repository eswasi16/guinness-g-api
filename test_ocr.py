import easyocr
import cv2
import numpy as np

reader = easyocr.Reader(['en'])  # downloads model once ~100MB

img = cv2.imread('guinness.jpg')
h, w = img.shape[:2]

results = reader.readtext(img)

print("All detected text:")
for (bbox, text, conf) in results:
    print(f"  '{text}' — conf: {conf:.2f}")

# Find GUINNESS
for (bbox, text, conf) in results:
    if 'GUINNESS' in text.upper() and conf > 0.3:
        pts = np.array(bbox, dtype=np.int32)
        center_y = int((pts[0][1] + pts[2][1]) / 2)
        print(f"\n✅ GUINNESS center_y = {center_y} ({center_y/h*100:.1f}% down image)")
        break