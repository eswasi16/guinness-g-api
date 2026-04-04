# test_vision.py — put this in your guinness-g-api folder
from vision import analyze_image

with open("test_pint.jpeg", "rb") as f:
    result = analyze_image(f.read())

print(result)