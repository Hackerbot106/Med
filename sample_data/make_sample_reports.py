"""
Generates a couple of SYNTHETIC lab-report-style images (plain PIL drawing,
no real data) used to demo the OCR pipeline. Run once from seed.py.
"""
import os
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def _font(size=22):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_report(filename, lines, size=(900, 500)):
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    font_title = _font(26)
    font_body = _font(22)
    draw.text((30, 20), "SYNTHETIC DEMO LAB SLIP - NOT A REAL PATIENT RECORD", fill="red", font=_font(18))
    draw.line((30, 55, size[0] - 30, 55), fill="black", width=2)
    y = 80
    for i, line in enumerate(lines):
        f = font_title if i == 0 else font_body
        draw.text((30, y), line, fill="black", font=f)
        y += 40
    path = os.path.join(OUT_DIR, filename)
    img.save(path)
    return path


def generate_all():
    make_report(
        "sample_lab_report_fever.png",
        [
            "Community Health Centre - Vitals & Lab Slip (SYNTHETIC)",
            "Patient ID: DEMO-0001 (synthetic)",
            "Temp: 103.2 F",
            "Pulse / HR: 118",
            "SpO2: 91 %",
            "BP: 100/62 mmHg",
            "RR: 26",
            "Notes: Patient reports fever x 3 days, dry cough.",
        ],
    )
    make_report(
        "sample_lab_report_routine.png",
        [
            "Primary Health Centre - Vitals Slip (SYNTHETIC)",
            "Patient ID: DEMO-0002 (synthetic)",
            "Temp: 99.1 F",
            "Pulse / HR: 82",
            "SpO2: 98 %",
            "BP: 118/76 mmHg",
            "RR: 16",
            "Notes: Routine occupational health screening, asymptomatic.",
        ],
    )


if __name__ == "__main__":
    generate_all()
    print("Sample synthetic lab report images generated in", OUT_DIR)
