# views.py
# -*- coding: utf-8 -*-

from django.shortcuts import render
from django.http import HttpResponse
from django.core.exceptions import ValidationError

from PIL import Image
import numpy as np
import pandas as pd
import os

from tensorflow.keras.models import load_model

# --------------------------------------------------
# CONFIGURATION (CHANGE PATHS IF NEEDED)
# --------------------------------------------------
MODEL_PATH = r"C:\Users\kondu\Music\Pill_FRONTEND\mobilenetv2_final.h5"
PILLS_CSV = r"C:\Users\kondu\Music\Pill_FRONTEND\pills_description.csv"
LABELS_TXT = r"C:\Users\kondu\Music\Pill_FRONTEND\labels.txt"

IMAGE_SIZE = (126, 126)

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def canonicalize_name(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

# --------------------------------------------------
# LOAD CLASS NAMES FROM Labels.txt (CRITICAL FIX)
# --------------------------------------------------
def load_class_names_from_labels_txt():
    if not os.path.exists(LABELS_TXT):
        raise RuntimeError("Labels.txt not found")

    class_names = []

    with open(LABELS_TXT, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                class_names.append(line)

    if len(class_names) == 0:
        raise RuntimeError("No classes found in Labels.txt")

    return np.array(class_names)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model = None
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print("⚠️ Model loading failed:", e)

# --------------------------------------------------
# LOAD PILLS DESCRIPTION DATA
# --------------------------------------------------
pills_df = pd.read_csv(PILLS_CSV)

# normalize column names (IMPORTANT)
pills_df.columns = pills_df.columns.str.strip().str.lower()

# create canonical column for matching
pills_df["name_canonical"] = pills_df["medicine_name"].apply(canonicalize_name)

# --------------------------------------------------
# LOAD CLASS NAMES
# --------------------------------------------------
class_names = load_class_names_from_labels_txt()

# --------------------------------------------------
# IMAGE UTILITIES
# --------------------------------------------------
def validate_image(img_file):
    try:
        Image.open(img_file).verify()
        img_file.seek(0)
    except Exception:
        raise ValidationError("Invalid image file")

def preprocess_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# --------------------------------------------------
# VIEWS
# --------------------------------------------------
def home(request):
    return render(request, "index.html")

def input(request):
    return render(request, "input.html")

def output(request):
    if request.method != "POST":
        return HttpResponse("Invalid request method")

    if "file" not in request.FILES:
        return HttpResponse("No image uploaded")

    if model is None:
        return HttpResponse("Model not loaded on server")

    img = request.FILES["file"]

    try:
        validate_image(img)
        processed_img = preprocess_image(img)
        preds = model.predict(processed_img)
    except Exception as e:
        return HttpResponse(f"Prediction error: {e}")

    idx = int(np.argmax(preds, axis=1)[0])

    if idx >= len(class_names):
        return HttpResponse("Prediction index out of range")

    predicted_label = class_names[idx]
    medicine_display = predicted_label

    medicine_canonical = canonicalize_name(predicted_label)

    # --------------------------------------------------
    # PILLS DESCRIPTION LOOKUP
    # --------------------------------------------------
    row = pills_df[pills_df["name_canonical"] == medicine_canonical]

    if not row.empty:
        row = row.iloc[0]
        drug_class = row.get("drug_class", "N/A")
        primary_use = row.get("primary_use", "N/A")
        description = row.get("description", "N/A")
    else:
        drug_class = primary_use = description = "N/A"

    context = {
        "medicine_name": medicine_display,
        "drug_class": drug_class,
        "primary_use": primary_use,
        "description": description,
        "predicted_class": predicted_label,
    }

    return render(request, "output.html", context)