import os
import sys
import joblib
import pandas as pd
import warnings

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings("ignore")


# ============================================================
# PATH SETUP
# ============================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)

# Needed so joblib can load TargetEncoder from preprocess.py
sys.path.append(BASE_DIR)

MODEL_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
ISOLATION_MODEL_PATH = os.path.join(MODEL_DIR, "isolation_forest_pipeline.pkl")


# index.html can be inside app/ or app/templates/
if os.path.exists(os.path.join(APP_DIR, "index.html")):
    TEMPLATE_DIR = APP_DIR
elif os.path.exists(os.path.join(APP_DIR, "templates", "index.html")):
    TEMPLATE_DIR = os.path.join(APP_DIR, "templates")
else:
    raise FileNotFoundError("index.html not found in app/ or app/templates/")


# ============================================================
# FASTAPI SETUP
# ============================================================

app = FastAPI(title="TrafficPolice - Network Intrusion Detection System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=TEMPLATE_DIR)


# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 60)
print("LOADING MODELS")
print("=" * 60)

print("BASE_DIR:", BASE_DIR)
print("TEMPLATE_DIR:", TEMPLATE_DIR)
print("BEST_MODEL_PATH:", BEST_MODEL_PATH)
print("ISOLATION_MODEL_PATH:", ISOLATION_MODEL_PATH)

best_model = joblib.load(BEST_MODEL_PATH)
isolation_forest = joblib.load(ISOLATION_MODEL_PATH)

print("Models loaded successfully.")
print("=" * 60)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def render_page(request: Request, result=None, error=None):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "result": result,
            "error": error
        }
    )


def clean_input_df(df):
    """
    Drop columns not used during model training.
    The saved pipeline already contains preprocessing.
    """
    return df.drop(
        columns=["id", "attack_cat", "label"],
        errors="ignore"
    )


def binary_label(value):
    """
    0 = Normal
    1 = Attack
    """
    try:
        value = int(value)
        return "ATTACK" if value == 1 else "NORMAL"
    except Exception:
        return "N/A"


def isolation_label(value):
    """
    Isolation Forest:
    1  = Normal
    -1 = Attack / Anomaly
    """
    try:
        value = int(value)
        return "ATTACK" if value == -1 else "NORMAL"
    except Exception:
        return "ERROR"


def get_confidence(model, input_df):
    """
    Confidence = highest class probability.
    Attack probability = probability of class 1.
    """
    try:
        probabilities = model.predict_proba(input_df)
        classes = list(model.classes_)

        confidence_scores = probabilities.max(axis=1)

        if 1 in classes:
            attack_index = classes.index(1)
            attack_probabilities = probabilities[:, attack_index]
        else:
            attack_probabilities = [0.0] * len(input_df)

        return attack_probabilities, confidence_scores

    except Exception:
        return [0.0] * len(input_df), [0.0] * len(input_df)


def build_result(df, filename):
    if df.empty:
        raise ValueError("Uploaded CSV file is empty.")

    original_df = df.copy()
    input_df = clean_input_df(df)

    # Random Forest prediction
    rf_predictions = best_model.predict(input_df)
    attack_probs, confidence_scores = get_confidence(best_model, input_df)

    # Isolation Forest prediction
    iso_predictions = isolation_forest.predict(input_df)

    rows = []

    rf_attack_count = 0
    rf_normal_count = 0

    iso_attack_count = 0
    iso_normal_count = 0

    actual_attack_count = 0
    actual_normal_count = 0

    preview_limit = 50

    for i in range(len(original_df)):
        row = original_df.iloc[i]

        rf_text = binary_label(rf_predictions[i])
        iso_text = isolation_label(iso_predictions[i])

        actual_text = "N/A"
        if "label" in original_df.columns:
            actual_text = binary_label(row["label"])

            if actual_text == "ATTACK":
                actual_attack_count += 1
            elif actual_text == "NORMAL":
                actual_normal_count += 1

        if rf_text == "ATTACK":
            rf_attack_count += 1
        elif rf_text == "NORMAL":
            rf_normal_count += 1

        if iso_text == "ATTACK":
            iso_attack_count += 1
        elif iso_text == "NORMAL":
            iso_normal_count += 1

        attack_category = str(row.get("attack_cat", "-"))

        if i < preview_limit:
            rows.append({
                "index": i + 1,
                "protocol": str(row.get("proto", "-")).upper(),
                "service": str(row.get("service", "-")),
                "state": str(row.get("state", "-")),
                "duration": str(row.get("dur", 0)),
                "sbytes": f"{int(row.get('sbytes', 0)):,}" if str(row.get("sbytes", 0)).replace(".", "", 1).isdigit() else str(row.get("sbytes", 0)),
                "dbytes": f"{int(row.get('dbytes', 0)):,}" if str(row.get("dbytes", 0)).replace(".", "", 1).isdigit() else str(row.get("dbytes", 0)),

                "actual": actual_text,
                "attack_category": attack_category,

                "random_forest": rf_text,
                "isolation_forest": iso_text,

                "attack_probability": round(float(attack_probs[i]) * 100, 2),
                "confidence": round(float(confidence_scores[i]) * 100, 2)
            })

    total_rows = len(original_df)

    result = {
        "filename": filename,
        "total_rows": total_rows,
        "preview_count": len(rows),

        "rf_attack_count": rf_attack_count,
        "rf_normal_count": rf_normal_count,

        "iso_attack_count": iso_attack_count,
        "iso_normal_count": iso_normal_count,

        "actual_attack_count": actual_attack_count,
        "actual_normal_count": actual_normal_count,

        "attack_percentage": round((rf_attack_count / total_rows) * 100, 2) if total_rows else 0,
        "normal_percentage": round((rf_normal_count / total_rows) * 100, 2) if total_rows else 0,

        "rows": rows
    }

    return result


# ============================================================
# ROUTES
# ============================================================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return render_page(request)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "template_dir": TEMPLATE_DIR,
        "best_model_exists": os.path.exists(BEST_MODEL_PATH),
        "isolation_model_exists": os.path.exists(ISOLATION_MODEL_PATH)
    }


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(".csv"):
            return render_page(
                request,
                error="Please upload a CSV file only."
            )

        os.makedirs(UPLOAD_DIR, exist_ok=True)

        safe_filename = os.path.basename(file.filename)
        file_path = os.path.join(UPLOAD_DIR, safe_filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        df = pd.read_csv(file_path)

        result = build_result(df, safe_filename)

        return render_page(request, result=result)

    except Exception as e:
        return render_page(request, error=str(e))