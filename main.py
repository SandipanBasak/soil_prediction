import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU-only mode for Render

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import io

# ✅ Allow VGG model class for safe deserialization
torch.serialization.add_safe_globals([models.vgg.VGG])

# ----------------------------------------------------
# 1️⃣ FastAPI App Setup
# ----------------------------------------------------
app = FastAPI(title="🌱 Soil Classification API (Full PyTorch Model)", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# 2️⃣ Model Loading (Full Model)
# ----------------------------------------------------
MODEL_PATH = "soil_classifier_full_vgg16.pth.pth"
model = None

def get_model():
    """Load full PyTorch model safely."""
    global model
    if model is None:
        print("⏳ Loading full PyTorch model (.pth)...")
        model = torch.load(
            MODEL_PATH,
            map_location=torch.device("cpu"),
            weights_only=False  # ✅ Required for full model loading
        )
        model.eval()
        print("✅ Full model loaded successfully.")
    return model

# ----------------------------------------------------
# 3️⃣ Soil Classes
# ----------------------------------------------------
CLASS_NAMES = [
    "alluvial", "black", "cinder", "clay", "laterite", "loamy",
    "peat", "red", "sandy", "sandy_loam", "yellow"
]

# ----------------------------------------------------
# 4️⃣ Soil → Crop Mapping
# ----------------------------------------------------
SOIL_TO_CROPS = {
    "alluvial": ["Rice", "Wheat", "Sugarcane", "Maize", "Jute", "Pulses", "Oilseeds"],
    "black": ["Cotton", "Soybean", "Sunflower", "Sorghum", "Wheat", "Citrus", "Groundnut"],
    "cinder": ["Tapioca", "Cashew", "Coconut", "Arecanut", "Pineapple"],
    "clay": ["Rice", "Sugarcane", "Jute", "Paddy", "Vegetables"],
    "laterite": ["Tea", "Coffee", "Cashew", "Rubber", "Coconut"],
    "loamy": ["Sugarcane", "Cotton", "Wheat", "Pulses", "Oilseeds", "Potato", "Vegetables"],
    "peat": ["Rice", "Jute", "Sugarcane", "Vegetables"],
    "red": ["Groundnut", "Millets", "Potato", "Rice", "Wheat", "Pulses"],
    "sandy": ["Peanuts", "Watermelon", "Potatoes", "Carrots", "Cabbage"],
    "sandy_loam": ["Groundnut", "Potato", "Maize", "Tomato", "Onion", "Melons"],
    "yellow": ["Maize", "Pulses", "Peas", "Groundnut", "Fruits", "Oilseeds"]
}

# ----------------------------------------------------
# 5️⃣ Image Transform
# ----------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ----------------------------------------------------
# 6️⃣ Routes
# ----------------------------------------------------
@app.get("/")
def home():
    return {"message": "✅ Soil Classification API (Full Model) is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        model_instance = get_model()

        # Read and preprocess image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)

        # Predict
        with torch.no_grad():
            outputs = model_instance(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            pred_index = torch.argmax(probabilities).item()
            confidence = probabilities[pred_index].item()

        pred_class = CLASS_NAMES[pred_index]
        crops = SOIL_TO_CROPS.get(pred_class, ["No data available"])

        return JSONResponse({
            "predicted_class": pred_class,
            "confidence": round(float(confidence), 3),
            "recommended_crops": crops
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

