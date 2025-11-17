import io
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
import torchvision
import torch.serialization

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# 🔥 Soil Class Names
# ----------------------------
SOIL_CLASSES = [
    "alluvial",
    "black",
    "cinder",
    "clay",
    "laterite",
    "loamy",
    "peat",
    "red",
    "sandy",
    "sandy_loam",
    "yellow"
]

# ----------------------------
# 🌱 Static Plant Recommendation Map
# ----------------------------
PLANT_RECOMMENDATIONS = {
    "alluvial": ["Wheat", "Rice", "Sugarcane", "Maize", "Pulses", "Oilseeds"],
    "black": ["Cotton", "Soybean", "Sorghum (Jowar)", "Sunflower", "Sugarcane"],
    "cinder": ["Citrus", "Grapes", "Pineapple", "Mango", "Vegetables"],
    "clay": ["Paddy", "Tomato", "Broccoli", "Cabbage", "Peas"],
    "laterite": ["Tea", "Coffee", "Rubber", "Coconut", "Cashew"],
    "loamy": ["Vegetables", "Potato", "Sugar beet", "Tomato", "Pulses"],
    "peat": ["Blueberries", "Cranberries", "Potatoes", "Root crops"],
    "red": ["Millets", "Groundnut", "Maize", "Pulses", "Tobacco"],
    "sandy": ["Watermelon", "Peanuts", "Barley", "Carrots", "Potato"],
    "sandy_loam": ["Cotton", "Wheat", "Strawberries", "Potato"],
    "yellow": ["Maize", "Peanut", "Cotton", "Pulses"]
}

# ----------------------------
# 🔥 Load Model
# ----------------------------
torch.serialization.add_safe_globals([
    torchvision.models.efficientnet.EfficientNet
])

MODEL_PATH = "soil_classifier_full_efficientnetb0.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the entire model
model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.eval()

# ----------------------------
# 🔄 EfficientNet Preprocessing
# ----------------------------
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----------------------------
# 🔮 Prediction Endpoint
# ----------------------------
@app.get("/")
def home():
    return {"message": "Soil Classification API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    # Preprocess
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)
        confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted_class].item()

    class_id = predicted_class.item()
    class_name = SOIL_CLASSES[class_id]
    recommended_plants = PLANT_RECOMMENDATIONS[class_name]

    return {
        "class_id": class_id,
        "class_name": class_name,
        "confidence": confidence,
        "recommended_plants": recommended_plants
    }
