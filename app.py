from flask import Flask, render_template, request
import pandas as pd
import pickle


# ==========================
# Flask App
# ==========================

app = Flask(__name__)


# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("dataset/crop_data_clean.csv")


# ==========================
# Load Model
# ==========================

with open("crop_model.pkl", "rb") as f:
    model = pickle.load(f)


# ==========================
# Load Label Encoders
# ==========================

with open("label_encoder.pkl", "rb") as f:
    encoders = pickle.load(f)

print(encoders.keys())


# ==========================
# Dynamic Dropdown Values
# ==========================

states = sorted(df["State"].unique())
seasons = sorted(df["Season"].unique())
soils = sorted(df["SoilType"].unique())


# ==========================
# Crop Details
# ==========================

crop_details = {

    "Bajra": {
        "fertilizer": "NPK + Urea",
        "water": "350-500 mm",
        "temperature": "25-35°C",
        "season": "Kharif",
        "description": "Nutritious millet crop."
    },

    "Barley": {
        "fertilizer": "DAP + Urea",
        "water": "300-500 mm",
        "temperature": "12-25°C",
        "season": "Rabi",
        "description": "Cereal crop used for food."
    },

    "Brinjal": {
        "fertilizer": "Organic Compost + NPK",
        "water": "600-800 mm",
        "temperature": "22-30°C",
        "season": "Kharif",
        "description": "Popular vegetable crop."
    },

    "Cabbage": {
        "fertilizer": "NPK",
        "water": "500-700 mm",
        "temperature": "15-25°C",
        "season": "Rabi",
        "description": "Leafy vegetable crop."
    },

    "Carrot": {
        "fertilizer": "Compost + NPK",
        "water": "350-500 mm",
        "temperature": "15-25°C",
        "season": "Rabi",
        "description": "Root vegetable rich in Vitamin A."
    },

    "Cauliflower": {
        "fertilizer": "NPK",
        "water": "500-700 mm",
        "temperature": "15-20°C",
        "season": "Rabi",
        "description": "Winter vegetable crop."
    },

    "Chilli": {
        "fertilizer": "Urea + Potash",
        "water": "600-800 mm",
        "temperature": "20-30°C",
        "season": "Kharif",
        "description": "Spice crop."
    },

    "Coriander": {
        "fertilizer": "Organic Compost",
        "water": "400-600 mm",
        "temperature": "18-28°C",
        "season": "Rabi",
        "description": "Herb used in cooking."
    },

    "Cotton": {
        "fertilizer": "NPK + Zinc",
        "water": "700-1300 mm",
        "temperature": "21-30°C",
        "season": "Kharif",
        "description": "Major fibre crop."
    },

    "Cucumber": {
        "fertilizer": "Organic Compost",
        "water": "500-700 mm",
        "temperature": "22-30°C",
        "season": "Summer",
        "description": "Refreshing vegetable crop."
    },

    "Gram": {
        "fertilizer": "DAP + Urea",
        "water": "400-600 mm",
        "temperature": "20-25°C",
        "season": "Rabi",
        "description": "Protein-rich pulse crop."
    },

    "Groundnut": {
        "fertilizer": "Gypsum + NPK",
        "water": "500-700 mm",
        "temperature": "25-30°C",
        "season": "Kharif",
        "description": "Important oilseed crop."
    },

    "Jowar": {
        "fertilizer": "NPK",
        "water": "400-600 mm",
        "temperature": "25-32°C",
        "season": "Kharif",
        "description": "Healthy cereal crop."
    },

    "Maize": {
        "fertilizer": "Urea + DAP",
        "water": "500-800 mm",
        "temperature": "18-27°C",
        "season": "Kharif",
        "description": "Food and fodder crop."
    },

    "Muskmelon": {
        "fertilizer": "Organic Compost",
        "water": "400-600 mm",
        "temperature": "25-35°C",
        "season": "Summer",
        "description": "Sweet fruit crop."
    },

    "Mustard": {
        "fertilizer": "NPK + Sulphur",
        "water": "350-450 mm",
        "temperature": "10-25°C",
        "season": "Rabi",
        "description": "Oilseed crop."
    },

    "Onion": {
        "fertilizer": "NPK + Organic Compost",
        "water": "350-550 mm",
        "temperature": "15-30°C",
        "season": "Rabi",
        "description": "Popular vegetable crop."
    },

    "Peas": {
        "fertilizer": "DAP + Potash",
        "water": "400-500 mm",
        "temperature": "10-25°C",
        "season": "Rabi",
        "description": "Protein-rich pulse crop."
    },

    "Potato": {
        "fertilizer": "NPK + Compost",
        "water": "500-700 mm",
        "temperature": "15-25°C",
        "season": "Rabi",
        "description": "Underground tuber crop."
    },

    "Ragi": {
        "fertilizer": "NPK",
        "water": "400-600 mm",
        "temperature": "20-30°C",
        "season": "Kharif",
        "description": "Nutritious millet crop."
    },

    "Rice": {
        "fertilizer": "Urea + DAP + Potash",
        "water": "1200-1500 mm",
        "temperature": "20-35°C",
        "season": "Kharif",
        "description": "Staple food crop."
    },

    "Sesame": {
        "fertilizer": "NPK + Organic Compost",
        "water": "400-600 mm",
        "temperature": "25-35°C",
        "season": "Kharif",
        "description": "Oilseed crop."
    },

    "Soybean": {
        "fertilizer": "DAP + Potash",
        "water": "450-700 mm",
        "temperature": "20-30°C",
        "season": "Kharif",
        "description": "Protein-rich oilseed crop."
    },

    "Spinach": {
        "fertilizer": "Organic Compost + NPK",
        "water": "300-500 mm",
        "temperature": "15-25°C",
        "season": "Winter",
        "description": "Leafy green vegetable."
    },

    "Sugarcane": {
        "fertilizer": "NPK + Organic Compost",
        "water": "1500-2500 mm",
        "temperature": "20-38°C",
        "season": "Annual",
        "description": "Main source of sugar."
    },

    "Sunflower": {
        "fertilizer": "NPK + Boron",
        "water": "500-700 mm",
        "temperature": "20-30°C",
        "season": "Kharif",
        "description": "Oilseed crop."
    },

    "Tomato": {
        "fertilizer": "Organic Compost + NPK",
        "water": "400-600 mm",
        "temperature": "20-30°C",
        "season": "Winter",
        "description": "Popular vegetable crop."
    },

    "Tur": {
        "fertilizer": "DAP + Potash",
        "water": "600-800 mm",
        "temperature": "25-35°C",
        "season": "Kharif",
        "description": "Pulse crop."
    },

    "Watermelon": {
        "fertilizer": "Organic Compost + NPK",
        "water": "400-600 mm",
        "temperature": "24-35°C",
        "season": "Summer",
        "description": "Juicy fruit crop."
    },

    "Wheat": {
        "fertilizer": "Urea + NPK",
        "water": "450-650 mm",
        "temperature": "10-25°C",
        "season": "Rabi",
        "description": "Major cereal crop."
    }
}


# ==========================
# Home / Welcome Page
# ==========================

@app.route("/")
def home():
    return render_template("welcome.html")


# ==========================
# About Page
# ==========================

@app.route("/about")
def about():
    return render_template("about.html")


# ==========================
# Recommendation Form
# ==========================

@app.route("/recommendation")
def recommendation():

    states = sorted(df["State"].unique())
    seasons = sorted(df["Season"].unique())
    soils = sorted(df["SoilType"].unique())

    return render_template(
        "index.html",
        states=states,
        seasons=seasons,
        soils=soils
    )


# ==========================
# Prediction
# ==========================

@app.route("/predict", methods=["POST"])
def predict():

    state = request.form["state"]
    season = request.form["season"]
    soil = request.form["soil"]

    N = int(request.form["N"])
    P = int(request.form["P"])
    K = int(request.form["K"])

    temperature = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    ph = float(request.form["ph"])
    rainfall = float(request.form["rainfall"])


    # ==========================
    # Encode Categorical Values
    # ==========================

    state = encoders["state"].transform([state])[0]
    season = encoders["season"].transform([season])[0]
    soil = encoders["soiltype"].transform([soil])[0]


    # ==========================
    # Input Data
    # ==========================

    input_data = pd.DataFrame([[
        state,
        season,
        soil,
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall
    ]], columns=[
        "State",
        "Season",
        "SoilType",
        "N",
        "P",
        "K",
        "Temperature",
        "Humidity",
        "pH",
        "Rainfall"
    ])


    # ==========================
    # Predict Crop
    # ==========================

    prediction = model.predict(input_data)


    # ==========================
    # Decode Prediction
    # ==========================

    crop = encoders["crop"].inverse_transform(prediction)[0]


    # ==========================
    # Get Crop Details
    # ==========================

    details = crop_details.get(crop, {
        "fertilizer": "Not Available",
        "water": "Not Available",
        "temperature": "Not Available",
        "season": "Not Available",
        "description": "Information not available."
    })


    # ==========================
    # Crop Image
    # ==========================

    image = crop.lower().replace(" ", "_") + ".jpg"


    # ==========================
    # Result Page
    # ==========================

    return render_template(
        "result.html",
        crop=crop,
        image=image,
        details=details
    )


# ==========================
# Crop Details Page
# ==========================

@app.route("/crop-details/<crop>")
def crop_details_page(crop):

    details = crop_details.get(crop)

    if details is None:
        return "Crop details not found", 404

    image = crop.lower().replace(" ", "_") + ".jpg"

    return render_template(
        "crop_details.html",
        crop=crop,
        image=image,
        details=details
    )


# ==========================
# Dashboard
# ==========================

@app.route("/dashboard")
def dashboard():

    crop_count = df["Crop"].value_counts()

    print("Crops:", crop_count.index.tolist())
    print("Counts:", crop_count.values.tolist())

    return render_template(
        "dashboard.html",
        crops=crop_count.index.tolist(),
        counts=crop_count.values.tolist()
    )


# ==========================
# Run App
# ==========================

if __name__ == "__main__":
    app.run(debug=True)