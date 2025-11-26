from flask import Flask, render_template, request
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)


csv_path = "crop_yield.csv"
df = pd.read_csv(csv_path)

bool_cols = ["Fertilizer_Used", "Irrigation_Used"]
for col in bool_cols:
    if df[col].dtype == object:
        df[col] = df[col].map({"True": 1, "False": 0})
    else:
        df[col] = df[col].astype(int)

cat_cols = ["Region", "Soil_Type", "Crop", "Weather_Condition"]
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

target_yield = "Yield_tons_per_hectare"
target_fert = "Fertilizer_Used"
target_irrig = "Irrigation_Used"

yield_features = df_encoded.drop(columns=[target_yield]).columns
fert_features = df_encoded.drop(columns=[target_yield, target_fert]).columns
irrig_features = df_encoded.drop(columns=[target_yield, target_irrig]).columns


regions = sorted(df["Region"].unique())
soils = sorted(df["Soil_Type"].unique())
crops = sorted(df["Crop"].unique())
weathers = sorted(df["Weather_Condition"].unique())


yield_model = load_model("yield_model.h5",compile=False)
fert_model = load_model("fertilizer_model.h5",compile=False)
irrig_model = load_model("irrigation_model.h5",compile=False,)

scaler_yield = joblib.load("scaler_yield.pkl")
scaler_fert = joblib.load("scaler_fertilizer.pkl")
scaler_irrig = joblib.load("scaler_irrigation.pkl")


def preprocess_input(form):
    region = form.get("Region")
    soil = form.get("Soil_Type")
    crop = form.get("Crop")
    weather = form.get("Weather_Condition")

    rainfall = float(form.get("Rainfall_mm"))
    temp = float(form.get("Temperature_Celsius"))
    days = float(form.get("Days_to_Harvest"))

    fert_used = form.get("Fertilizer_Used")  
    irr_used = form.get("Irrigation_Used")   

    fert_val = 1 if fert_used == "yes" else 0
    irr_val = 1 if irr_used == "yes" else 0

    base_dict = {
        "Region": region,
        "Soil_Type": soil,
        "Crop": crop,
        "Rainfall_mm": rainfall,
        "Temperature_Celsius": temp,
        "Fertilizer_Used": fert_val,
        "Irrigation_Used": irr_val,
        "Weather_Condition": weather,
        "Days_to_Harvest": days,
        "Yield_tons_per_hectare": 0.0,
    }

    input_df = pd.DataFrame([base_dict])

  
    input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)

    
    X_yield = input_encoded.reindex(columns=yield_features, fill_value=0)
    X_fert = input_encoded.reindex(columns=fert_features, fill_value=0)
    X_irrig = input_encoded.reindex(columns=irrig_features, fill_value=0)

    return X_yield, X_fert, X_irrig


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        X_yield, X_fert, X_irrig = preprocess_input(request.form)

        
        X_yield_scaled = scaler_yield.transform(X_yield)
        X_fert_scaled = scaler_fert.transform(X_fert)
        X_irrig_scaled = scaler_irrig.transform(X_irrig)

        yield_pred = float(yield_model.predict(X_yield_scaled)[0][0])
        fert_prob = float(fert_model.predict(X_fert_scaled)[0][0])
        irrig_prob = float(irrig_model.predict(X_irrig_scaled)[0][0])

        fert_label = "Recommended" if fert_prob >= 0.5 else "Not Recommended"
        irrig_label = "Required" if irrig_prob >= 0.5 else "Not Required"

        result = {
            "yield_pred": round(yield_pred, 3),
            "fert_prob": round(fert_prob, 3),
            "fert_label": fert_label,
            "irrig_prob": round(irrig_prob, 3),
            "irrig_label": irrig_label,
        }

    return render_template(
        "index.html",
        regions=regions,
        soils=soils,
        crops=crops,
        weathers=weathers,
        result=result,
    )

if __name__ == "__main__":
    app.run(debug=True)
