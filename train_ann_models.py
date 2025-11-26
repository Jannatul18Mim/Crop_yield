

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


df = pd.read_csv("crop_yield.csv")

bool_cols = ["Fertilizer_Used", "Irrigation_Used"]
for col in bool_cols:
    if df[col].dtype == object:
        df[col] = df[col].map({"True": 1, "False": 0})
    else:
        df[col] = df[col].astype(int)


cat_cols = ["Region", "Soil_Type", "Crop", "Weather_Condition"]

df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

print("Columns after encoding:\n", df_encoded.columns)


def build_regression_model(input_dim: int):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(1, activation='linear') 
    ])
    model.compile(optimizer="adam",
                  loss="mse",
                  metrics=["mae"])
    return model


def build_binary_classification_model(input_dim: int):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  
    ])
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


es = EarlyStopping(monitor="val_loss",
                   patience=10,
                   restore_best_weights=True)



target_yield = "Yield_tons_per_hectare"

X_yield = df_encoded.drop(columns=[target_yield])
y_yield = df_encoded[target_yield].values

Xy_train, Xy_test, yy_train, yy_test = train_test_split(
    X_yield, y_yield, test_size=0.2, random_state=42
)

scaler_yield = StandardScaler()
Xy_train_scaled = scaler_yield.fit_transform(Xy_train)
Xy_test_scaled = scaler_yield.transform(Xy_test)

model_yield = build_regression_model(Xy_train_scaled.shape[1])

history_yield = model_yield.fit(
    Xy_train_scaled, yy_train,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[es],
    verbose=1
)

yield_loss, yield_mae = model_yield.evaluate(Xy_test_scaled, yy_test, verbose=0)
print(f"\n[Yield model] Test MAE: {yield_mae:.4f}")

model_yield.save("yield_model.h5")
joblib.dump(scaler_yield, "scaler_yield.pkl")



target_fert = "Fertilizer_Used"

X_fert = df_encoded.drop(columns=[target_fert, target_yield])
y_fert = df_encoded[target_fert].values

Xf_train, Xf_test, yf_train, yf_test = train_test_split(
    X_fert, y_fert, test_size=0.2, random_state=42
)

scaler_fert = StandardScaler()
Xf_train_scaled = scaler_fert.fit_transform(Xf_train)
Xf_test_scaled = scaler_fert.transform(Xf_test)

model_fert = build_binary_classification_model(Xf_train_scaled.shape[1])

history_fert = model_fert.fit(
    Xf_train_scaled, yf_train,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[es],
    verbose=1
)

fert_loss, fert_acc = model_fert.evaluate(Xf_test_scaled, yf_test, verbose=0)
print(f"\n[Fertilizer model] Test accuracy: {fert_acc:.4f}")

model_fert.save("fertilizer_model.h5")
joblib.dump(scaler_fert, "scaler_fertilizer.pkl")



target_irrig = "Irrigation_Used"

X_irrig = df_encoded.drop(columns=[target_irrig, target_yield])
y_irrig = df_encoded[target_irrig].values

Xi_train, Xi_test, yi_train, yi_test = train_test_split(
    X_irrig, y_irrig, test_size=0.2, random_state=42
)

scaler_irrig = StandardScaler()
Xi_train_scaled = scaler_irrig.fit_transform(Xi_train)
Xi_test_scaled = scaler_irrig.transform(Xi_test)

model_irrig = build_binary_classification_model(Xi_train_scaled.shape[1])

history_irrig = model_irrig.fit(
    Xi_train_scaled, yi_train,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[es],
    verbose=1
)

irrig_loss, irrig_acc = model_irrig.evaluate(Xi_test_scaled, yi_test, verbose=0)
print(f"\n[Irrigation model] Test accuracy: {irrig_acc:.4f}")

model_irrig.save("irrigation_model.h5")
joblib.dump(scaler_irrig, "scaler_irrigation.pkl")

print("\nAll three models trained and saved successfully.")
