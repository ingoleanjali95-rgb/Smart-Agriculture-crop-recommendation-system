import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# SMART AGRICULTURE CROP RECOMMENDATION SYSTEM
# MACHINE LEARNING MODEL TRAINING
# ============================================================


# ============================================================
# 1. LOAD CLEAN DATASET
# ============================================================

df = pd.read_csv("dataset/crop_data_clean.csv")

print("=" * 60)
print("       SMART AGRICULTURE - MODEL TRAINING")
print("=" * 60)

print("\nDataset Shape:", df.shape)

print("\nTotal Crops:", df["Crop"].nunique())

print("Total States:", df["State"].nunique())


# ============================================================
# 2. LABEL ENCODING
# ============================================================

le_state = LabelEncoder()
le_season = LabelEncoder()
le_soil = LabelEncoder()
le_crop = LabelEncoder()


df["State"] = le_state.fit_transform(df["State"])

df["Season"] = le_season.fit_transform(df["Season"])

df["SoilType"] = le_soil.fit_transform(df["SoilType"])

df["Crop"] = le_crop.fit_transform(df["Crop"])


print("\n✅ Label Encoding Completed")


# ============================================================
# 3. FEATURES AND TARGET
# ============================================================

X = df.drop("Crop", axis=1)

y = df["Crop"]


print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print("Crop")


# ============================================================
# 4. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTraining Records:", len(X_train))

print("Testing Records:", len(X_test))


# ============================================================
# 5. RANDOM FOREST CLASSIFIER
# ============================================================

model = RandomForestClassifier(

    n_estimators=200,

    random_state=42,

    max_depth=None,

    min_samples_split=2,

    n_jobs=-1

)


# ============================================================
# 6. TRAIN MODEL
# ============================================================

print("\nTraining Random Forest Model...")

model.fit(
    X_train,
    y_train
)

print("✅ Model Training Completed")


# ============================================================
# 7. MODEL PREDICTION
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 8. MODEL ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(
    f"\nModel Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# 9. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=le_crop.classes_,
        zero_division=0
    )
)


# ============================================================
# 10. SAVE RANDOM FOREST MODEL
# ============================================================

with open(
    "crop_model.pkl",
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )


# ============================================================
# 11. SAVE STATE ENCODER
# ============================================================

with open(
    "state_encoder.pkl",
    "wb"
) as file:

    pickle.dump(
        le_state,
        file
    )


# ============================================================
# 12. SAVE SEASON ENCODER
# ============================================================

with open(
    "season_encoder.pkl",
    "wb"
) as file:

    pickle.dump(
        le_season,
        file
    )


# ============================================================
# 13. SAVE SOIL ENCODER
# ============================================================

with open(
    "soil_encoder.pkl",
    "wb"
) as file:

    pickle.dump(
        le_soil,
        file
    )


# ============================================================
# 14. SAVE CROP ENCODER
# ============================================================

with open(
    "crop_encoder.pkl",
    "wb"
) as file:

    pickle.dump(
        le_crop,
        file
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("✅ MODEL SAVED SUCCESSFULLY!")
print("=" * 60)

print("\nGenerated Files:")

print("✅ crop_model.pkl")
print("✅ state_encoder.pkl")
print("✅ season_encoder.pkl")
print("✅ soil_encoder.pkl")
print("✅ crop_encoder.pkl")

print("\n🎉 SMART AGRICULTURE ML MODEL READY!")
print("=" * 60)