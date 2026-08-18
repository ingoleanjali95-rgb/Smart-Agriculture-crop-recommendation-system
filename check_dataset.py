import pandas as pd

# ============================================================
# SMART AGRICULTURE CROP RECOMMENDATION SYSTEM
# DATASET CHECK
# ============================================================

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("dataset/crop_data.csv")

print("=" * 60)
print("       SMART AGRICULTURE DATASET CHECK")
print("=" * 60)


# ==========================
# First 5 Rows
# ==========================

print("\nFirst 5 Rows:")
print(df.head())


# ==========================
# Dataset Shape
# ==========================

print("\nDataset Shape:")
print(df.shape)

print("\nTotal Records:", df.shape[0])
print("Total Columns:", df.shape[1])


# ==========================
# Columns
# ==========================

print("\nColumns:")
print(df.columns.tolist())


# ==========================
# Data Types
# ==========================

print("\nData Types:")
print(df.dtypes)


# ==========================
# Missing Values
# ==========================

print("\nMissing Values:")
print(df.isnull().sum())

if df.isnull().sum().sum() == 0:
    print("✅ No Missing Values Found!")


# ==========================
# Duplicate Rows
# ==========================

print("\nDuplicate Rows:")

duplicate_rows = df.duplicated().sum()

print(duplicate_rows)

if duplicate_rows == 0:
    print("✅ No Duplicate Records Found!")


# ==========================
# Total States
# ==========================

print("\nTotal States:")
print(df["State"].nunique())

print("\nStates List:")

for state in sorted(df["State"].unique()):
    print("-", state)


# ==========================
# Total Crops
# ==========================

print("\nTotal Crops:")
print(df["Crop"].nunique())

print("\nCrop List:")

for crop in sorted(df["Crop"].unique()):
    print("-", crop)


# ==========================
# Total Seasons
# ==========================

print("\nTotal Seasons:")
print(df["Season"].nunique())

print("\nSeason List:")

for season in sorted(df["Season"].unique()):
    print("-", season)


# ==========================
# Total Soil Types
# ==========================

print("\nTotal Soil Types:")
print(df["SoilType"].nunique())

print("\nSoil Type List:")

for soil in sorted(df["SoilType"].unique()):
    print("-", soil)


# ==========================
# Crop Distribution
# ==========================

print("\n" + "=" * 60)
print("CROP DISTRIBUTION")
print("=" * 60)

print(df["Crop"].value_counts())


# ==========================
# State Distribution
# ==========================

print("\n" + "=" * 60)
print("STATE DISTRIBUTION")
print("=" * 60)

print(df["State"].value_counts())


# ==========================
# Final Summary
# ==========================

print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print("Total Records :", len(df))
print("Total Columns :", len(df.columns))
print("Total States  :", df["State"].nunique())
print("Total Crops   :", df["Crop"].nunique())
print("Total Seasons :", df["Season"].nunique())
print("Total Soils   :", df["SoilType"].nunique())

print("\n" + "=" * 60)
print("✅ DATASET CHECKED SUCCESSFULLY!")
print("=" * 60)