import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# SMART AGRICULTURE CROP RECOMMENDATION SYSTEM
# EXPLORATORY DATA ANALYSIS
# ============================================================

print("=" * 70)
print("        SMART AGRICULTURE - EXPLORATORY DATA ANALYSIS")
print("=" * 70)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("dataset/crop_data_clean.csv")

print("\n✅ Dataset Loaded Successfully!")


# ============================================================
# 2. DATASET INFORMATION
# ============================================================

print("\n" + "=" * 50)
print("DATASET INFORMATION")
print("=" * 50)

print("\nDataset Shape:")
print(df.shape)

print("\nTotal Rows:", df.shape[0])
print("Total Columns:", df.shape[1])

print("\nColumns:")
for column in df.columns:
    print("-", column)


# ============================================================
# 3. FIRST FIVE RECORDS
# ============================================================

print("\n" + "=" * 50)
print("FIRST FIVE RECORDS")
print("=" * 50)

print(df.head())


# ============================================================
# 4. DATA TYPES
# ============================================================

print("\n" + "=" * 50)
print("DATA TYPES")
print("=" * 50)

print(df.dtypes)


# ============================================================
# 5. CROP DISTRIBUTION
# ============================================================

print("\n" + "=" * 50)
print("CROP DISTRIBUTION")
print("=" * 50)

crop_count = df["Crop"].value_counts()

print(crop_count)

plt.figure(figsize=(14, 7))

crop_count.plot(
    kind="bar"
)

plt.title(
    "Crop Distribution",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel(
    "Crop",
    fontsize=12
)

plt.ylabel(
    "Number of Records",
    fontsize=12
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    "crop_distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# 6. STATE DISTRIBUTION
# ============================================================

print("\n" + "=" * 50)
print("STATE DISTRIBUTION")
print("=" * 50)

state_count = df["State"].value_counts()

print(state_count)

plt.figure(figsize=(15, 7))

state_count.plot(
    kind="bar"
)

plt.title(
    "State-wise Dataset Distribution",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel(
    "State",
    fontsize=12
)

plt.ylabel(
    "Number of Records",
    fontsize=12
)

plt.xticks(
    rotation=90
)

plt.tight_layout()

plt.savefig(
    "state_distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# 7. SEASON DISTRIBUTION
# ============================================================

print("\n" + "=" * 50)
print("SEASON DISTRIBUTION")
print("=" * 50)

season_count = df["Season"].value_counts()

print(season_count)

plt.figure(figsize=(8, 5))

season_count.plot(
    kind="bar"
)

plt.title(
    "Season Distribution",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel(
    "Season"
)

plt.ylabel(
    "Number of Records"
)

plt.tight_layout()

plt.savefig(
    "season_distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# 8. SOIL TYPE DISTRIBUTION
# ============================================================

print("\n" + "=" * 50)
print("SOIL TYPE DISTRIBUTION")
print("=" * 50)

soil_count = df["SoilType"].value_counts()

print(soil_count)

plt.figure(figsize=(10, 6))

soil_count.plot(
    kind="bar"
)

plt.title(
    "Soil Type Distribution",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel(
    "Soil Type"
)

plt.ylabel(
    "Number of Records"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    "soil_distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# 9. NUMERICAL STATISTICAL SUMMARY
# ============================================================

print("\n" + "=" * 50)
print("STATISTICAL SUMMARY")
print("=" * 50)

print(
    df.describe()
)


# ============================================================
# 10. MISSING VALUES
# ============================================================

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)

missing_values = df.isnull().sum()

print(missing_values)

if missing_values.sum() == 0:

    print("\n✅ No Missing Values Found!")

else:

    print("\n⚠️ Missing Values Found!")


# ============================================================
# 11. DUPLICATE RECORDS
# ============================================================

print("\n" + "=" * 50)
print("DUPLICATE RECORDS")
print("=" * 50)

duplicate_count = df.duplicated().sum()

print(
    "Duplicate Rows:",
    duplicate_count
)

if duplicate_count == 0:

    print("✅ No Duplicate Records Found!")

else:

    print("⚠️ Duplicate Records Found!")


# ============================================================
# 12. UNIQUE CROPS
# ============================================================

print("\n" + "=" * 50)
print("UNIQUE CROPS")
print("=" * 50)

unique_crops = df["Crop"].nunique()

print(
    "Total Unique Crops:",
    unique_crops
)

print("\nCrop Names:")

for crop in sorted(df["Crop"].unique()):

    print("-", crop)


# ============================================================
# 13. UNIQUE STATES
# ============================================================

print("\n" + "=" * 50)
print("UNIQUE STATES")
print("=" * 50)

unique_states = df["State"].nunique()

print(
    "Total Unique States:",
    unique_states
)

print("\nState Names:")

for state in sorted(df["State"].unique()):

    print("-", state)


# ============================================================
# 14. NUMERICAL FEATURES
# ============================================================

print("\n" + "=" * 50)
print("NUMERICAL FEATURES")
print("=" * 50)

numeric_columns = df.select_dtypes(
    include="number"
).columns

print(
    list(numeric_columns)
)


# ============================================================
# 15. CORRELATION MATRIX
# ============================================================

print("\n" + "=" * 50)
print("CORRELATION ANALYSIS")
print("=" * 50)

if len(numeric_columns) > 1:

    correlation = df[numeric_columns].corr()

    print(correlation)

    plt.figure(figsize=(10, 8))

    plt.imshow(
        correlation,
        interpolation="nearest"
    )

    plt.title(
        "Correlation Matrix",
        fontsize=18,
        fontweight="bold"
    )

    plt.colorbar()

    plt.xticks(
        range(len(numeric_columns)),
        numeric_columns,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(numeric_columns)),
        numeric_columns
    )

    plt.tight_layout()

    plt.savefig(
        "correlation_matrix.png",
        dpi=300
    )

    plt.show()


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("        ✅ EDA COMPLETED SUCCESSFULLY!")
print("=" * 70)

print("\nDataset Summary:")

print("Total Records :", len(df))
print("Total Columns :", len(df.columns))
print("Unique Crops  :", df["Crop"].nunique())
print("Unique States :", df["State"].nunique())
print("Missing Values:", df.isnull().sum().sum())
print("Duplicate Rows:", df.duplicated().sum())

print("\n📊 EDA Graphs Generated Successfully!")
print("=" * 70)