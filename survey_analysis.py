# ------------------------------------------
# Task 4: Data Cleaning and Insight Generation
# ------------------------------------------

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------
# Load Dataset
# ------------------------------------------

df = pd.read_csv("survey.csv")

print("First 5 Rows of Dataset")
print(df.head())

print("\nDataset Information")
print(df.info())

# ------------------------------------------
# Check Missing Values
# ------------------------------------------

print("\nMissing Values")
print(df.isnull().sum())

# ------------------------------------------
# Remove Duplicate Rows
# ------------------------------------------

df = df.drop_duplicates()

# ------------------------------------------
# Fill Missing Values
# ------------------------------------------

for column in df.columns:
    if df[column].dtype == "object":
        df[column].fillna(df[column].mode()[0], inplace=True)
    else:
        df[column].fillna(df[column].mean(), inplace=True)

print("\nMissing Values After Cleaning")
print(df.isnull().sum())

# ------------------------------------------
# Encode Categorical Variables
# ------------------------------------------

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

for column in df.select_dtypes(include='object').columns:
    df[column] = encoder.fit_transform(df[column].astype(str))

print("\nDataset After Encoding")
print(df.head())

# ------------------------------------------
# Summary Statistics
# ------------------------------------------

print("\nSummary Statistics")
print(df.describe())

# ------------------------------------------
# Visualization
# ------------------------------------------

sns.set_style("whitegrid")

# 1 Age Distribution
if "Age" in df.columns:
    plt.figure()
    sns.histplot(df["Age"], kde=True)
    plt.title("Age Distribution of Respondents")
    plt.show()

# 2 Gender Distribution
if "Gender" in df.columns:
    plt.figure()
    sns.countplot(x="Gender", data=df)
    plt.title("Gender Distribution")
    plt.show()

# 3 Country Distribution
if "Country" in df.columns:
    plt.figure(figsize=(10,5))
    sns.countplot(y="Country", data=df,
                  order=df["Country"].value_counts().index[:10])
    plt.title("Top Countries of Respondents")
    plt.show()

# 4 Correlation Heatmap
plt.figure(figsize=(10,6))
corr = df.corr()

sns.heatmap(corr, annot=False, cmap="coolwarm")

plt.title("Dataset Correlation Heatmap")
plt.show()

# ------------------------------------------
# Insights
# ------------------------------------------

print("\nTop 5 Insights:")

print("1. The dataset was cleaned by removing duplicates and filling missing values.")
print("2. Categorical variables were converted into numerical form using label encoding.")
print("3. Age distribution shows the common age group of respondents.")
print("4. Gender distribution reveals participation differences.")
print("5. Country analysis shows where most respondents are from.")

print("\nTask 4 Completed Successfully")