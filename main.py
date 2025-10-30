# data cleaning


# Step 0. Import libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler

# load data
df = pd.read_csv(r"C:\Users\laura\Documents\University\3rd semester\Machine Learning\Final Project\Project_description_and_data\claims_train.csv")
print(df.info())
print(df.head())

# categorical columns
cat_cols = ['Area', 'VehBrand', 'VehGas', 'Region']
df[cat_cols] = df[cat_cols].astype('category')

# missing values
print("\nMissing values per column:")
print(df.isnull().sum())
# df['VehBrand'].fillna('Unknown', inplace=True)
# df['VehAge'].fillna(df['VehAge'].median(), inplace=True)

# invalid or unrealistic values
df = df[(df['Exposure'] > 0) & (df['Exposure'] <= 1)]
df = df[(df['DrivAge'] >= 18) & (df['DrivAge'] <= 100)]
df = df[df['VehAge'] >= 0]
df = df[df['BonusMalus'] > 0]

# target variable
df['ClaimFrequency'] = df['ClaimNb'] / df['Exposure']
df['ClaimFrequency'].fillna(0, inplace=True)

# encode categoricals
df = pd.get_dummies(df, columns=['Area', 'VehGas', 'VehBrand', 'Region'], drop_first=True)

# scale numeric features
num_cols = ['VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density']
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# check results
print("\nData cleaning complete!")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist()[:10], "...")
print(df.describe())

# save cleaned data
df.to_csv("claims_cleaned.csv", index=False)
print("\nSaved as 'claims_cleaned.csv'")



