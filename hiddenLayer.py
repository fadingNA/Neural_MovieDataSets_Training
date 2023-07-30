import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('movie_statistic_dataset.csv')

# Preprocess the data
non_numeric_columns = ['runtime_minutes', 'director_birthYear', 'director_deathYear', 'movie_averageRating',
                       'movie_numerOfVotes', 'approval_Index', 'Production budget $', 'Domestic gross $',
                       'Worldwide gross $']
df[non_numeric_columns] = df[non_numeric_columns].replace('-', pd.NA).apply(pd.to_numeric, errors='coerce')
df.dropna(subset=['director_name'], inplace=True)
# Convert production_date to numerical feature (number of days since a reference date)
reference_date = pd.to_datetime('2023-07-30')  # Replace with your desired reference date
df['production_date'] = (reference_date - pd.to_datetime(df['production_date'])).dt.days

# Handle missing values denoted by '\\N'
# Replace '\\N' and other non-numeric values with NaN
non_numeric_columns = ['runtime_minutes', 'director_birthYear', 'director_deathYear', 'movie_averageRating',
                       'movie_numerOfVotes', 'approval_Index', 'Production budget $', 'Domestic gross $',
                       'Worldwide gross $']
df[non_numeric_columns] = df[non_numeric_columns].replace({'\\N': pd.NA}).apply(pd.to_numeric)

# Drop rows with missing values in target variable 'director_name'
df.dropna(subset=['director_name'], inplace=True)

# Fill missing values in other numeric columns using mean imputation
imputer = SimpleImputer(strategy='mean')
df[non_numeric_columns] = imputer.fit_transform(df[non_numeric_columns])

# Convert genres to one-hot encoded features using NumPy
genres = df['genres'].str.get_dummies(sep=',')
df = pd.concat([df, genres], axis=1)

# Convert director_professions to one-hot encoded features using NumPy
mlb = MultiLabelBinarizer()
director_professions = pd.DataFrame(mlb.fit_transform(df['director_professions'].str.split(',')), columns=mlb.classes_)
df = pd.concat([df, director_professions], axis=1)

# Drop unnecessary columns
df.drop(['movie_title', 'genres', 'director_professions'], axis=1, inplace=True)

# Define target variable (y)
y = df['director_name']

# Drop the target variable from features (X)
X = df.drop('director_name', axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert X_train and X_test to numpy arrays for vectorized operations
X_train_np = X_train.values
X_test_np = X_test.values

# Standardize features using vectorized operations with NumPy
scaler = StandardScaler()
X_train_scaled_np = scaler.fit_transform(X_train_np)
X_test_scaled_np = scaler.transform(X_test_np)

# Create and train the logistic regression model with adjusted max_iter
logReg = LogisticRegression(max_iter=1000)
logReg.fit(X_train_scaled_np, y_train)

# Make predictions on the test set
y_pred = logReg.predict(X_test_scaled_np)

# Calculate the accuracy of the model
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy Percentage: {acc}")
