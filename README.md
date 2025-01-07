# Employee Attrition Prediction with Deep Learning

This project implements a neural network to predict employee attrition and recommend suitable departments using a dataset containing various employee-related features. The model is built with TensorFlow's Keras library and trained on preprocessed data to classify employees into different attrition and department categories.

## Project Overview
Goals
Predict whether employees are likely to leave the company.
Recommend the best department for each employee based on their features.
## Workflow
1. Data Preparation
Dataset: The dataset attrition.csv is loaded and explored for relevant features and target variables.
Features: Columns such as Age, DistanceFromHome, HourlyRate, YearsWithCurrManager, etc., are used as inputs.
Target Variables: The Attrition and Department columns serve as targets, representing employee attrition status and department recommendations.
python

Verify
Run
Copy code
# Load dataset
import pandas as pd

attrition_df = pd.read_csv('https://static.bc-edx.com/ai/ail-v-1-0/m19/lms/datasets/attrition.csv')

# Define target and features
y_df = attrition_df[["Attrition", "Department"]]
selected_columns = [
    'Age', 'DistanceFromHome', 'HourlyRate', 'YearsWithCurrManager',
    'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears',
    'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion'
]
X_df = attrition_df[selected_columns]
2. Data Preprocessing
Train-Test Split: The dataset is split into training (75%) and testing (25%) subsets using scikit-learn.
Feature Scaling: Numerical features are scaled using StandardScaler for better model performance.
Encoding: Categorical variables for Attrition and Department are encoded using OneHotEncoder.
python

Verify
Run
Copy code
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, random_state=78)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Encode categorical variables
enc_department = OneHotEncoder(sparse=False)
enc_department.fit(y_train[['Department']])
y_train_encoded_dept = enc_department.transform(y_train[['Department']])
y_test_encoded_dept = enc_department.transform(y_test[['Department']])

enc_attrition = OneHotEncoder(sparse=False)
enc_attrition.fit(y_train[['Attrition']])
y_train_encoded_attr = enc_attrition.transform(y_train[['Attrition']])
y_test_encoded_attr = enc_attrition.transform(y_test[['Attrition']])
3. Neural Network Model
Architecture:
Input layer: Matches the number of features in the dataset.
Hidden layers: Two dense layers with 64 and 32 nodes respectively, using the ReLU activation function.
Output layers: Two branches for multi-class classification using softmax for both attrition and department predictions.
Compilation:
Loss function: categorical_crossentropy (for multi-class classification).
Optimizer: adam (adaptive learning rate optimization).
Metric: Accuracy.
python

Verify
Run
Copy code
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Create the input layer
input_layer = Input(shape=(X_train_scaled.shape[1],))

# Create shared layers
shared1 = Dense(64, activation='relu')(input_layer)
shared2 = Dense(32, activation='relu')(shared1)

# Create output branch for department
output_department = Dense(enc_department.categories_[0].size, activation='softmax', name='output_department')(shared2)

# Create output branch for attrition
output_attrition = Dense(enc_attrition.categories_[0].size, activation='softmax', name='output_attrition')(shared2)

# Create the model
model = Model(inputs=input_layer, outputs=[output_attrition, output_department])

# Compile the model
model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'], optimizer='adam', metrics=['accuracy'])
4. Training and Evaluation
Training: The model is trained on the scaled training data for 50 epochs.
Evaluation: Loss and accuracy are calculated on the test dataset to evaluate model performance.
python

Verify
Run
Copy code
# Train the model
history = model.fit(X_train_scaled, [y_train_encoded_attr, y_train_encoded_dept], epochs=50, validation_data=(X_test_scaled, [y_test_encoded_attr, y_test_encoded_dept]))

# Evaluate the model

