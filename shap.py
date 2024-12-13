import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error  # Added MAE
import matplotlib.pyplot as plt
import shap

# Load data
data_file = 'yourinputfile.csv'
data = pd.read_csv(data_file, header=0)

# Define features and labels
feature_data = data[['t2m', 'cbh', 'cape', 'cvh', 'kx', 'lai_hv', 'sp']]
label_data = data['num']

# Define feature types
continuous_features = ['t2m', 'cbh', 'cape', 'cvh', 'kx', 'lai_hv', 'sp']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), continuous_features)
    ])

# Create Random Forest Regression model
rf_regressor = RandomForestRegressor(random_state=42)

# Create pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', rf_regressor)
])

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    feature_data, label_data, test_size=0.2, random_state=40
)

# Fit model
pipeline.fit(X_train, y_train)

# Predictions
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

# Output predictions for test set
predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})
print(predictions)

# Calculate MSE
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

# Calculate R^2
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Calculate MAE
train_mae = mean_absolute_error(y_train, y_train_pred)  # Calculate MAE for training set
test_mae = mean_absolute_error(y_test, y_test_pred)     # Calculate MAE for test set

# Print results
print(f"Training set Mean Squared Error (MSE): {train_mse:.4f}")
print(f"Test set Mean Squared Error (MSE): {test_mse:.4f}")
print(f"Training set Coefficient of Determination (R^2): {train_r2:.4f}")
print(f"Test set Coefficient of Determination (R^2): {test_r2:.4f}")
print(f"Training set Mean Absolute Error (MAE): {train_mae:.4f}")  # Print MAE for training set
print(f"Test set Mean Absolute Error (MAE): {test_mae:.4f}")       # Print MAE for test set

# SHAP analysis
explainer = shap.TreeExplainer(pipeline.named_steps['regressor'])
shap_values = explainer.shap_values(X_train)

shap.summary_plot(shap_values, X_train, plot_type="bar")
plt.show()

shap.summary_plot(shap_values, X_train)
plt.show()

shap_interaction_values = explainer.shap_interaction_values(X_train.iloc[:2000,:])
shap.summary_plot(shap_interaction_values, X_train.iloc[:2000,:])
plt.show()

random_picks = np.arange(0, len(X_test), 50)
S = X_test.iloc[random_picks]

def save_shap_plot(j, filename):
    shap_values_Model = explainer.shap_values(S)
    p = shap.force_plot(explainer.expected_value, shap_values_Model[j], S.iloc[[j]])
    shap.save_html(filename, p)

for j, index in enumerate(random_picks):
    save_shap_plot(j, f'shap_force_plot_{index}.html')
    print(f"SHAP force plot for sample {index} has been saved to shap_force_plot_{index}.html")