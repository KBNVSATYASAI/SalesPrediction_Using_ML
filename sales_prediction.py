import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Data Preparation ---
# Create a simulated dataset for demonstration
data = {'TV': [230.1, 44.5, 17.2, 151.5, 180.8, 8.6, 55.6, 120.2, 8.7, 199.8],
        'Radio': [37.8, 39.3, 45.9, 41.3, 10.8, 4.8, 29.3, 19.6, 2.1, 2.6],
        'Newspaper': [69.2, 45.1, 69.3, 58.5, 58.4, 75.3, 23.5, 11.6, 1.0, 21.2],
        'Sales': [22.1, 10.4, 9.3, 18.5, 12.9, 7.2, 16.9, 11.2, 4.8, 10.6]}
df = pd.DataFrame(data)
print("--- Initial Data ---")
print(df.head())
print("\n--- Data Info ---")
df.info()

# Define features (X) and target (y)
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# --- 2. Model Training and Evaluation ---
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance    
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")
print(f"\nModel Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_:.2f}")

# --- 3. Prediction on New Data ---
# Predict sales for a hypothetical new budget
new_ad_spend = pd.DataFrame({'TV': [250], 'Radio': [20], 'Newspaper': [50]})
predicted_sales = model.predict(new_ad_spend)
print(f"\nPredicted sales for the new ad spend: {predicted_sales[0]:.2f} million")

# --- 4. Visualization (for a single feature, e.g., TV) ---
# This visualization shows the linear relationship
plt.figure(figsize=(10, 6))
plt.scatter(df['TV'], df['Sales'], color='blue', label='Actual Sales Data')
# Plot the regression line for TV alone
X_tv = np.array(df['TV']).reshape(-1, 1)
model_tv = LinearRegression()
model_tv.fit(X_tv, df['Sales'])
plt.plot(df['TV'], model_tv.predict(X_tv), color='red', linewidth=2, label='Regression Line')
plt.title('Linear Regression: TV Ad Spend vs. Sales')
plt.xlabel('TV Ad Spend (in thousands $)')
plt.ylabel('Sales (in millions $)')
plt.legend()
plt.grid(True)
plt.show()
