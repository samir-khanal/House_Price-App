from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

df = pd.read_csv('House Price Prediction Dataset.csv',index_col='Id')

def calculate_final_price(row):
    condition_multiplier = {
        "Excellent": 1.5,
        "Good": 1.2,
        "Fair": 1.0,
        "Poor": 0.8
    }
    # calculating house age in numbers to use in formula
    age = 2025 - row["YearBuilt"]
    
    if age <= 4:
        age_multiplier = 1.8  
    elif 5 <= age <= 9:
        age_multiplier = 1.5
    elif 10 <= age <= 17:
        age_multiplier = 1.2
    elif 18 <= age <= 30:
        age_multiplier = 1.0
    elif 31 <= age <= 50:
        age_multiplier = 0.8
    else:
        age_multiplier = 0.4  
    
    base_price = 1000000  # 1 floor new house base price
    final_price = base_price * row["Floors"] * condition_multiplier[row["Condition"]] * age_multiplier
    return round(final_price, 2)

df['Age'] = 2025 - df['YearBuilt']
df["ConditionMultiplier"] = df["Condition"].map({
    "Excellent": 1.5,
    "Good": 1.2,
    "Fair": 1.0,
    "Poor": 0.8
})
df["FinalPrice"] = df.apply(calculate_final_price, axis=1)
#df.head(10)
X = df[['Floors', 'Age','ConditionMultiplier']] 
#X = df[['Floors', 'YearBuilt','ConditionMultiplier']]  
Y = df['FinalPrice'] 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
# print("\nTraining data (X_train):")
# print(X_train)
# print("\nTesting data (X_train):")
# print(X_test)
linear_model = LinearRegression()
linear_model.fit(X_train, Y_train)
sample_input = [[2, 13, 0.8]]  # Example input: 2 floors, YearBuilt(Age)-2012,Condition Multiplyer-0.8
predicted_price = linear_model.predict(sample_input)
print(f"\nPredicted final price for input {sample_input}: RS {predicted_price[0]:.2f}")

y_pred = linear_model.predict(X_test)

mse = mean_squared_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)
print(f"\nMean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")
import joblib
joblib.dump(linear_model, 'house_price_prediction.pkl')
print("Model training completed and saved as 'house_price_prediction.pkl'")