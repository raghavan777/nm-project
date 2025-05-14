import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LinearRegression

# Simulated Traffic Data (Example: Cars passing an intersection)
traffic_data = pd.DataFrame({
    "hour": np.arange(0, 24),  # Hours of the day
    "traffic_volume": [random.randint(50, 500) for _ in range(24)]  # Simulated vehicle count
})

# Train AI Model for Traffic Prediction
X = traffic_data[["hour"]]
y = traffic_data["traffic_volume"]

model = LinearRegression()
model.fit(X, y)

# Predict Traffic Congestion for a Given Hour
def predict_congestion(hour):
    X_pred = pd.DataFrame([[hour]], columns=["hour"])
    prediction = model.predict(X_pred)
    return round(prediction[0])

# Adaptive Signal Control Function
def adjust_traffic_signals(hour):
    congestion = predict_congestion(hour)
    if congestion > 400:
        return "Increase Green Light Duration 🚦"
    elif congestion < 150:
        return "Reduce Green Light Duration ⏳"
    else:
        return "Maintain Normal Signal Timing 🔄"

# Example Usage:
current_hour = 17  # 5 PM, peak traffic time
print(f"Predicted Traffic Volume at {current_hour}: {predict_congestion(current_hour)} vehicles")
print(f"Traffic Signal Adjustment: {adjust_traffic_signals(current_hour)}")
print("Prediction Successful !!!")
