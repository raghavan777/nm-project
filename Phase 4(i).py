import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Generate Simulated Traffic Data
traffic_data = pd.DataFrame({
    "hour": np.arange(0, 24),
    "day_type": [random.choice(["weekday", "weekend"]) for _ in range(24)],
    "traffic_volume": [random.randint(500, 1500) for _ in range(24)]
})

# Convert 'day_type' to numerical representation
traffic_data["day_type"] = traffic_data["day_type"].map({"weekday": 0, "weekend": 1})

# Train AI Model for Traffic Prediction (Polynomial Regression for Non-Linearity)
X = traffic_data[["hour", "day_type"]]
y = traffic_data["traffic_volume"]

model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
model.fit(X, y)

# Prediction Function
def predict_congestion(hour, day_type):
    X_pred = pd.DataFrame([[hour, 0 if day_type.lower() == "weekday" else 1]], columns=["hour", "day_type"])
    prediction = model.predict(X_pred)
    return round(prediction[0])

# Adaptive Signal Control Logic
def adjust_traffic_signals(hour, day_type):
    congestion = predict_congestion(hour, day_type)
    if congestion > 400:
        if hour in range(7, 10) or hour in range(17, 20):
            return "Extend Green Light Duration 🚦"
        elif day_type.lower() == "weekend" and congestion > 300:
            return "Adapt Traffic Timing for Weekend Flow"
    return "Maintain Normal Signal Timing 🔄"

# Example Usage
current_hour = 17
current_day_type = "weekday"

print(f"Predicted Traffic Volume at {current_hour}: {predict_congestion(current_hour, current_day_type)} vehicles")
print(f"Traffic Signal Adjustment: {adjust_traffic_signals(current_hour, current_day_type)}")

