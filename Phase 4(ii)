import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Generate Simulated Traffic Data with Additional Features
traffic_data = pd.DataFrame({
    "hour": np.arange(0, 24),
    "day_type": [random.choice(["weekday", "weekend"]) for _ in range(24)],
    "weather": [random.choice(["clear", "rainy", "foggy"]) for _ in range(24)],
    "traffic_volume": [random.randint(500, 1500) for _ in range(24)]
})

# Convert categorical variables to numerical representation
traffic_data["day_type"] = traffic_data["day_type"].map({"weekday": 0, "weekend": 1})
traffic_data["weather"] = traffic_data["weather"].map({"clear": 0, "rainy": 1, "foggy": 2})

# Train AI Model with Expanded Features and Polynomial Regression
X = traffic_data[["hour", "day_type", "weather"]]
y = traffic_data["traffic_volume"]

model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
model.fit(X, y)

# Prediction Function
def predict_congestion(hour, day_type, weather):
    X_pred = pd.DataFrame([[hour, 0 if day_type.lower() == "weekday" else 1,
                            {"clear": 0, "rainy": 1, "foggy": 2}[weather.lower()]]],
                          columns=["hour", "day_type", "weather"])
    prediction = model.predict(X_pred)
    return round(prediction[0])

# Adaptive Signal Control Logic with More Dynamic Adjustments
def adjust_traffic_signals(hour, day_type, weather):
    congestion = predict_congestion(hour, day_type, weather)

    if congestion > 450 or weather.lower() in ["rainy", "foggy"]:
        return "Extend Green Light Duration 🔁 (Weather & Heavy Traffic Impact)"
    elif congestion < 120:
        return "Reduce Green Light Duration ⏱"
    elif day_type.lower() == "weekend" and congestion > 350:
        return "Adapt Traffic Timing for Weekend Flow"
    else:
        return "Maintain Normal Signal Timing 🔄"

# Example Usage
current_hour = 17
current_day_type = "weekday"
current_weather = "rainy"

print(f"Predicted Traffic Volume at {current_hour}: {predict_congestion(current_hour, current_day_type, current_weather)} vehicles")
print(f"Traffic Signal Adjustment: {adjust_traffic_signals(current_hour, current_day_type, current_weather)}")
