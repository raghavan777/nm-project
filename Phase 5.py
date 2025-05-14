import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Expanded Traffic Data Simulation with Future Deployment Considerations
traffic_data = pd.DataFrame({
    "hour": np.arange(0, 24),
    "day_type": [random.choice(["weekday", "weekend", "holiday"]) for _ in range(24)],
    "weather": [random.choice(["clear", "rainy", "foggy", "stormy"]) for _ in range(24)],
    "event_type": [random.choice(["none", "sports", "concert", "protest", "festival"]) for _ in range(24)],
    "road_condition": [random.choice(["normal", "construction", "accident", "flooding"]) for _ in range(24)],
    "public_transport_status": [random.choice(["operational", "delayed", "strike"]) for _ in range(24)],
    "peak_hour_effect": [random.randint(0, 1) for _ in range(24)],
    "emergency_situation": [random.randint(0, 1) for _ in range(24)],
    "multilingual_support": [random.choice(["enabled", "disabled"]) for _ in range(24)],
    "traffic_volume": [random.randint(50, 500) for _ in range(24)]
})

# Convert categorical variables to numerical representation for AI model
traffic_data["day_type"] = traffic_data["day_type"].map({"weekday": 0, "weekend": 1, "holiday": 2})
traffic_data["weather"] = traffic_data["weather"].map({"clear": 0, "rainy": 1, "foggy": 2, "stormy": 3})
traffic_data["event_type"] = traffic_data["event_type"].map({"none": 0, "sports": 1, "concert": 2, "protest": 3, "festival": 4})
traffic_data["road_condition"] = traffic_data["road_condition"].map({"normal": 0, "construction": 1, "accident": 2, "flooding": 3})
traffic_data["public_transport_status"] = traffic_data["public_transport_status"].map({"operational": 0, "delayed": 1, "strike": 2})
traffic_data["multilingual_support"] = traffic_data["multilingual_support"].map({"enabled": 1, "disabled": 0})

# Train AI Model with Future Scalability Considerations
X = traffic_data[["hour", "day_type", "weather", "event_type", "road_condition", "public_transport_status", "peak_hour_effect", "emergency_situation", "multilingual_support"]]
y = traffic_data["traffic_volume"]

model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
model.fit(X, y)

# Prediction function with Future Support Expansion
def predict_congestion(hour, day_type, weather, event_type, road_condition, public_transport_status, peak_hour_effect, emergency_situation, multilingual_support):
    X_pred = pd.DataFrame([[
        hour,
        {"weekday": 0, "weekend": 1, "holiday": 2}[day_type.lower()],
        {"clear": 0, "rainy": 1, "foggy": 2, "stormy": 3}[weather.lower()],
        {"none": 0, "sports": 1, "concert": 2, "protest": 3, "festival": 4}[event_type.lower()],
        {"normal": 0, "construction": 1, "accident": 2, "flooding": 3}[road_condition.lower()],
        {"operational": 0, "delayed": 1, "strike": 2}[public_transport_status.lower()],
        peak_hour_effect,
        emergency_situation,
        multilingual_support
    ]], columns=[
        "hour", "day_type", "weather", "event_type", "road_condition",
        "public_transport_status", "peak_hour_effect", "emergency_situation", "multilingual_support"
    ])
    
    prediction = model.predict(X_pred)
    return round(prediction[0])

# Adaptive Signal Control Logic Optimized for Future Expansions
def adjust_traffic_signals(hour, day_type, weather, event_type, road_condition, public_transport_status, peak_hour_effect, emergency_situation, multilingual_support):
    congestion = predict_congestion(hour, day_type, weather, event_type, road_condition, public_transport_status, peak_hour_effect, emergency_situation, multilingual_support)

    if congestion > 450 or weather.lower() in ["rainy", "foggy", "stormy"] or emergency_situation == 1:
        return "Extend Green Light Duration (Emergency or Severe Weather Impact)"
    elif congestion < 120:
        return "Reduce Green Light Duration ⏳"
    elif day_type.lower() == "weekend" and congestion > 350:
        return "Adapt Traffic Timing for Weekend Flow"
    elif event_type.lower() in ["sports", "concert", "protest", "festival"]:
        return "Adjust Signals for Event-Based Traffic Control"
    elif road_condition.lower() in ["construction", "accident", "flooding"]:
        return "Re-route Traffic or Reduce Flow Near Blocked Roads"
    elif public_transport_status.lower() == "strike":
        return "Increase Green Light Duration for Expected Road Congestion"
    elif peak_hour_effect == 1:
        return "Extend Green Light Flow for Rush Hours"
    elif multilingual_support == 1:
        return "Enable Multilingual User Guidance for Smart City Navigation"
    else:
        return "Maintain Normal Signal Timing ⚖️"

# Example Usage
current_hour = 17
current_day_type = "weekday"
current_weather = "rainy"
current_event_type = "sports"
current_road_condition = "normal"
current_public_transport_status = "operational"
current_peak_hour_effect = 1
current_emergency_situation = 0
current_multilingual_support = 1

print(f"Predicted Traffic Volume at {current_hour}: {predict_congestion(current_hour, current_day_type, current_weather, current_event_type, current_road_condition, current_public_transport_status, current_peak_hour_effect, current_emergency_situation, current_multilingual_support)} vehicles")
print(f"Traffic Signal Adjustment: {adjust_traffic_signals(current_hour, current_day_type, current_weather, current_event_type, current_road_condition, current_public_transport_status, current_peak_hour_effect, current_emergency_situation, current_multilingual_support)}")
