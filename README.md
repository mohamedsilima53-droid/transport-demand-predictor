# 🚌 Public Transport Demand Predictor

> A Machine Learning app that predicts passenger demand (per hour) on public transport routes across Tanzania.

---

## 📌 Project Overview

This project predicts **how many passengers** will use a given public transport route per hour based on time, weather, route type, infrastructure, and other contextual factors. It helps transport planners and operators allocate vehicles efficiently.

**Built for:** Machine Learning Course — TEST2 | Tanzania 🇹🇿

---

## 📁 Files

| File | Description |
|------|-------------|
| `ml_project.ipynb` | Full Jupyter notebook (data → model → evaluation) |
| `app.py` | Streamlit web application |
| `transport_model.pkl` | Trained model bundle (joblib) |
| `public_transport_dataset.csv` | Tanzanian transport dataset (800 records) |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

---

## 🗂️ Dataset (800 records × 16 columns)

| Feature | Type | Description |
|---------|------|-------------|
| City | Categorical | 10 Tanzanian cities (DSM, Arusha, Mwanza...) |
| Route | Categorical | 10 common routes |
| Transport_Type | Categorical | Daladala, Bus, BRT, Taxi, Tuk-tuk, Ferry |
| Day_Type | Categorical | Weekday, Weekend, Public Holiday |
| Time_Slot | Categorical | 6 time slots (Morning Peak, Evening Peak...) |
| Weather | Categorical | Sunny, Cloudy, Rainy, Heavy Rain |
| Season | Categorical | Dry Season, Short Rains, Long Rains |
| Route_Distance_km | Numerical | 2 – 120 km |
| Fare_TZS | Numerical | 300 – 5,000 TZS |
| Available_Vehicles | Numerical | 1 – 40 vehicles |
| Population_Density | Numerical | 500 – 12,000 people/km² |
| Avg_Wait_Min | Numerical | 2 – 60 minutes |
| Temp_Celsius | Numerical | 18 – 36 °C |
| Near_Market | Binary | 1 = near market, 0 = not |
| Near_School | Binary | 1 = near school, 0 = not |
| **Passengers_Per_Hour** | **Target** | **10 – 700 passengers/hr** |

---

## 📊 Model Results

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| **Linear Regression** | 56.19 | 69.02 | **0.5988 ✔ Best** |
| Decision Tree | 65.81 | 83.45 | 0.4135 |

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy on Streamlit Cloud

1. Push all files to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New App
3. Set main file: `app.py`
4. Deploy!

---

## 🖥️ App Features

- 🔵 Blue transit-themed UI
- Inputs: City, Route, Transport Type, Time Slot, Weather, Season, Fare, Infrastructure
- Outputs: Passengers/hr, Daily estimate, Revenue estimate, Vehicles needed
- Demand level tags: HIGH / MEDIUM / LOW with operational advice

---

*🚌 Public Transport Demand Predictor · Tanzania ML Project · 2026*
