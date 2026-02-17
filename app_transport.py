import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="🚌 Transport Demand Predictor", page_icon="🚌", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a1628 0%, #0d2137 50%, #0a1628 100%); }
.hero { text-align:center; padding:1.5rem 0 0.5rem; }
.hero-title { font-size:2.6rem; font-weight:800; color:#38bdf8;
              text-shadow:0 0 24px rgba(56,189,248,0.4); margin-bottom:0.2rem; }
.hero-sub { color:#7dd3fc; font-size:1rem; margin-bottom:0.2rem; }
.hero-badge { display:inline-block; background:rgba(56,189,248,0.15);
              border:1px solid rgba(56,189,248,0.3); color:#38bdf8;
              border-radius:20px; padding:3px 14px; font-size:0.8rem; margin-top:6px; }
.section-card { background:rgba(255,255,255,0.05); border:1px solid rgba(56,189,248,0.2);
                border-radius:16px; padding:1.2rem 1.4rem; margin-bottom:1rem; }
.section-title { color:#38bdf8; font-weight:700; font-size:1rem;
                 margin-bottom:0.8rem; letter-spacing:0.5px; }
.result-wrap { background:linear-gradient(135deg,#0369a1,#0284c7,#38bdf8);
               border-radius:20px; padding:2rem; text-align:center;
               box-shadow:0 0 50px rgba(56,189,248,0.35); margin-top:1.5rem; }
.res-label { color:rgba(255,255,255,0.8); font-size:0.95rem; }
.res-value { color:#fff; font-size:3rem; font-weight:900; line-height:1.1; }
.res-unit  { color:rgba(255,255,255,0.7); font-size:1rem; }
.res-sub   { color:rgba(255,255,255,0.65); font-size:0.85rem; margin-top:0.4rem; }
.demand-tag { display:inline-block; border-radius:12px; padding:4px 16px;
              font-weight:700; font-size:0.85rem; margin-top:8px; }
.tag-high   { background:#16a34a; color:#fff; }
.tag-medium { background:#d97706; color:#fff; }
.tag-low    { background:#dc2626; color:#fff; }
.stButton>button { width:100%; background:linear-gradient(90deg,#0369a1,#38bdf8);
    color:#fff; font-size:1.1rem; font-weight:700; border:none;
    border-radius:12px; padding:0.85rem; margin-top:0.8rem;
    box-shadow:0 4px 20px rgba(56,189,248,0.3); }
label { color:#cbd5e1 !important; font-size:0.88rem !important; }
.footer { text-align:center; color:rgba(148,163,184,0.4);
          font-size:0.72rem; margin-top:2rem; padding-top:1rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_bundle():
    return joblib.load("transport_model.pkl")

try:
    bun         = load_bundle()
    model       = bun["model"]
    encoders    = bun["encoders"]
    scaler      = bun["scaler"]
    feat_cols   = bun["feature_cols"]
    cat_cols    = bun["cat_cols"]
    num_cols    = bun["num_cols"]
    use_scaler  = bun["use_scaler"]
    best_name   = bun["best_name"]
    lr_m        = bun["lr_metrics"]
    dt_m        = bun["dt_metrics"]
    ok = True
except Exception as e:
    ok = False; err = str(e)

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">🚌 Public Transport Demand Predictor</div>
  <div class="hero-sub">Tanzania Transit Intelligence · Powered by Machine Learning</div>
  <span class="hero-badge">🇹🇿 Tanzania · Daladala · BRT · Bus · Ferry</span>
</div>
""", unsafe_allow_html=True)

if not ok:
    st.error(f"Could not load transport_model.pkl — {err}")
    st.stop()

st.markdown(f"<p style='text-align:center;color:#64748b;font-size:0.78rem;margin-top:8px'>"
            f"Best Model: {best_name} &nbsp;|&nbsp; R²={lr_m['R2']} &nbsp;|&nbsp; MAE={lr_m['MAE']} passengers</p>",
            unsafe_allow_html=True)

# ── INPUTS ────────────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.markdown('<div class="section-card"><div class="section-title">📍 Route Information</div>', unsafe_allow_html=True)
    city           = st.selectbox("City", ['Dar es Salaam','Dodoma','Arusha','Mwanza','Mbeya','Morogoro','Tanga','Zanzibar','Iringa','Tabora'])
    route          = st.selectbox("Route", ['City Centre - Airport','Ubungo - Kariakoo','Mbezi - Posta','Arusha - Moshi','Dodoma - Kondoa','Mwanza - Musoma','Tanga - Korogwe','Mbeya - Songea','Iringa - Morogoro','Tabora - Shinyanga'])
    transport_type = st.selectbox("Transport Type", ['Daladala','Bus','BRT','Taxi','Tuk-tuk','Ferry'])
    route_dist     = st.slider("Route Distance (km)", 2.0, 120.0, 25.0, step=0.5)
    fare           = st.slider("Fare (TZS)", 300, 5000, 800, step=100)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="section-card"><div class="section-title">🕐 Time & Conditions</div>', unsafe_allow_html=True)
    day_type    = st.selectbox("Day Type", ['Weekday','Weekend','Public Holiday'])
    time_slot   = st.selectbox("Time Slot", ['Early Morning (5-7am)','Morning Peak (7-9am)','Midday (10am-12pm)','Afternoon (1-3pm)','Evening Peak (4-7pm)','Night (8-10pm)'])
    weather     = st.selectbox("Weather", ['Sunny','Cloudy','Rainy','Heavy Rain'])
    season      = st.selectbox("Season", ['Dry Season','Short Rains','Long Rains'])
    temp        = st.slider("Temperature (°C)", 18, 36, 27)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card"><div class="section-title">🏙️ Infrastructure & Capacity</div>', unsafe_allow_html=True)
ic1, ic2, ic3, ic4, ic5 = st.columns(5)
avail_vehicles = ic1.number_input("Vehicles Available", 1, 40, 10)
pop_density    = ic2.number_input("Pop. Density (per km²)", 500, 12000, 4000, step=100)
avg_wait       = ic3.slider("Avg Wait (min)", 2, 60, 15)
near_market    = 1 if ic4.checkbox("Near Market?", value=True) else 0
near_school    = 1 if ic5.checkbox("Near School?") else 0
st.markdown('</div>', unsafe_allow_html=True)

# ── PREDICT ───────────────────────────────────────────────────────────────────
if st.button("🚌 Predict Passenger Demand"):
    raw = pd.DataFrame([{
        'City': city, 'Route': route, 'Transport_Type': transport_type,
        'Day_Type': day_type, 'Time_Slot': time_slot, 'Weather': weather, 'Season': season,
        'Route_Distance_km': route_dist, 'Fare_TZS': fare,
        'Available_Vehicles': avail_vehicles, 'Population_Density': pop_density,
        'Avg_Wait_Min': avg_wait, 'Temp_Celsius': temp,
        'Near_Market': near_market, 'Near_School': near_school
    }])

    for col in cat_cols:
        raw[col] = encoders[col].transform(raw[col])

    if use_scaler:
        raw[num_cols] = scaler.transform(raw[num_cols])

    pred = float(model.predict(raw[feat_cols])[0])
    pred = max(10.0, min(700.0, pred))

    if pred >= 400:
        level = "HIGH DEMAND"; tag_cls = "tag-high"; advice = "Deploy additional vehicles immediately"
    elif pred >= 200:
        level = "MEDIUM DEMAND"; tag_cls = "tag-medium"; advice = "Current capacity may be sufficient"
    else:
        level = "LOW DEMAND"; tag_cls = "tag-low"; advice = "Consider reducing vehicle frequency"

    st.markdown(f"""
    <div class="result-wrap">
      <div class="res-label">Predicted Passenger Demand</div>
      <div class="res-value">{pred:,.0f}</div>
      <div class="res-unit">passengers per hour</div>
      <span class="demand-tag {tag_cls}">{level}</span>
      <div class="res-sub" style="margin-top:10px">💡 {advice}</div>
      <div class="res-sub">{city} · {route} · {transport_type} · {time_slot}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🚌 Passengers/hr",   f"{pred:,.0f}")
    m2.metric("📅 Daily Est.",       f"{pred*16:,.0f}", "16 peak hrs")
    m3.metric("💰 Revenue Est.",     f"TZS {pred*fare:,.0f}/hr")
    m4.metric("🚗 Vehicles Needed",  f"{max(1, int(pred/35))} buses")

    # Model metrics expander
    with st.expander("📊 Model Performance Details"):
        col_a, col_b = st.columns(2)
        col_a.markdown("**Linear Regression**")
        col_a.write(f"MAE: {lr_m['MAE']} | RMSE: {lr_m['RMSE']} | R²: {lr_m['R2']} ✔")
        col_b.markdown("**Decision Tree**")
        col_b.write(f"MAE: {dt_m['MAE']} | RMSE: {dt_m['RMSE']} | R²: {dt_m['R2']}")

st.markdown('<div class="footer">🚌 Public Transport Demand Predictor · Tanzania ML Project · 2026</div>', unsafe_allow_html=True)
