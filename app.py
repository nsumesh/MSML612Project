"""
This is the main Streamlit app. It has two tabs — Live Prediction and Model Performance.

In the Live Prediction tab, you pick a season, circuit, and driver, and it pulls the race
data through FastF1. From there you can drag a slider to any point in the race, and the
model will use the previous 15 laps as context to predict the next 5. You get a chart
showing the context laps, the model's predictions, and the actual lap times side by side
(when they're available), and a breakdown table showing predicted vs actual for each lap.

The Model Performance tab loads all three data splits, runs them through the model, and
shows you the overall MAE and RMSE, a leaderboard of how accurately the model predicted
each driver's laps (filterable by season and race), and a bar chart comparing prediction
error circuit by circuit.
"""

import json
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fastf1
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import streamlit as st

from src.model.transformer import LapTimeTransformer
from src.data.inference import prepare_windows, lookback_context, prediction_horizon

fastf1_track_map = {"australia":"Australia","china":"China", "japan":"Japan", "bahrain":"Bahrain", "saudi":"Saudi Arabia", "miami":"Miami","imola":"Emilia Romagna", "emilia":"Emilia Romagna", "monaco":"Monaco", "canada":"Canada","spain":"Spain","austria":"Austria","great britain": "Great Britain","silverstone":"Great Britain","hungary":"Hungary","belgium":"Belgium","netherlands":"Netherlands","italy":"Italy","monza":"Italy","azerbaijan":"Azerbaijan","baku":"Azerbaijan","singapore":"Singapore","united states": "United States","austin":"United States","mexico":"Mexico City","brazil":"São Paulo","são paulo":"São Paulo","las vegas":"Las Vegas","qatar":"Qatar","abu dhabi":"Abu Dhabi",}

compound_colors = {"SOFT":"#E8002D", "MEDIUM":"#FFC906", "HARD":"#AAAAAA", "INTERMEDIATE":"#39B54A","WET":"#0067FF",}

driver_names = {"ALB": "Alexander Albon", "ALO": "Fernando Alonso", "ANT": "Andrea Kimi Antonelli", "BEA": "Oliver Bearman", "BOR": "Gabriel Bortoleto", "BOT": "Valtteri Bottas", "COL": "Franco Colapinto", "DOO": "Jack Doohan", "GAS": "Pierre Gasly", "HAD": "Isack Hadjar", "HAM": "Lewis Hamilton", "HUL": "Nico Hülkenberg", "LAW": "Liam Lawson", "LEC": "Charles Leclerc", "MAG": "Kevin Magnussen", "NOR": "Lando Norris", "OCO": "Esteban Ocon", "PER": "Sergio Pérez", "PIA": "Oscar Piastri", "RIC": "Daniel Ricciardo", "RUS": "George Russell", "SAI": "Carlos Sainz", "SAR": "Logan Sargeant","STR": "Lance Stroll", "TSU": "Yuki Tsunoda", "VER": "Max Verstappen", "ZHO": "Guanyu Zhou"}


@st.cache_resource
def load_model_and_info():
    info = json.loads(open("data/splits/info.json").read())
    scaler = pickle.load(open("data/splits/scaler.pkl", "rb"))
    device = torch.device("cpu")
    model = LapTimeTransformer(n_drivers=info["n_drivers"], n_tracks=info["n_tracks"])
    model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
    model.eval()
    return model, info, scaler, device


@st.cache_data(show_spinner=False)
def get_completed_tracks(year, trained_tracks):
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        past = schedule[pd.to_datetime(schedule["EventDate"]) < pd.Timestamp.now()]
        available = set()
        for i, row in past.iterrows():
            candidates = " ".join([str(row.get("EventName", "")), str(row.get("Country", "")), str(row.get("Location", "")),]).lower()
            for key, track in fastf1_track_map.items():
                if key in candidates and track in trained_tracks:
                    available.add(track)
        return sorted(available) if available else list(trained_tracks)
    except Exception:
        return list(trained_tracks)

@st.cache_data(show_spinner=False)
def fetch_race_cached(year, race_name, driver, info, _scaler):
    return prepare_windows(year, race_name, driver, info, _scaler)


@st.cache_data(show_spinner=False)
def run_inference_cached(_model, _device, _tensor):
    with torch.no_grad():
        out = _model(_tensor.to(_device))
    return out[:, -prediction_horizon:].cpu().numpy()


@st.cache_data(show_spinner=False)
def load_performance_data(_model, _device):
    parts = []
    for split in ("train", "val", "test"):
        m = pd.read_csv(f"data/splits/meta_{split}.csv")
        m["split"] = split
        parts.append((m,np.load(f"data/splits/data_{split}.npy"),np.load(f"data/splits/labels_{split}.npy")))
    meta = pd.concat([p[0] for p in parts], ignore_index=True)
    data = np.concatenate([p[1] for p in parts], axis=0)
    labels = np.concatenate([p[2] for p in parts], axis=0)
    preds = []
    with torch.no_grad():
        for i in range(0, len(data), 256):
            x = torch.tensor(data[i:i+256], dtype=torch.float32).to(_device)
            preds.append(_model(x)[:, -prediction_horizon:].cpu().numpy())
    preds = np.concatenate(preds)
    last = meta["last_lap_time"].values[:, None]
    preds_abs = preds + last
    labs_abs = labels + last
    meta["mae"] = np.abs(preds_abs - labs_abs).mean(axis=1)
    return meta, preds_abs, labs_abs


def fmt_time(s):
    return f"{int(s // 60)}:{s % 60:06.3f}"


st.set_page_config(page_title="F1 Lap Time Predictor", layout="wide")

st.markdown("""
<style>
    html, body, [class*="css"] { color: #111111 !important; }
    [data-testid="stSidebar"] {
        background: #1a1a2e;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.2) !important;
    }
    .block-container { padding-top: 1.8rem; }
    [data-testid="metric-container"] {
        background: #f7f7f9;
        border: 1px solid #ebebeb;
        border-radius: 10px;
        padding: 14px 18px;
    }
    div[data-testid="metric-container"] label {
        color: #666666 !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="metric-container"] div[data-testid="metric-value"] {
        color: #111111 !important;
        font-size: 1.4rem !important;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.9rem;
        color: #555;
    }
    .stTabs [aria-selected="true"] {
        color: #E8002D !important;
        border-bottom-color: #E8002D !important;
    }
    .stButton > button[kind="primary"] {
        background: #E8002D;
        border: none;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .stButton > button[kind="primary"]:hover {
        background: #c0001f;
    }
</style>
""", unsafe_allow_html=True)

model, info, scaler, device = load_model_and_info()
trained_tracks  = sorted(info["track_to_id"].keys())
trained_drivers = sorted(info["driver_to_id"].keys())


with st.sidebar:
    st.markdown("## F1 Lap Predictor")
    st.markdown("**15-lap context → 5-lap forecast**")
    st.divider()
    st.markdown("**Model trained on**")
    st.markdown(f"- 2024 & 2025 seasons  \n- {len(trained_tracks)} circuits  \n- {len(trained_drivers)} drivers")
    st.divider()
    st.markdown("Works on **any season** using existing circuits and drivers — including **2026**.")


tab1, tab2 = st.tabs(["Live Prediction", "Model Performance"])


with tab1:
    c1, c2, c3, c4 = st.columns([1, 2, 2, 1])
    with c1:
        year = st.number_input("Season", min_value=2018, max_value=2030, value=2025, step=1, label_visibility="visible")
    available_tracks = get_completed_tracks(int(year), trained_tracks)
    with c2:
        race_name = st.selectbox("Circuit", available_tracks)
    with c3:
        driver_options = [f"{code} — {driver_names.get(code, code)}" for code in trained_drivers]
        driver_label = st.selectbox("Driver", driver_options)
        driver = driver_label.split(" — ")[0]
    with c4:
        st.markdown("<div style='padding-top:26px'>", unsafe_allow_html=True)
        load = st.button("Load Race ›", use_container_width=True, type="primary")
        st.markdown("</div>", unsafe_allow_html=True)
    st.divider()
    session_key = (year, race_name, driver)
    if load:
        st.session_state["live_key"] = session_key

    if "live_key" not in st.session_state or st.session_state["live_key"] != session_key:
        st.markdown("<div style='text-align:center;padding:60px 0;color:#888;font-size:1.1rem'>Select a season, circuit, and driver above, then click <b>Load Race ›</b></div>", unsafe_allow_html=True)
        st.stop()

    with st.spinner(f"Fetching {race_name} {year} · {driver}…"):
        try:
            tensor, meta, drv_laps = fetch_race_cached(year, race_name, driver, info, scaler)
        except Exception as e:
            st.error(f"**Could not load race data.** {e}")
            st.stop()

    residuals  = run_inference_cached(model, device, tensor)
    lap_options = [m["start_lap"] for m in meta]

    driver_full = driver_names.get(driver, driver)
    st.markdown(f"### {driver_full}  ·  {race_name} {year}")

    if st.session_state.get("slider_race_key") != session_key:
        st.session_state["slider_race_key"] = session_key
        st.session_state["prediction_lap"]  = lap_options[0]

    nav_l, nav_r = st.columns([4, 1])
    with nav_l:
        chosen_lap = st.select_slider("Slide to a point in the race — the model predicts the next 5 laps from there",options=lap_options,key="prediction_lap",)
    win_idx       = lap_options.index(chosen_lap)
    m             = meta[win_idx]
    last_lap_time = m["last_lap_time"]
    pred_residual = residuals[win_idx]
    pred_abs      = pred_residual + last_lap_time
    start_lap     = m["start_lap"]
    context_laps  = list(range(start_lap - lookback_context, start_lap))
    future_laps   = list(range(start_lap, start_lap + prediction_horizon))
    ctx_rows = drv_laps[drv_laps["LapNumber"].isin(context_laps)]
    act_rows = drv_laps[drv_laps["LapNumber"].isin(future_laps)]
    has_actuals = len(act_rows) == prediction_horizon
    with nav_r:
        st.metric("Race laps available", len(drv_laps))
        st.metric("Tyre at prediction", m["compound"])

    if has_actuals:
        act_vals = act_rows.set_index("LapNumber")["LapTime"]
        errors   = [pred_abs[i] - float(act_vals[future_laps[i]]) for i in range(prediction_horizon)]
        window_mae = np.abs(errors).mean()
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Lap +1", fmt_time(pred_abs[0]))
        m2.metric("Actual Lap +1",    fmt_time(float(act_vals[future_laps[0]])))
        m3.metric("Window MAE",       f"{window_mae:.3f} s")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Lap +1", fmt_time(pred_abs[0]))
        m2.metric("Predicted Lap +3", fmt_time(pred_abs[2]))
        m3.metric("Predicted Lap +5", fmt_time(pred_abs[4]))
    fig = go.Figure()

    y_vals = list(ctx_rows["LapTime"]) + list(pred_abs)
    y_min  = min(y_vals) - 0.5
    y_max  = max(y_vals) + 0.5
    fig.add_shape(type="rect",x0=start_lap - 0.5, x1=start_lap + prediction_horizon - 0.5,y0=y_min, y1=y_max, fillcolor="rgba(232,0,45,0.04)", line=dict(width=0),layer="below",)

    fig.add_trace(go.Scatter(x=ctx_rows["LapNumber"].tolist(),y=ctx_rows["LapTime"].tolist(),mode="lines",name="Context",line=dict(color="#cccccc", width=2),showlegend=True,hovertemplate="Lap %{x}: %{y:.3f}s<extra>Context</extra>",))

    for compound, grp in ctx_rows.groupby("Compound"):
        color = compound_colors.get(compound, "#888888")
        fig.add_trace(go.Scatter(x=grp["LapNumber"].tolist(),y=grp["LapTime"].tolist(),mode="markers",name=compound, marker=dict(size=9, color=color, line=dict(width=1, color="white")), hovertemplate="Lap %{x}: %{y:.3f}s<extra>" + compound + "</extra>",))

    if has_actuals:
        fig.add_trace(go.Scatter(x=act_rows["LapNumber"].tolist(),y=act_rows["LapTime"].tolist(),mode="lines+markers",name="Actual",line=dict(color="#2196F3", width=2.5),marker=dict(size=10, symbol="circle", color="#2196F3", line=dict(width=1.5, color="white")),hovertemplate="Lap %{x}: %{y:.3f}s<extra>Actual</extra>",))

    fig.add_trace(go.Scatter(x=future_laps,y=pred_abs.tolist(),mode="lines+markers",name="Predicted",line=dict(color="#E8002D", width=2.5, dash="dash"),marker=dict(size=10, symbol="square", color="#E8002D", line=dict(width=1.5, color="white")),hovertemplate="Lap %{x}: %{y:.3f}s<extra>Predicted</extra>",))

    fig.add_shape(type="line",x0=start_lap - 0.5, x1=start_lap - 0.5,y0=y_min, y1=y_max,line=dict(color="#aaaaaa", width=1.5, dash="dot"),)
    fig.add_annotation(x=start_lap - 0.4, y=y_max - 0.1,text="◀ observed  ·  predicted ▶",showarrow=False,font=dict(size=11, color="#888888"),xanchor="left",bgcolor="white",borderpad=3,)

    fig.update_layout(template="plotly_white",font=dict(family="Inter, sans-serif", size=13, color="#111111"),paper_bgcolor="white",plot_bgcolor="white",hovermode="x unified",margin=dict(t=20, b=40, l=60, r=20),xaxis=dict(title="Lap number", showgrid=True, gridcolor="#f0f0f0", zeroline=False, color="#333333"),yaxis=dict(title="Lap time (s)", showgrid=True, gridcolor="#f0f0f0", zeroline=False, color="#333333", range=[y_min, y_max]),legend=dict(orientation="h", x=0, y=-0.18, bgcolor="rgba(255,255,255,0.9)", bordercolor="#e0e0e0", borderwidth=1, font=dict(size=12, color="#111111")),height=420,)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Lap-by-lap forecast")
    rows = []
    for i, lap in enumerate(future_laps):
        row = {"Lap": f"Lap {lap}  (+{i+1})", "Predicted": fmt_time(pred_abs[i])}
        if has_actuals and lap in act_rows["LapNumber"].values:
            act_t  = float(act_rows[act_rows["LapNumber"] == lap]["LapTime"].values[0])
            delta  = pred_abs[i] - act_t
            row["Actual"] = fmt_time(act_t)
            row["Error"]  = f"{delta:+.3f}s"
        rows.append(row)

    tdf = pd.DataFrame(rows)
    st.dataframe(tdf,use_container_width=True,hide_index=True,column_config={"Lap": st.column_config.TextColumn("Lap", width="medium"),"Predicted": st.column_config.TextColumn("Predicted", width="medium"),"Actual": st.column_config.TextColumn("Actual", width="medium"),"Error": st.column_config.TextColumn("Error", width="small"),},)

with tab2:
    with st.spinner("Loading performance data"):
        perf_meta, perf_preds, perf_labs = load_performance_data(model, device)

    overall_mae  = float(np.abs(perf_preds - perf_labs).mean())
    overall_rmse = float(np.sqrt(((perf_preds - perf_labs) ** 2).mean()))
    test_meta    = perf_meta[perf_meta["split"] == "test"]

    st.markdown("### Model accuracy on held-out test set")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Test MAE",     f"{overall_mae:.3f} s")
    mc2.metric("Test RMSE",    f"{overall_rmse:.3f} s")
    mc3.metric("Test windows", f"{len(test_meta):,}")
    mc4.metric("Seasons",      "2024 – 2025")

    st.divider()

    col_l, col_r = st.columns([1, 2], gap="large")

    with col_l:
        st.markdown("#### Driver accuracy")
        yr = st.selectbox("Season", sorted(perf_meta["year"].unique(), reverse=True), key="p_yr")
        rc = st.selectbox("Race", ["All races"] + sorted(perf_meta[perf_meta["year"] == yr]["race"].unique()), key="p_rc")
        filt = (perf_meta[perf_meta["year"] == yr] if rc == "All races" else perf_meta[(perf_meta["year"] == yr) & (perf_meta["race"] == rc)])
        board = filt.groupby("driver")["mae"].agg(mean="mean", min="min", count="count").round(3).sort_values("mean").reset_index().rename(columns={"driver": "Driver", "mean": "Avg MAE (s)", "min": "Best MAE (s)", "count": "Windows"})
        st.dataframe(board, use_container_width=True, hide_index=True)

    with col_r:
        st.markdown("#### Prediction error by circuit")
        yr2 = st.selectbox("Season ", sorted(perf_meta["year"].unique(), reverse=True), key="p_yr2")
        race_summary = perf_meta[perf_meta["year"] == yr2].groupby("race")["mae"].mean().sort_values().reset_index()
        colors = ["#2ECC71" if v < 0.4 else "#F39C12" if v < 0.7 else "#E74C3C" for v in race_summary["mae"]]
        fig2 = go.Figure(go.Bar(x=race_summary["mae"],y=race_summary["race"],orientation="h",marker_color=colors,hovertemplate="%{y}: %{x:.3f}s<extra></extra>",text=[f"{v:.3f}s" for v in race_summary["mae"]],textposition="outside",textfont=dict(size=11, color="#333333"),))
        fig2.add_vline(x=0.5, line_dash="dash", line_color="#555555", line_width=1.5,annotation_text="0.5 s target",annotation_font=dict(size=11, color="#555555"),annotation_position="top right",)
        fig2.update_layout(template="plotly_white",font=dict(family="Inter, sans-serif", size=12, color="#111111"),paper_bgcolor="white",plot_bgcolor="white",margin=dict(t=20, b=20, l=10, r=80),xaxis=dict(title="Avg MAE (s)", color="#333333", showgrid=True, gridcolor="#f0f0f0", zeroline=False),yaxis=dict(color="#333333", tickfont=dict(size=11)),showlegend=False,height=520,)
        st.plotly_chart(fig2, use_container_width=True)
