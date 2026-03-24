import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from scipy.stats import pearsonr
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

BACKEND_URL = "http://localhost:8000"
REFRESH_MS  = 300_000
OWM_API_KEY = "2336ee800c626251c5e3c1e52a8c5709"
OWM_CITY    = "Knightsbridge,London,GB"
OWM_URL     = "https://api.openweathermap.org/data/2.5/weather"

WHO_SLEEP, WHO_DAY = 35, 45



CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=DM+Serif+Display&display=swap');
:root {
    --bg-deep:#0F0A06;--bg-panel:#1A1208;--bg-card:#221A0E;--border:#3A2D1A;
    --border-lit:#6B4F2A;--amber:#F5A623;--amber-dim:#B87320;--cream:#F2E8D5;
    --cream-dim:rgba(242,232,213,0.5);--cream-mute:rgba(242,232,213,0.2);
    --red:#C0392B;--green:#27AE60;--blue:#2E86AB;--violet:#8E44AD;
    --mono:'IBM Plex Mono',monospace;--serif:'DM Serif Display',serif;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg-deep);color:var(--cream);font-family:var(--mono);min-height:100vh;overflow-x:hidden;}
body::before{content:'';position:fixed;inset:0;background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");pointer-events:none;z-index:0;opacity:0.4;}
body::after{content:'';position:fixed;inset:0;background-image:repeating-linear-gradient(0deg,transparent,transparent 28px,rgba(245,166,35,0.025) 28px,rgba(245,166,35,0.025) 29px);pointer-events:none;z-index:0;}
.dashboard-wrapper{position:relative;z-index:1;max-width:1600px;margin:0 auto;padding:32px 40px;}
.header{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:36px;padding-bottom:24px;border-bottom:1px solid var(--border-lit);}
.header-eyebrow{font-family:var(--mono);font-size:10px;font-weight:500;color:var(--amber-dim);letter-spacing:4px;text-transform:uppercase;margin-bottom:10px;}
.header-title{font-family:var(--serif);font-size:56px;font-weight:400;line-height:1;color:var(--amber);letter-spacing:-1px;text-shadow:0 0 40px rgba(245,166,35,0.3);}
.header-sub{font-family:var(--mono);font-size:11px;color:var(--cream-mute);margin-top:10px;letter-spacing:2px;}
.header-right{display:flex;flex-direction:column;align-items:flex-end;gap:10px;padding-top:4px;}
.live-pill{display:flex;align-items:center;gap:8px;font-family:var(--mono);font-size:11px;letter-spacing:3px;font-weight:500;padding:6px 14px;border-radius:2px;border:1px solid;}
.live-pill.online{color:var(--green);border-color:rgba(39,174,96,0.4);background:rgba(39,174,96,0.06);}
.live-pill.offline{color:var(--red);border-color:rgba(192,57,43,0.4);background:rgba(192,57,43,0.06);}
.live-dot{width:6px;height:6px;border-radius:50%;background:var(--green);box-shadow:0 0 8px var(--green);animation:blink 1.5s ease-in-out infinite;}
.offline-dot{width:6px;height:6px;border-radius:50%;background:var(--red);}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:0.3;}}
.sync-time{font-family:var(--mono);font-size:10px;color:var(--cream-mute);letter-spacing:2px;}
.weather-banner{display:grid;grid-template-columns:repeat(7,1fr);gap:0;margin-bottom:20px;background:var(--bg-panel);border:1px solid var(--border);border-left:3px solid #2E86AB;overflow:hidden;}
.weather-item{padding:16px 20px;border-right:1px solid var(--border);display:flex;flex-direction:column;justify-content:center;gap:6px;min-width:0;}
.weather-item:last-child{border-right:none;}
.weather-label{font-family:var(--mono);font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--amber-dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.weather-value{font-family:var(--mono);font-size:22px;font-weight:600;color:#5BA4CF;line-height:1.1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.weather-sub{font-family:var(--mono);font-size:10px;color:var(--cream-mute);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.controls-bar{display:flex;align-items:center;gap:20px;margin-bottom:28px;padding:14px 20px;background:var(--bg-panel);border:1px solid var(--border);border-radius:2px;}
.control-label{font-family:var(--mono);font-size:10px;color:var(--amber-dim);letter-spacing:3px;text-transform:uppercase;}
.Select-control{background-color:#1A1208!important;border:1px solid #3A2D1A!important;border-radius:2px!important;color:#F2E8D5!important;}
.Select-control:hover{border-color:#6B4F2A!important;box-shadow:none!important;}
.Select--single>.Select-control .Select-value,.Select-placeholder{color:#F2E8D5!important;font-family:'IBM Plex Mono',monospace!important;font-size:11px!important;line-height:36px!important;}
.Select-value-label{color:#F2E8D5!important;font-family:'IBM Plex Mono',monospace!important;font-size:11px!important;}
.Select-arrow-zone .Select-arrow{border-top-color:#B87320!important;}
.Select-menu-outer{background-color:#1A1208!important;border:1px solid #3A2D1A!important;border-radius:2px!important;box-shadow:0 4px 20px rgba(0,0,0,0.6)!important;}
.Select-menu{background-color:#1A1208!important;}
.Select-option{background-color:#1A1208!important;color:#F2E8D5!important;font-family:'IBM Plex Mono',monospace!important;font-size:11px!important;padding:10px 16px!important;}
.Select-option:hover,.Select-option.is-focused{background-color:#3A2D1A!important;color:#F5A623!important;}
.Select-option.is-selected{background-color:#221A0E!important;color:#F5A623!important;}
.is-open>.Select-control{border-color:#6B4F2A!important;background-color:#1A1208!important;}
.stats-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px;}
.stat-card{background:var(--bg-card);border:1px solid var(--border);border-top:2px solid;padding:20px 24px;}
.stat-card.amber{border-top-color:var(--amber);}
.stat-card.cream{border-top-color:var(--cream-dim);}
.stat-card.green{border-top-color:var(--green);}
.stat-card.red{border-top-color:#E67E22;}
.stat-card.blue{border-top-color:var(--blue);}
.stat-card.violet{border-top-color:var(--violet);}
.stat-ticker{font-family:var(--mono);font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--amber-dim);margin-bottom:10px;}
.stat-value{font-family:var(--mono);font-size:32px;font-weight:600;line-height:1;letter-spacing:-1px;}
.stat-value.amber{color:var(--amber);}
.stat-value.cream{color:var(--cream);}
.stat-value.green{color:var(--green);}
.stat-value.red{color:#E67E22;}
.stat-value.blue{color:var(--blue);}
.stat-value.violet{color:var(--violet);}
.stat-unit{font-family:var(--mono);font-size:10px;color:var(--cream-mute);margin-top:6px;letter-spacing:1px;}
.chart-card{background:var(--bg-card);border:1px solid var(--border);padding:24px 28px;margin-bottom:16px;position:relative;}
.chart-card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,var(--amber),transparent 60%);}
.chart-eyebrow{font-family:var(--mono);font-size:9px;letter-spacing:4px;text-transform:uppercase;color:var(--amber-dim);margin-bottom:4px;}
.chart-title{font-family:var(--serif);font-size:20px;color:var(--cream);margin-bottom:4px;font-weight:400;}
.chart-desc{font-family:var(--mono);font-size:10px;color:var(--cream-mute);margin-bottom:16px;letter-spacing:0.5px;}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;}
.three-col{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;}
.four-col{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;}
.section-divider{height:1px;background:linear-gradient(90deg,var(--border-lit),transparent);margin:8px 0 20px 0;}
.section-header{font-family:var(--serif);font-size:28px;color:var(--amber);margin:32px 0 8px 0;padding-bottom:12px;border-bottom:1px solid var(--border-lit);}
.section-sub{font-family:var(--mono);font-size:10px;color:var(--cream-mute);letter-spacing:2px;margin-bottom:20px;}
.footer{margin-top:24px;padding-top:20px;border-top:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;}
.footer-text{font-family:var(--mono);font-size:10px;color:var(--cream-mute);letter-spacing:2px;}
.footer-mark{font-family:var(--mono);font-size:10px;color:var(--amber-dim);letter-spacing:1px;}
"""

app = dash.Dash(__name__, title="RoomPulse Terminal")
app.index_string = (
    "<!DOCTYPE html>\n<html>\n    <head>\n        {%metas%}\n"
    "        <title>{%title%}</title>\n        {%favicon%}\n        {%css%}\n"
    "        <style>\n" + CUSTOM_CSS + "        </style>\n    </head>\n"
    "    <body>\n        {%app_entry%}\n        <footer>\n"
    "            {%config%}\n            {%scripts%}\n            {%renderer%}\n"
    "        </footer>\n    </body>\n</html>"
)


# ---------- Data fetching ----------

def fetch_data(hours=168):
    try:
        r = requests.get(f"{BACKEND_URL}/data/hourly", params={"hours": hours}, timeout=5)
        return r.json()
    except:
        return []

def fetch_weather_history(hours=168):
    try:
        r = requests.get(f"{BACKEND_URL}/data/weather", params={"hours": hours}, timeout=5)
        data = r.json()
        if not data: return pd.DataFrame()
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values("timestamp").reset_index(drop=True)
    except:
        return pd.DataFrame()

def fetch_status():
    try:
        r = requests.get(f"{BACKEND_URL}/status", timeout=3)
        return r.json()
    except:
        return {}

def fetch_weather():
    try:
        r = requests.get(OWM_URL, params={"q": OWM_CITY, "appid": OWM_API_KEY, "units": "metric"}, timeout=5)
        d = r.json()
        return {
            "temp":        round(d["main"]["temp"], 1),
            "feels_like":  round(d["main"]["feels_like"], 1),
            "humidity":    d["main"]["humidity"],
            "description": d["weather"][0]["description"],
            "clouds":      d["clouds"]["all"],
            "wind_speed":  round(d["wind"]["speed"], 1),
            "rain":        d.get("rain", {}).get("1h", 0.0),
        }
    except:
        return {}

def build_df(data):
    if not data: return pd.DataFrame()
    df = pd.DataFrame(data)
    df["hour_ts"] = pd.to_datetime(df["hour_ts"])
    return df.sort_values("hour_ts").reset_index(drop=True)




PLOT_BG = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono, monospace", color="rgba(242,232,213,0.5)", size=10),
    xaxis=dict(gridcolor="rgba(58,45,26,0.8)", linecolor="rgba(107,79,42,0.5)",
               tickcolor="rgba(107,79,42,0.5)", tickfont=dict(color="rgba(242,232,213,0.4)", size=9),
               showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="rgba(58,45,26,0.8)", linecolor="rgba(107,79,42,0.5)",
               tickcolor="rgba(107,79,42,0.5)", tickfont=dict(color="rgba(242,232,213,0.4)", size=9),
               showgrid=True, zeroline=False),
    margin=dict(l=54, r=16, t=10, b=40), showlegend=False, hovermode="x unified",
)

AMBER_C  = "#F5A623"
TEAL_C   = "#2E86AB"
CORAL_C  = "#E67E22"
CREAM_C  = "#F2E8D5"
GREEN_C  = "#27AE60"
VIOLET_C = "#8E44AD"
RED_C    = "#C0392B"
MUTED_C  = "#B87320"


def empty_fig(msg="AWAITING DATA...", height=220):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                       font=dict(size=11, color="rgba(184,115,32,0.5)", family="IBM Plex Mono"))
    fig.update_layout(**PLOT_BG, height=height)
    return fig


def corr_scatter(x, y, x_label, y_label, color, height=260):
    """Scatter plot with pearson r annotation and trendline."""
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    if n < 5: return empty_fig("COLLECTING DATA...", height)

    r_val, _ = pearsonr(x, y)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    xl = np.linspace(x.min(), x.max(), 100)
    strength = "STRONG" if abs(r_val) > 0.6 else "MOD." if abs(r_val) > 0.3 else "WEAK"
    sign = "+" if r_val >= 0 else "-"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers",
                             marker=dict(color=color, size=5, opacity=0.6,
                                         line=dict(color="rgba(242,232,213,0.08)", width=1)),
                             hovertemplate=f"{x_label}: %{{x:.1f}}<br>{y_label}: %{{y:.1f}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=xl, y=p(xl), mode="lines",
                             line=dict(color=color, width=1.5, dash="dot"), opacity=0.5, hoverinfo="skip"))
    fig.add_annotation(text=f"r = {r_val:.3f}  ·  {sign}{strength}",
                       xref="paper", yref="paper", x=0.04, y=0.93, showarrow=False,
                       font=dict(size=11, color=color, family="IBM Plex Mono"),
                       bgcolor="rgba(15,10,6,0.8)", borderpad=5)

    layout = {**PLOT_BG}
    layout["xaxis"]  = {**PLOT_BG["xaxis"], "title": x_label}
    layout["yaxis"]  = {**PLOT_BG["yaxis"], "title": y_label}
    layout["height"] = height
    layout["hovermode"] = "closest"
    fig.update_layout(**layout)
    return fig


# ---------- Sensor charts ----------

def fig_sound(df):
    if df.empty: return empty_fig()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["hour_ts"], y=df["avg_db"], fill="tozeroy",
                             fillcolor="rgba(245,166,35,0.07)", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=df["hour_ts"], y=df["avg_db"], mode="lines",
                             line=dict(color="#F5A623", width=1.5),
                             hovertemplate="<b>%{y:.1f} dB SPL</b><extra></extra>"))
    layout = {**PLOT_BG}; layout["yaxis"] = {**PLOT_BG["yaxis"], "title": "dB SPL"}; layout["height"] = 220
    fig.update_layout(**layout)
    return fig


def fig_pir(df):
    if df.empty: return empty_fig()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["hour_ts"], y=df["total_pir"],
                         marker=dict(color=df["total_pir"],
                                     colorscale=[[0,"rgba(242,232,213,0.08)"],[1,"rgba(242,232,213,0.7)"]],
                                     line=dict(width=0)),
                         hovertemplate="<b>%{y} events</b><extra></extra>"))
    layout = {**PLOT_BG}; layout["yaxis"] = {**PLOT_BG["yaxis"], "title": "Events / hr"}; layout["height"] = 220
    fig.update_layout(**layout)
    return fig


def fig_light(df):
    if df.empty: return empty_fig()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["hour_ts"], y=df["avg_light"], fill="tozeroy",
                             fillcolor="rgba(230,126,34,0.07)", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=df["hour_ts"], y=df["avg_light"], mode="lines",
                             line=dict(color="#E67E22", width=1.5),
                             hovertemplate="<b>%{y:.0f}%</b><extra></extra>"))
    layout = {**PLOT_BG}; layout["yaxis"] = {**PLOT_BG["yaxis"], "title": "Light %", "range": [0,105]}; layout["height"] = 220
    fig.update_layout(**layout)
    return fig


def fig_heatmap(df):
    if df.empty or len(df) < 2: return empty_fig("NEED MORE DATA", 280)
    df = df.copy()
    df["day"] = df["hour_ts"].dt.day_name()
    df["hr"]  = df["hour_ts"].dt.hour
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = df.pivot_table(index="day", columns="hr", values="avg_db", aggfunc="mean")
    pivot = pivot.reindex([d for d in order if d in pivot.index])

    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=[f"{h:02d}:00" for h in pivot.columns], y=pivot.index.tolist(),
        colorscale=[[0,"rgba(34,26,14,1)"],[0.4,"rgba(184,115,32,0.5)"],[1,"rgba(245,166,35,1)"]],
        showscale=True,
        colorbar=dict(title=dict(text="dB", font=dict(color="rgba(242,232,213,0.4)", size=10)),
                      tickfont=dict(color="rgba(242,232,213,0.4)", size=9),
                      thickness=10, len=0.8, bgcolor="rgba(0,0,0,0)", bordercolor="rgba(58,45,26,0.5)"),
        hovertemplate="<b>%{y} %{x}: %{z:.1f} dB</b><extra></extra>",
    ))
    layout = {**PLOT_BG}; layout["height"] = 280; layout["margin"] = dict(l=90,r=70,t=10,b=50)
    layout["xaxis"] = {**PLOT_BG["xaxis"], "title": "Hour of Day"}
    fig.update_layout(**layout)
    return fig


def fig_fft(df):
    if df.empty or len(df) < 12: return empty_fig("NEED 12+ HRS", 260)
    x = df["avg_db"].fillna(0).values - df["avg_db"].mean()
    fft_vals = np.abs(np.fft.rfft(x))
    freq = np.fft.rfftfreq(len(x), d=1.0)
    mask = freq[1:] > 0
    periods = 1.0/freq[1:][mask]
    amps = fft_vals[1:][mask]
    ok = (periods >= 1) & (periods <= 200)
    periods, amps = periods[ok], amps[ok]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=periods, y=amps, mode="lines", fill="tozeroy",
                             fillcolor="rgba(245,166,35,0.06)", line=dict(color="#F5A623", width=1.5),
                             hovertemplate="<b>%{x:.1f}h period, amp: %{y:.2f}</b><extra></extra>"))
    for p, lbl in [(24,"24H"),(168,"7D")]:
        if len(periods) and periods.min() <= p <= periods.max():
            fig.add_vline(x=p, line_dash="dash", line_color="rgba(184,115,32,0.4)", line_width=1,
                          annotation_text=lbl,
                          annotation_font=dict(color="rgba(184,115,32,0.7)", size=9, family="IBM Plex Mono"),
                          annotation_position="top left")
    layout = {**PLOT_BG}
    layout["xaxis"] = {**PLOT_BG["xaxis"], "title": "Period (hours)", "type": "log"}
    layout["yaxis"] = {**PLOT_BG["yaxis"], "title": "Amplitude"}
    layout["height"] = 260
    fig.update_layout(**layout)
    return fig


# ---------- Weather correlation charts ----------

def fig_wx_sound_vs_temp(wxdf):
    if wxdf.empty or "temp" not in wxdf.columns: return empty_fig("COLLECTING WEATHER DATA...", 260)
    return corr_scatter(wxdf["temp"].dropna().values, wxdf["avg_db"].dropna().values, "Outdoor Temp (°C)", "Sound Level (dB SPL)", AMBER_C)

def fig_wx_pir_vs_clouds(wxdf):
    if wxdf.empty or "clouds" not in wxdf.columns: return empty_fig("COLLECTING WEATHER DATA...", 260)
    return corr_scatter(wxdf["clouds"].dropna().values, wxdf["pir_count"].dropna().values, "Cloud Cover (%)", "PIR Motion Events", CORAL_C)

def fig_wx_light_vs_humidity(wxdf):
    if wxdf.empty or "humidity" not in wxdf.columns: return empty_fig("COLLECTING WEATHER DATA...", 260)
    return corr_scatter(wxdf["humidity"].dropna().values, wxdf["light_pct"].dropna().values, "Outdoor Humidity (%)", "Indoor Light (%)", CREAM_C)


# ---------- ML features ----------


def compute_ml_features(wxdf):
    """Run all five ML features. Returns a dict keyed by feature name."""
    results = {}
    if wxdf.empty or len(wxdf) < 20:
        print(f"[ML] Skipping, only {len(wxdf) if not wxdf.empty else 0} records (need 20+)")
        return results

    print(f"[ML] Running on {len(wxdf)} records")
    df = wxdf.copy()
    df["hour"]         = df["timestamp"].dt.hour
    df["dow"]          = df["timestamp"].dt.dayofweek
    df["date"]         = df["timestamp"].dt.date
    df["sin_hour"]     = np.sin(2*np.pi*df["hour"]/24)
    df["cos_hour"]     = np.cos(2*np.pi*df["hour"]/24)
    df["sin_dow"]      = np.sin(2*np.pi*df["dow"]/7)
    df["cos_dow"]      = np.cos(2*np.pi*df["dow"]/7)
    df["night_window"] = ((df["hour"]>=23)|(df["hour"]<7)).astype(int)

    # Feature 1: KMeans behavioural clustering
    try:
        feats = ["avg_db","pir_count","light_pct"]
        if all(c in df.columns for c in feats):
            scaler = StandardScaler()
            X = scaler.fit_transform(df[feats].fillna(0))
            km = KMeans(n_clusters=3, random_state=42, n_init=10)
            df["state"] = km.fit_predict(X)
            pir_rank = df.groupby("state")["pir_count"].mean().rank()
            state_map = {s: ("Away" if pir_rank[s]==1 else "Active at Home" if pir_rank[s]==3 else "Resting") for s in range(3)}
            df["state_label"] = df["state"].map(state_map)
            results["routine"] = df[["timestamp","hour","dow","state_label"]].copy()
            results["routine_pct"] = df["state_label"].value_counts(normalize=True)*100
            results["routine_peak"] = df[df["state_label"]=="Active at Home"].groupby("hour").size().idxmax() if (df["state_label"]=="Active at Home").any() else 0
            print(f"[ML] Routine: {dict(results['routine_pct'].round(1))}")
    except Exception as e:
        print(f"[ML] Routine failed: {e}")

    # Feature 2: Random Forest home-arrival predictor
    try:
        if "state_label" in df.columns:
            # target: will the occupant be home (Active) within the next hour?
            df["home_soon"] = 0
            for i in range(len(df)-12):
                if "Active at Home" in df["state_label"].iloc[i+1:i+13].values:
                    df.loc[df.index[i], "home_soon"] = 1

            wx_feats = ["temp","humidity","clouds","wind_speed","rain_1h","sin_hour","cos_hour","sin_dow","cos_dow"]
            available = [f for f in wx_feats if f in df.columns]
            if len(available) >= 5:
                Xh = df[available].fillna(0).values
                yh = df["home_soon"].values
                tscv = TimeSeriesSplit(n_splits=5)
                rf = RandomForestClassifier(n_estimators=200, random_state=42)
                acc = cross_val_score(rf, Xh, yh, cv=tscv, scoring="accuracy").mean()
                base = max(yh.mean(), 1-yh.mean())
                rf.fit(Xh, yh)
                imp = pd.Series(rf.feature_importances_, index=available).sort_values(ascending=False)

                # build hourly probability curves for weekday vs weekend
                hours = np.arange(24)
                med_vals = df[["temp","humidity","clouds","wind_speed","rain_1h"]].median().values if all(c in df.columns for c in ["temp","humidity","clouds","wind_speed","rain_1h"]) else np.zeros(5)
                sin_h = np.sin(2*np.pi*hours/24)
                cos_h = np.cos(2*np.pi*hours/24)
                wkday_sin, wkday_cos = np.sin(2*np.pi*1/7), np.cos(2*np.pi*1/7)
                wkend_sin, wkend_cos = np.sin(2*np.pi*6/7), np.cos(2*np.pi*6/7)
                X_wkday = np.c_[np.tile(med_vals,(24,1)), sin_h, cos_h, np.full(24,wkday_sin), np.full(24,wkday_cos)]
                X_wkend = np.c_[np.tile(med_vals,(24,1)), sin_h, cos_h, np.full(24,wkend_sin), np.full(24,wkend_cos)]

                if X_wkday.shape[1] == len(available):
                    prob_wkday = rf.predict_proba(X_wkday)[:,1]
                    prob_wkend = rf.predict_proba(X_wkend)[:,1]
                else:
                    prob_wkday = np.zeros(24)
                    prob_wkend = np.zeros(24)

                results["heating"] = {"acc": acc, "base": base, "imp": imp, "prob_wkday": prob_wkday, "prob_wkend": prob_wkend}
                print(f"[ML] Heating: acc={acc:.1%}, baseline={base:.1%}")
    except Exception as e:
        print(f"[ML] Heating failed: {e}")

    # Feature 3: WHO noise benchmarking
    try:
        df["daytime_flag"] = ((df["hour"]>=7)&(df["hour"]<23)).astype(int)
        df["sleep_noise"]  = ((df["night_window"]==1)&(df["avg_db"]>WHO_SLEEP)).astype(int)
        df["day_noise"]    = ((df["daytime_flag"]==1)&(df["avg_db"]>WHO_DAY)).astype(int)
        daily = df.groupby("date").agg(
            avg_db=("avg_db","mean"),
            sleep_pct=("sleep_noise", lambda x: x.mean()*100),
            day_noise_pct=("day_noise", lambda x: x.mean()*100),
        ).reset_index()
        daily["day_lbl"] = pd.to_datetime(daily["date"]).dt.strftime("%a %d")
        pct_sleep = (df["avg_db"]>WHO_SLEEP).mean()*100
        pct_day = (df["avg_db"]>WHO_DAY).mean()*100
        hourly_noise = df.groupby("hour")["avg_db"].mean()
        results["noise"] = {"daily": daily, "pct_sleep": pct_sleep, "pct_day": pct_day, "hourly": hourly_noise}
        print(f"[ML] Noise: {pct_sleep:.1f}% above sleep limit, {pct_day:.1f}% above day limit")
    except Exception as e:
        print(f"[ML] Noise failed: {e}")

    # Feature 4: Isolation Forest anomaly detection + rule-based alerts
    try:
        iso_feats = ["avg_db","pir_count","light_pct","temp","wind_speed","rain_1h"]
        avail_iso = [f for f in iso_feats if f in df.columns]
        if len(avail_iso) >= 3:
            iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=200)
            X_a = df[avail_iso].fillna(0).values
            df["anomaly_flag"] = iso.fit_predict(X_a)

            def make_alert(row):
                h = int(row["hour"])
                if row.get("light_pct",0)>8 and row.get("pir_count",0)==0 and 1<=h<=5:
                    return "Lights Left On"
                if row.get("avg_db",0)>34 and row.get("night_window",0)==1:
                    return "Noisy Night"
                if row.get("pir_count",0)>25:
                    return "High Activity"
                if row.get("pir_count",0)>5 and row.get("light_pct",0)<2 and 2<=h<=5:
                    return "Motion in Dark"
                if row.get("anomaly_flag",1)==-1:
                    return "Unusual Pattern"
                return None

            df["alert_type"] = df.apply(make_alert, axis=1)
            alerts = df[df["alert_type"].notna()].copy()
            results["alerts"] = alerts
            print(f"[ML] Alerts: {len(alerts)} total")
    except Exception as e:
        print(f"[ML] Alerts failed: {e}")

    # Feature 5: nightly sleep environment score (out of 100)
    try:
        sleep_w = df[(df["hour"]>=22)|(df["hour"]<8)].copy()
        if len(sleep_w) > 0:
            nightly = sleep_w.groupby("date").agg(
                avg_noise=("avg_db","mean"),
                avg_light=("light_pct","mean"),
                motion_total=("pir_count","sum"),
                n=("avg_db","count"),
            ).reset_index()

            # score components: lower noise/light/motion = better sleep
            nightly["noise_score"]  = ((1-(nightly.avg_noise-nightly.avg_noise.min())/(nightly.avg_noise.max()-nightly.avg_noise.min()+1e-9))*40).clip(0,40)
            nightly["light_score"]  = ((1-nightly.avg_light/(nightly.avg_light.max()+1e-9))*35).clip(0,35)
            nightly["motion_score"] = ((1-nightly.motion_total/(nightly.motion_total.max()+1e-9))*25).clip(0,25)
            nightly["total"]        = (nightly.noise_score+nightly.light_score+nightly.motion_score).round(1)
            nightly["day_lbl"]      = pd.to_datetime(nightly["date"]).dt.strftime("%a %d %b")

            def grade(s):
                if s>=75: return "Excellent"
                if s>=55: return "Good"
                if s>=35: return "Fair"
                return "Poor"
            nightly["grade"] = nightly["total"].apply(grade)
            results["sleep"] = nightly
            print(f"[ML] Sleep: avg {nightly['total'].mean():.0f}/100")
    except Exception as e:
        print(f"[ML] Sleep failed: {e}")

    return results


# ---------- ML figure builders ----------

def fig_routine(ml):
    """Pie chart of time split + hourly stacked bar of behavioural states."""
    if "routine" not in ml: return empty_fig("COMPUTING ROUTINE...", 300)
    rdf = ml["routine"]
    pct = ml["routine_pct"]
    labels = ["Away","Resting","Active at Home"]
    colors = {"Away": TEAL_C, "Resting": VIOLET_C, "Active at Home": AMBER_C}

    fig = make_subplots(rows=1, cols=2, specs=[[{"type":"domain"},{"type":"xy"}]],
                        column_widths=[0.35, 0.65], horizontal_spacing=0.08)
    vals = [pct.get(l, 0) for l in labels]
    fig.add_trace(go.Pie(labels=labels, values=vals,
                         marker=dict(colors=[colors[l] for l in labels], line=dict(color="#0F0A06", width=2)),
                         textinfo="label+percent", textfont=dict(size=10, color=CREAM_C),
                         hole=0.35), row=1, col=1)

    hourly = rdf.groupby(["hour","state_label"]).size().unstack(fill_value=0).reindex(columns=labels, fill_value=0)
    hourly_pct = hourly.div(hourly.sum(axis=1), axis=0)*100
    for lbl in labels:
        if lbl in hourly_pct:
            fig.add_trace(go.Bar(x=list(range(24)), y=hourly_pct[lbl].values, name=lbl,
                                 marker_color=colors[lbl], opacity=0.85), row=1, col=2)

    layout_r = {k:v for k,v in PLOT_BG.items() if k != "showlegend"}
    fig.update_layout(**layout_r, height=300, barmode="stack", showlegend=True,
                      legend=dict(font=dict(size=9, color=CREAM_C), bgcolor="rgba(0,0,0,0)", x=0.65, y=0.1))
    fig.update_xaxes(title_text="Hour of Day", row=1, col=2, dtick=2)
    fig.update_yaxes(title_text="State %", row=1, col=2)
    return fig


def fig_heating(ml):
    """Weekday/weekend home probability curves + feature importance bars."""
    if "heating" not in ml: return empty_fig("COMPUTING PREDICTION...", 280)
    h = ml["heating"]
    hours = list(range(24))

    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45], horizontal_spacing=0.1)
    fig.add_trace(go.Scatter(x=hours, y=h["prob_wkday"], mode="lines", name="Weekday",
                             line=dict(color=AMBER_C, width=2), fill="tozeroy",
                             fillcolor="rgba(245,166,35,0.1)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=hours, y=h["prob_wkend"], mode="lines", name="Weekend",
                             line=dict(color=TEAL_C, width=2), fill="tozeroy",
                             fillcolor="rgba(46,134,171,0.1)"), row=1, col=1)
    fig.add_hline(y=0.5, line_dash="dash", line_color=RED_C, opacity=0.5, row=1, col=1)

    imp = h["imp"].sort_values(ascending=True)
    fig.add_trace(go.Bar(y=imp.index.tolist(), x=imp.values, orientation="h",
                         marker_color=[TEAL_C if "dow" in f else AMBER_C if "hour" in f else CORAL_C for f in imp.index]),
                  row=1, col=2)

    layout_h = {k:v for k,v in PLOT_BG.items() if k != "showlegend"}
    fig.update_layout(**layout_h, height=280, showlegend=True,
                      legend=dict(font=dict(size=9, color=CREAM_C), bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(title_text="Hour of Day", row=1, col=1, dtick=2)
    fig.update_yaxes(title_text="P(home in 60 min)", row=1, col=1)
    fig.update_xaxes(title_text="Feature Importance", row=1, col=2)
    return fig


def fig_noise_report(ml):
    """Daily average dB bars + 24-hour noise profile, both with WHO threshold lines."""
    if "noise" not in ml: return empty_fig("COMPUTING NOISE REPORT...", 280)
    n = ml["noise"]
    daily = n["daily"]
    hourly = n["hourly"]

    fig = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5], horizontal_spacing=0.1)

    bar_colors = [GREEN_C if v < WHO_SLEEP else AMBER_C if v < WHO_DAY else RED_C for v in daily["avg_db"]]
    fig.add_trace(go.Bar(x=daily["day_lbl"], y=daily["avg_db"], marker_color=bar_colors,
                         hovertemplate="<b>%{x}: %{y:.1f} dB</b><extra></extra>"), row=1, col=1)
    fig.add_hline(y=WHO_SLEEP, line_dash="dash", line_color=VIOLET_C, opacity=0.6, row=1, col=1,
                  annotation_text=f"Sleep {WHO_SLEEP}dB", annotation_font=dict(size=8, color=VIOLET_C))
    fig.add_hline(y=WHO_DAY, line_dash="dash", line_color=RED_C, opacity=0.6, row=1, col=1,
                  annotation_text=f"Day {WHO_DAY}dB", annotation_font=dict(size=8, color=RED_C))

    h_colors = [RED_C if (h>=23 or h<7) and v>WHO_SLEEP else AMBER_C if v>WHO_DAY else GREEN_C for h, v in zip(hourly.index, hourly.values)]
    fig.add_trace(go.Bar(x=list(hourly.index), y=hourly.values, marker_color=h_colors,
                         hovertemplate="<b>%{x}:00: %{y:.1f} dB</b><extra></extra>"), row=1, col=2)
    fig.add_hline(y=WHO_SLEEP, line_dash="dash", line_color=VIOLET_C, opacity=0.6, row=1, col=2)
    fig.add_hline(y=WHO_DAY, line_dash="dash", line_color=RED_C, opacity=0.6, row=1, col=2)

    layout_n = {k:v for k,v in PLOT_BG.items() if k != "showlegend"}
    fig.update_layout(**layout_n, height=280, showlegend=False)
    fig.update_xaxes(title_text="Day", row=1, col=1, tickangle=30)
    fig.update_yaxes(title_text="Avg dB", row=1, col=1)
    fig.update_xaxes(title_text="Hour", row=1, col=2, dtick=2)
    fig.update_yaxes(title_text="Avg dB", row=1, col=2)
    return fig


def fig_alert_log(ml):
    """Alert type counts + hourly distribution of when alerts fire."""
    if "alerts" not in ml: return empty_fig("COMPUTING ALERTS...", 280)
    alerts = ml["alerts"]
    if alerts.empty: return empty_fig("NO ALERTS DETECTED", 280)

    fig = make_subplots(rows=1, cols=2, column_widths=[0.45, 0.55], horizontal_spacing=0.1)
    ac = alerts["alert_type"].value_counts()
    type_colors = {"Lights Left On":RED_C, "Noisy Night":AMBER_C, "High Activity":GREEN_C,
                   "Motion in Dark":VIOLET_C, "Unusual Pattern":MUTED_C}
    fig.add_trace(go.Bar(y=ac.index.tolist(), x=ac.values, orientation="h",
                         marker_color=[type_colors.get(t, TEAL_C) for t in ac.index],
                         text=ac.values, textposition="outside",
                         textfont=dict(color=CREAM_C, size=10),
                         hovertemplate="<b>%{y}: %{x}</b><extra></extra>"), row=1, col=1)

    ah = alerts.groupby(alerts["timestamp"].dt.hour).size()
    fig.add_trace(go.Bar(x=list(ah.index), y=ah.values,
                         marker_color=[RED_C if h>=23 or h<7 else AMBER_C for h in ah.index],
                         hovertemplate="<b>%{x}:00: %{y} alerts</b><extra></extra>"), row=1, col=2)

    layout_a = {k:v for k,v in PLOT_BG.items() if k != "showlegend"}
    fig.update_layout(**layout_a, height=280, showlegend=False)
    fig.update_xaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Hour", row=1, col=2, dtick=2)
    fig.update_yaxes(title_text="Alerts", row=1, col=2)
    return fig


def fig_sleep_score(ml):
    """Nightly sleep scores with grade colours + stacked component breakdown."""
    if "sleep" not in ml: return empty_fig("COMPUTING SLEEP SCORES...", 280)
    nightly = ml["sleep"]

    fig = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5], horizontal_spacing=0.1)

    def grade_color(g):
        return {"Excellent": GREEN_C, "Good": TEAL_C, "Fair": AMBER_C, "Poor": RED_C}.get(g, MUTED_C)

    bar_colors = [grade_color(g) for g in nightly["grade"]]
    fig.add_trace(go.Bar(x=nightly["day_lbl"], y=nightly["total"], marker_color=bar_colors,
                         text=[f"{s:.0f}" for s in nightly["total"]], textposition="outside",
                         textfont=dict(color=CREAM_C, size=9),
                         hovertemplate="<b>%{x}: %{y:.0f}/100</b><extra></extra>"), row=1, col=1)
    for val, col in [(75, GREEN_C), (55, TEAL_C), (35, AMBER_C)]:
        fig.add_hline(y=val, line_dash="dash", line_color=col, opacity=0.4, row=1, col=1)

    fig.add_trace(go.Bar(x=nightly["day_lbl"], y=nightly["noise_score"], name="Noise (40)", marker_color=AMBER_C, opacity=0.85), row=1, col=2)
    fig.add_trace(go.Bar(x=nightly["day_lbl"], y=nightly["light_score"], name="Light (35)", marker_color=CORAL_C, opacity=0.85), row=1, col=2)
    fig.add_trace(go.Bar(x=nightly["day_lbl"], y=nightly["motion_score"], name="Motion (25)", marker_color=TEAL_C, opacity=0.85), row=1, col=2)

    layout_s = {k:v for k,v in PLOT_BG.items() if k != "showlegend"}
    fig.update_layout(**layout_s, height=280, barmode="stack", showlegend=True,
                      legend=dict(font=dict(size=9, color=CREAM_C), bgcolor="rgba(0,0,0,0)", x=0.55, y=0.95))
    fig.update_xaxes(title_text="Night", row=1, col=1, tickangle=30)
    fig.update_yaxes(title_text="Score / 100", row=1, col=1, range=[0, 115])
    fig.update_xaxes(title_text="Night", row=1, col=2, tickangle=30)
    fig.update_yaxes(title_text="Points", row=1, col=2)
    return fig


# ---------- Page layout ----------

app.layout = html.Div([
    html.Div(className="dashboard-wrapper", children=[

        # header
        html.Div(className="header", children=[
            html.Div(children=[
                html.Div("ELEC70126  IoT and Applications", className="header-eyebrow"),
                html.H1("RoomPulse", className="header-title"),
                html.Div("Multi-Modal Indoor Environment Monitor", className="header-sub"),
            ]),
            html.Div(className="header-right", children=[
                html.Div(id="live-badge"),
                html.Div(id="last-updated", className="sync-time"),
            ]),
        ]),

        html.Div(id="weather-banner"),

        # time window selector
        html.Div(className="controls-bar", children=[
            html.Span("WINDOW:", className="control-label"),
            dcc.Dropdown(id="hours-selector",
                options=[
                    {"label":"Last 24 hours","value":24},
                    {"label":"Last 48 hours","value":48},
                    {"label":"Last 7 days","value":168},
                    {"label":"Last 2 weeks","value":336},
                ],
                value=336, clearable=False, style={"width":"200px"}),
            dcc.Interval(id="auto-refresh", interval=REFRESH_MS, n_intervals=0),
        ]),

        html.Div(id="stats-row", className="stats-grid"),
        html.Div(className="section-divider"),

        # sound time-series
        html.Div(className="chart-card", children=[
            html.Div("SIGNAL STREAM", className="chart-eyebrow"),
            html.Div("Acoustic Activity", className="chart-title"),
            html.Div("Hourly average sound level from MEMS I²S microphone", className="chart-desc"),
            dcc.Graph(id="sound-fig", config={"displayModeBar": False}),
        ]),

        # PIR + light side by side
        html.Div(className="two-col", children=[
            html.Div(className="chart-card", children=[
                html.Div("OCCUPANCY SIGNAL", className="chart-eyebrow"),
                html.Div("PIR Motion Events / Hour", className="chart-title"),
                html.Div("HC-SR501 interrupt-counted motion events", className="chart-desc"),
                dcc.Graph(id="pir-fig", config={"displayModeBar": False}),
            ]),
            html.Div(className="chart-card", children=[
                html.Div("PHOTOMETRIC SIGNAL", className="chart-eyebrow"),
                html.Div("Ambient Light Level %", className="chart-title"),
                html.Div("LDR voltage divider, 0% dark, 100% bright", className="chart-desc"),
                dcc.Graph(id="light-fig", config={"displayModeBar": False}),
            ]),
        ]),

        # sensor cross-correlations (3 scatter plots)
        html.Div(className="chart-card", children=[
            html.Div("SENSOR CROSS-CORRELATION", className="chart-eyebrow"),
            html.Div("Signal Pair Relationships", className="chart-title"),
            html.Div("Pearson r coefficient measures linear correlation strength. |r| > 0.6 = strong, 0.3-0.6 = moderate, < 0.3 = weak", className="chart-desc"),
            html.Div(className="three-col", children=[
                dcc.Graph(id="corr1-fig", config={"displayModeBar": False}),
                dcc.Graph(id="corr2-fig", config={"displayModeBar": False}),
                dcc.Graph(id="corr3-fig", config={"displayModeBar": False}),
            ]),
        ]),

        # indoor vs outdoor weather correlations
        html.Div(className="chart-card", children=[
            html.Div("ENVIRONMENTAL CORRELATION", className="chart-eyebrow"),
            html.Div("Indoor Activity vs Outdoor Conditions", className="chart-title"),
            html.Div("Each point is one sensor reading paired with concurrent OpenWeatherMap data. Collecting since dashboard launched.", className="chart-desc"),
            html.Div(className="three-col", children=[
                dcc.Graph(id="weather-corr1", config={"displayModeBar": False}),
                dcc.Graph(id="weather-corr2", config={"displayModeBar": False}),
                dcc.Graph(id="weather-corr3", config={"displayModeBar": False}),
            ]),
        ]),

        # heatmap + FFT side by side
        html.Div(className="two-col", children=[
            html.Div(className="chart-card", children=[
                html.Div("TEMPORAL DISTRIBUTION", className="chart-eyebrow"),
                html.Div("Sound Level, Hour x Day Heatmap", className="chart-title"),
                html.Div("Average dB SPL by hour of day and day of week", className="chart-desc"),
                dcc.Graph(id="heatmap-fig", config={"displayModeBar": False}),
            ]),
            html.Div(className="chart-card", children=[
                html.Div("SPECTRAL ANALYSIS", className="chart-eyebrow"),
                html.Div("FFT Dominant Signal Periodicities", className="chart-title"),
                html.Div("Frequency decomposition of acoustic signal. 24H and 7D markers shown", className="chart-desc"),
                dcc.Graph(id="fft-fig", config={"displayModeBar": False}),
            ]),
        ]),

        # ---------- ML intelligence section ----------

        html.Div(className="section-divider"),
        html.H2("Smart Living Intelligence", className="section-header"),
        html.Div("Machine learning applied to sensor data: clustering, prediction, health benchmarks, anomaly detection, and sleep scoring", className="section-sub"),

        html.Div(id="ml-stats-row", className="stats-grid"),

        html.Div(className="chart-card", children=[
            html.Div("FEATURE 1: ROUTINE MIRROR", className="chart-eyebrow"),
            html.Div("Your Behavioural States", className="chart-title"),
            html.Div("KMeans clustering on sound, motion, and light discovers three states. No labels, no manual input", className="chart-desc"),
            dcc.Graph(id="ml-routine-fig", config={"displayModeBar": False}),
        ]),

        html.Div(className="chart-card", children=[
            html.Div("FEATURE 2: SMART HEATING PREDICTOR", className="chart-eyebrow"),
            html.Div("When Should Your Flat Pre-Heat?", className="chart-title"),
            html.Div("Random Forest trained on weather + time to predict home arrival. Weekday vs weekend probability curves", className="chart-desc"),
            dcc.Graph(id="ml-heating-fig", config={"displayModeBar": False}),
        ]),

        html.Div(className="chart-card", children=[
            html.Div("FEATURE 3: NOISE EXPOSURE HEALTH REPORT", className="chart-eyebrow"),
            html.Div("WHO Noise Guidelines Benchmark", className="chart-title"),
            html.Div(f"WHO Environmental Noise Guidelines: sleep disruption above {WHO_SLEEP} dB, daytime stress above {WHO_DAY} dB", className="chart-desc"),
            dcc.Graph(id="ml-noise-fig", config={"displayModeBar": False}),
        ]),

        html.Div(className="chart-card", children=[
            html.Div("FEATURE 4: PLAIN-ENGLISH ALERT LOG", className="chart-eyebrow"),
            html.Div("Anomaly Detection & Smart Alerts", className="chart-title"),
            html.Div("Isolation Forest detects anomalous sensor combinations. Each event translated into an actionable alert", className="chart-desc"),
            dcc.Graph(id="ml-alerts-fig", config={"displayModeBar": False}),
        ]),

        html.Div(className="chart-card", children=[
            html.Div("FEATURE 5: SLEEP ENVIRONMENT SCORER", className="chart-eyebrow"),
            html.Div("Nightly Sleep Quality Score", className="chart-title"),
            html.Div("Each night scored out of 100: noise (40 pts), light (35 pts), motion (25 pts)", className="chart-desc"),
            dcc.Graph(id="ml-sleep-fig", config={"displayModeBar": False}),
        ]),

        # hidden div to satisfy the callback output (footer was removed)
        html.Div(id="footer-record-count", style={"display":"none"}),
    ]),
])


# ---------- Main callback ----------
# fires on page load, dropdown change, and every REFRESH_MS

@app.callback(
    Output("sound-fig","figure"), Output("pir-fig","figure"), Output("light-fig","figure"),
    Output("corr1-fig","figure"), Output("corr2-fig","figure"), Output("corr3-fig","figure"),
    Output("weather-corr1","figure"), Output("weather-corr2","figure"), Output("weather-corr3","figure"),
    Output("heatmap-fig","figure"), Output("fft-fig","figure"),
    Output("stats-row","children"), Output("live-badge","children"),
    Output("last-updated","children"), Output("footer-record-count","children"),
    Output("weather-banner","children"),
    Output("ml-routine-fig","figure"), Output("ml-heating-fig","figure"),
    Output("ml-noise-fig","figure"), Output("ml-alerts-fig","figure"),
    Output("ml-sleep-fig","figure"), Output("ml-stats-row","children"),
    Input("hours-selector","value"), Input("auto-refresh","n_intervals"),
)
def update_all(hours, _n):
    data   = fetch_data(hours)
    status = fetch_status()
    wx     = fetch_weather()
    wxdf   = fetch_weather_history(hours)
    df     = build_df(data)

    def card(ticker, value, unit, cls):
        return html.Div(className=f"stat-card {cls}", children=[
            html.Div(ticker, className="stat-ticker"),
            html.Div(value,  className=f"stat-value {cls}"),
            html.Div(unit,   className="stat-unit"),
        ])

    stats = [
        card("AVG.DB  / SOUND",  f"{df['avg_db'].mean():.1f}"    if not df.empty else "-", "dB SPL",        "amber"),
        card("AVG.LX  / LIGHT",  f"{df['avg_light'].mean():.0f}" if not df.empty else "-", "% Intensity",   "red"),
        card("SUM.PIR / MOTION", f"{int(df['total_pir'].sum())}" if not df.empty else "-", "Motion Events", "cream"),
        card("REC     / TOTAL",  f"{status.get('total_records','-')}",                      "Data Points",   "green"),
    ]

    is_live = status.get("status") == "running"
    badge = html.Div(
        className="live-pill online" if is_live else "live-pill offline",
        children=[
            html.Div(className="live-dot" if is_live else "offline-dot"),
            "LIVE FEED" if is_live else "OFFLINE",
        ],
    )
    updated      = datetime.now(timezone.utc).strftime("SYNC  %Y-%m-%d  %H:%M UTC")
    footer_count = f"{status.get('total_records','-')} SENSOR  ·  {status.get('weather_records','-')} WEATHER RECORDS"

    # weather banner
    if wx:
        weather_banner = html.Div(className="weather-banner", children=[
            html.Div(className="weather-item", children=[html.Div("TEMPERATURE", className="weather-label"), html.Div(f"{wx['temp']}°C", className="weather-value")]),
            html.Div(className="weather-item", children=[html.Div("FEELS LIKE", className="weather-label"), html.Div(f"{wx['feels_like']}°C", className="weather-value")]),
            html.Div(className="weather-item", children=[html.Div("CONDITIONS", className="weather-label"), html.Div(wx["description"].title(), className="weather-value")]),
            html.Div(className="weather-item", children=[html.Div("HUMIDITY", className="weather-label"), html.Div(f"{wx['humidity']}%", className="weather-value")]),
            html.Div(className="weather-item", children=[html.Div("CLOUD COVER", className="weather-label"), html.Div(f"{wx['clouds']}%", className="weather-value")]),
            html.Div(className="weather-item", children=[html.Div("WIND SPEED", className="weather-label"), html.Div(f"{wx['wind_speed']} m/s", className="weather-value")]),
            html.Div(className="weather-item", children=[html.Div("RAIN 1H", className="weather-label"), html.Div(f"{wx.get('rain', 0.0)} mm", className="weather-value")]),
        ])
    else:
        weather_banner = html.Div(className="weather-banner",
                                  children=[html.Div(className="weather-item",
                                                     children=[html.Div("WEATHER UNAVAILABLE", className="weather-label")])])

    # sensor pair correlations (hourly data)
    c1 = empty_fig("NEED MORE DATA", 260) if df.empty or len(df) < 3 else \
         corr_scatter(df["avg_db"].values, df["total_pir"].values, "dB SPL", "PIR Events/hr", AMBER_C)
    c2 = empty_fig("NEED MORE DATA", 260) if df.empty or len(df) < 3 else \
         corr_scatter(df["avg_db"].values, df["avg_light"].values, "dB SPL", "Light %", CORAL_C)
    c3 = empty_fig("NEED MORE DATA", 260) if df.empty or len(df) < 3 else \
         corr_scatter(df["total_pir"].values, df["avg_light"].values, "PIR Events/hr", "Light %", CREAM_C)

    # run all ML features
    ml = compute_ml_features(wxdf)

    ml_routine_fig = fig_routine(ml)
    ml_heating_fig = fig_heating(ml)
    ml_noise_fig   = fig_noise_report(ml)
    ml_alerts_fig  = fig_alert_log(ml)
    ml_sleep_fig   = fig_sleep_score(ml)

    # ML summary stat cards
    ml_stats = []
    if "routine_pct" in ml:
        away_pct = ml["routine_pct"].get("Away", 0)
        ml_stats.append(card("AWAY TIME", f"{away_pct:.0f}%", "of recorded time", "blue"))
    if "heating" in ml:
        ml_stats.append(card("PREDICTION", f"{ml['heating']['acc']:.0%}", "home forecast accuracy", "amber"))
    if "noise" in ml:
        ml_stats.append(card("WHO SLEEP", f"{ml['noise']['pct_sleep']:.1f}%", f"above {WHO_SLEEP} dB limit", "green"))
    if "alerts" in ml:
        ml_stats.append(card("ALERTS", f"{len(ml['alerts'])}", "events detected", "red"))
    if "sleep" in ml:
        avg_score = ml["sleep"]["total"].mean()
        ml_stats.append(card("SLEEP SCORE", f"{avg_score:.0f}", "avg nightly / 100", "violet"))

    while len(ml_stats) < 4:
        ml_stats.append(card("-", "-", "computing...", "cream"))
    ml_stats = ml_stats[:4]

    return (
        fig_sound(df), fig_pir(df), fig_light(df),
        c1, c2, c3,
        fig_wx_sound_vs_temp(wxdf),
        fig_wx_pir_vs_clouds(wxdf),
        fig_wx_light_vs_humidity(wxdf),
        fig_heatmap(df), fig_fft(df),
        stats, badge, updated, footer_count, weather_banner,
        ml_routine_fig, ml_heating_fig, ml_noise_fig, ml_alerts_fig, ml_sleep_fig,
        ml_stats,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)