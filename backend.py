import sqlite3
import time
import requests
import logging
import threading
from datetime import datetime, timezone
from contextlib import contextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("roompulse")

DB_PATH     = "roompulse.db"
OWM_API_KEY = "2336ee800c626251c5e3c1e52a8c5709"
OWM_URL     = "https://api.openweathermap.org/data/2.5/weather?q=Knightsbridge,London,GB&appid={}&units=metric".format(OWM_API_KEY)

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db():
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS readings (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                unix_ts   INTEGER NOT NULL,
                avg_db    REAL, max_db REAL, pir_count INTEGER, light_pct INTEGER, uptime_s INTEGER
            )""")
        db.execute("CREATE INDEX IF NOT EXISTS idx_ts ON readings(unix_ts)")
        # Weather table — one row per sensor packet
        db.execute("""
            CREATE TABLE IF NOT EXISTS weather (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                unix_ts     INTEGER NOT NULL,
                timestamp   TEXT NOT NULL,
                temp        REAL,
                feels_like  REAL,
                humidity    INTEGER,
                clouds      INTEGER,
                wind_speed  REAL,
                rain_1h     REAL,
                description TEXT
            )""")
        db.execute("CREATE INDEX IF NOT EXISTS idx_wx_ts ON weather(unix_ts)")
    log.info(f"Database ready at {DB_PATH}")

def fetch_weather():
    try:
        d = requests.get(OWM_URL, timeout=5).json()
        return {
            "temp":        round(d["main"]["temp"], 1),
            "feels_like":  round(d["main"]["feels_like"], 1),
            "humidity":    d["main"]["humidity"],
            "clouds":      d["clouds"]["all"],
            "wind_speed":  round(d["wind"]["speed"], 1),
            "rain_1h":     d.get("rain", {}).get("1h", 0.0),
            "description": d["weather"][0]["description"],
        }
    except Exception as e:
        log.warning(f"Weather fetch failed: {e}")
        return None

app = FastAPI(title="RoomPulse Backend", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
def startup():
    init_db()
    log.info("RoomPulse backend started.")

class SensorPayload(BaseModel):
    avg_db: float
    max_db: float
    pir:    int
    light:  int
    uptime: int

@app.post("/sensor")
def receive_sensor(payload: SensorPayload):
    now     = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    unix_ts = int(time.time())

    # Save sensor reading
    with get_db() as db:
        db.execute(
            "INSERT INTO readings (timestamp,unix_ts,avg_db,max_db,pir_count,light_pct,uptime_s) VALUES (?,?,?,?,?,?,?)",
            (now, unix_ts, payload.avg_db, payload.max_db, payload.pir, payload.light, payload.uptime)
        )

    log.info(f"Received: avg_db={payload.avg_db} pir={payload.pir} light={payload.light}")

    # Fetch and save weather in background so it never blocks the POST response
    def save_weather(ts, ts_str):
        wx = fetch_weather()
        if wx:
            with get_db() as db:
                db.execute(
                    "INSERT INTO weather (unix_ts,timestamp,temp,feels_like,humidity,clouds,wind_speed,rain_1h,description) VALUES (?,?,?,?,?,?,?,?,?)",
                    (ts, ts_str, wx["temp"], wx["feels_like"], wx["humidity"],
                     wx["clouds"], wx["wind_speed"], wx["rain_1h"], wx["description"])
                )
            log.info(f"Weather saved: {wx['temp']}°C {wx['description']}")

    threading.Thread(target=save_weather, args=(unix_ts, now), daemon=True).start()

    return {"status": "ok", "timestamp": now}

@app.get("/data")
def get_data(hours: int = 168):
    cutoff = int(time.time()) - (hours * 3600)
    with get_db() as db:
        rows = db.execute(
            "SELECT timestamp,unix_ts,avg_db,max_db,pir_count,light_pct FROM readings WHERE unix_ts>=? ORDER BY unix_ts ASC",
            (cutoff,)).fetchall()
    return [dict(r) for r in rows]

@app.get("/data/hourly")
def get_hourly(hours: int = 168):
    cutoff = int(time.time()) - (hours * 3600)
    with get_db() as db:
        rows = db.execute("""
            SELECT strftime('%Y-%m-%dT%H:00:00', timestamp) AS hour_ts,
                   ROUND(AVG(avg_db),2) AS avg_db,
                   ROUND(AVG(light_pct),2) AS avg_light,
                   SUM(pir_count) AS total_pir
            FROM readings WHERE unix_ts>=? GROUP BY hour_ts ORDER BY hour_ts ASC
        """, (cutoff,)).fetchall()
    return [dict(r) for r in rows]

@app.get("/data/weather")
def get_weather_data(hours: int = 168):
    """Return weather records joined with sensor readings by unix_ts proximity."""
    cutoff = int(time.time()) - (hours * 3600)
    with get_db() as db:
        # Join sensor + weather on closest timestamp
        rows = db.execute("""
            SELECT
                r.timestamp, r.avg_db, r.pir_count, r.light_pct,
                w.temp, w.feels_like, w.humidity, w.clouds, w.wind_speed, w.rain_1h, w.description
            FROM readings r
            JOIN weather w ON ABS(r.unix_ts - w.unix_ts) < 60
            WHERE r.unix_ts >= ?
            ORDER BY r.unix_ts ASC
        """, (cutoff,)).fetchall()
    return [dict(r) for r in rows]

@app.get("/status")
def status():
    with get_db() as db:
        count  = db.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
        latest = db.execute("SELECT * FROM readings ORDER BY unix_ts DESC LIMIT 1").fetchone()
        wx_count = db.execute("SELECT COUNT(*) FROM weather").fetchone()[0]
        latest_wx = db.execute("SELECT * FROM weather ORDER BY unix_ts DESC LIMIT 1").fetchone()
    return {
        "status": "running", "total_records": count,
        "latest": dict(latest) if latest else None,
        "weather_records": wx_count,
        "latest_weather": dict(latest_wx) if latest_wx else None,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=False)