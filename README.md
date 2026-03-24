# RoomPulse


Three sensors on a Heltec WiFi LoRa 32 V3 board capture sound level, ambient light, and occupancy motion every five minutes. A Python backend stores the data in SQLite alongside weather observations from OpenWeatherMap, and a Dash web dashboard displays everything in real time with built-in machine learning analytics.

## What's in the repo

| File | What it does |
|------|-------------|
| `roompulse_wifi.ino` | ESP32 firmware (Arduino/PlatformIO). Reads the I2S microphone, LDR, and PIR sensor, then POSTs a JSON payload to the backend every 5 minutes over WiFi. |
| `backend.py` | FastAPI server. Receives sensor data, stores it in SQLite, fetches weather from OpenWeatherMap in a background thread. Serves read endpoints for the dashboard. |
| `receiver.py` | Serial receiver (alternative to WiFi). Reads JSON packets from the ESP32 over USB serial and writes them to the same SQLite database. Useful for debugging or when WiFi is unavailable. |
| `dashboard.py` | Dash web app. Top half shows sensor time-series, cross-correlations, weather comparisons, heatmap, and FFT. Bottom half runs five ML features: KMeans clustering, Random Forest occupancy prediction, WHO noise benchmarking, Isolation Forest anomaly detection, and a nightly sleep environment scorer. |
| `roompulse.db` | SQLite database with the collected sensor and weather data. |

## Hardware

- **Board:** Heltec WiFi LoRa 32 V3 (ESP32-S3)
- **Microphone:** INMP441 MEMS (I2S, GPIO5/6/7)
- **Light sensor:** GL5528 LDR (ADC, GPIO1)
- **Motion sensor:** HC-SR501 PIR (GPIO interrupt, GPIO45)
- **Power:** USB 5V from mains adapter

## Setup

### Backend

```bash
pip install fastapi uvicorn requests pydantic
python backend.py
```

Runs on `http://localhost:8000`. The ESP32 firmware sends POST requests to `/sensor`.

### Dashboard

```bash
pip install dash plotly pandas numpy scipy scikit-learn requests
python dashboard.py
```

Opens on `http://localhost:8050`. Pulls data from the backend API.

### Firmware

Open `roompulse_wifi.ino` in PlatformIO or Arduino IDE. Update the WiFi credentials and backend IP address at the top of the file, then flash to the Heltec board.

### Serial receiver (optional)

```bash
pip install pyserial
python receiver.py
```

Connects to the ESP32 over USB serial at 115200 baud. Use this instead of the WiFi path if you want a direct wired connection.

## How it works

1. The ESP32 samples all three sensors continuously. Every 300 seconds it averages the microphone readings, counts PIR events, reads the LDR, and sends a JSON packet to the backend.
2. The backend validates the payload, stores it in SQLite, and fetches the current weather from OpenWeatherMap in a background thread so the ESP32 gets its response back immediately.
3. The dashboard queries the backend API, renders the sensor and weather charts, and runs the ML features on each page load.

## API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/sensor` | POST | Receive a sensor payload from the ESP32 |
| `/data` | GET | Raw sensor records (optional `?hours=` param) |
| `/data/hourly` | GET | Hourly aggregates (avg dB, avg light, total PIR) |
| `/data/weather` | GET | Sensor readings joined with weather by timestamp |
| `/status` | GET | Record counts and latest readings |

## Dashboard ML features

- **Routine Mirror:** KMeans clustering identifies Away, Resting, and Active at Home states
- **Smart Heating Predictor:** Random Forest predicts home arrival from weather and time
- **Noise Health Report:** Benchmarks against WHO 35 dB (sleep) and 45 dB (daytime) guidelines
- **Alert Log:** Isolation Forest anomaly detection plus rule-based alerts (lights left on, noisy nights, etc.)
- **Sleep Scorer:** Nightly score out of 100 based on noise, light, and motion during sleep hours
