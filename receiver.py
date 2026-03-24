
import serial
import serial.tools.list_ports
import sqlite3
import json
import time
import logging
from datetime import datetime, timezone

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("receiver")

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH    = "roompulse.db"
BAUD_RATE  = 115200
PORT       = "/dev/cu.usbserial-0001"

# ── Database ──────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS readings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            unix_ts     INTEGER NOT NULL,
            avg_db      REAL,
            max_db      REAL,
            pir_count   INTEGER,
            light_pct   INTEGER,
            uptime_s    INTEGER
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ts ON readings(unix_ts)
    """)
    conn.commit()
    conn.close()
    log.info(f"Database ready at {DB_PATH}")


def save_reading(data):
    now     = datetime.now(timezone.utc)
    unix_ts = int(now.timestamp())
    conn    = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO readings
            (timestamp, unix_ts, avg_db, max_db, pir_count, light_pct, uptime_s)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        now.isoformat(),
        unix_ts,
        data.get("avg_db",  0),
        data.get("max_db",  0),
        data.get("pir",     0),
        data.get("light",   0),
        data.get("uptime",  0),
    ))
    conn.commit()
    conn.close()
    log.info(
        f"Saved — dB: {data.get('avg_db')}, "
        f"max: {data.get('max_db')}, "
        f"PIR: {data.get('pir')}, "
        f"Light: {data.get('light')}%, "
        f"Uptime: {data.get('uptime')}s"
    )


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    init_db()

    log.info(f"Connecting to {PORT} at {BAUD_RATE} baud...")

    while True:
        try:
            with serial.Serial(PORT, BAUD_RATE, timeout=10) as ser:
                log.info("Connected — waiting for packets...")
                while True:
                    line = ser.readline().decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    if not line.startswith("{"):
                        continue
                    try:
                        data = json.loads(line)
                        save_reading(data)
                    except json.JSONDecodeError:
                        log.warning(f"Bad JSON: {line}")

        except serial.SerialException as e:
            log.error(f"Serial error: {e}")
            log.info("Retrying in 5 seconds...")
            time.sleep(5)

        except KeyboardInterrupt:
            log.info("Receiver stopped.")
            break


if __name__ == "__main__":
    main()