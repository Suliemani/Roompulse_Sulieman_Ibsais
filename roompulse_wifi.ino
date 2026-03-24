#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <driver/i2s.h>
#include <math.h>

// ── WiFi ──────────────────────────────────────────────────────────────────────
const char* WIFI_SSID     = "TALKTALKAE4D07";
const char* WIFI_PASSWORD = "PAAXXHDB";

// ── Backend ───────────────────────────────────────────────────────────────────
const char* BACKEND_URL = "http://192.168.1.11:8000/sensor";

// ── Pins ──────────────────────────────────────────────────────────────────────
#define I2S_WS    7
#define I2S_SD    6
#define I2S_SCK   5
#define PIR_PIN   45
#define LDR_PIN   1

#define TX_INTERVAL_MS  300000
#define PIR_WARMUP_MS   30000

// ── I2S config ────────────────────────────────────────────────────────────────
#define I2S_PORT        I2S_NUM_0
#define I2S_SAMPLE_RATE 16000
#define I2S_BUFFER_LEN  1024

// ── State ─────────────────────────────────────────────────────────────────────
volatile int  pirCount  = 0;
float         dbAccum   = 0.0f;
float         dbMax     = 0.0f;
int           dbSamples = 0;
unsigned long lastTx    = 0;

void IRAM_ATTR onPIR() {
  if (millis() > PIR_WARMUP_MS) pirCount++;
}

void setupI2S() {
  i2s_config_t cfg = {
    .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate          = I2S_SAMPLE_RATE,
    .bits_per_sample      = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format       = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count        = 4,
    .dma_buf_len          = I2S_BUFFER_LEN,
    .use_apll             = false,
    .tx_desc_auto_clear   = false,
    .fixed_mclk           = 0,
  };
  i2s_pin_config_t pins = {
    .bck_io_num   = I2S_SCK,
    .ws_io_num    = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num  = I2S_SD,
  };
  i2s_driver_install(I2S_PORT, &cfg, 0, NULL);
  i2s_set_pin(I2S_PORT, &pins);
}

void connectWiFi() {
  Serial.print("Connecting to WiFi: ");
  Serial.println(WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected!");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nWiFi FAILED to connect!");
  }
}

float readMicDB() {
  int32_t samples[I2S_BUFFER_LEN];
  size_t  bytesRead = 0;
  i2s_read(I2S_PORT, samples, sizeof(samples), &bytesRead, pdMS_TO_TICKS(100));
  int count = bytesRead / sizeof(int32_t);
  if (count == 0) return 0.0f;
  double sum = 0.0;
  for (int i = 0; i < count; i++) {
    float s = (float)(samples[i] >> 8) / (float)0x7FFFFF;
    sum += s * s;
  }
  float rms = sqrt(sum / count);
  if (rms < 1e-10f) return 0.0f;
  return 20.0f * log10f(rms) + 90.0f;
}

int readLight() {
  int raw = analogRead(LDR_PIN);
  return map(raw, 0, 4095, 0, 100);
}

void transmitPacket() {
  Serial.println("--- Transmitting ---");
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected, reconnecting...");
    connectWiFi();
  }
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Still no WiFi, skipping transmit.");
    return;
  }

  float avgDb = (dbSamples > 0) ? (dbAccum / dbSamples) : 0.0f;
  int   light = readLight();
  int   pir   = pirCount;

  char payload[200];
  snprintf(payload, sizeof(payload),
    "{\"avg_db\":%.1f,\"max_db\":%.1f,\"pir\":%d,\"light\":%d,\"uptime\":%lu}",
    avgDb, dbMax, pir, light, millis() / 1000UL
  );

  Serial.print("Payload: ");
  Serial.println(payload);
  Serial.print("Posting to: ");
  Serial.println(BACKEND_URL);

  HTTPClient http;
  http.begin(BACKEND_URL);
  http.addHeader("Content-Type", "application/json");
  int httpCode = http.POST(payload);

  Serial.print("HTTP response code: ");
  Serial.println(httpCode);

  if (httpCode == 200) {
    Serial.println("SUCCESS — record saved!");
  } else {
    Serial.print("FAILED — error: ");
    Serial.println(http.errorToString(httpCode));
  }
  http.end();

  pirCount  = 0;
  dbAccum   = 0.0f;
  dbMax     = 0.0f;
  dbSamples = 0;
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("RoomPulse booting...");

  pinMode(PIR_PIN, INPUT);
  attachInterrupt(digitalPinToInterrupt(PIR_PIN), onPIR, RISING);

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  setupI2S();

  WiFi.mode(WIFI_STA);
  connectWiFi();

  lastTx = millis();
  Serial.println("Setup complete. Transmitting every 15 seconds.");
}

void loop() {
  float db = readMicDB();
  if (db > 0.0f) {
    dbAccum += db;
    dbSamples++;
    if (db > dbMax) dbMax = db;
  }

  if (millis() - lastTx >= TX_INTERVAL_MS) {
    transmitPacket();
    lastTx = millis();
  }

  delay(10);
}