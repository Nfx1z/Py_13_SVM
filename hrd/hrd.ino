#include <Wire.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Adafruit_MPU6050.h>

// ==============================================================================
// 1. KONFIGURASI KONEKSI WI-FI & SERVER TARGET (UDP)
// ==============================================================================
const char* WIFI_SSID   = "yunde";
const char* WIFI_PASS   = "yunde123";
const char* SERVER_IP   = "10.247.55.149";
const int   SERVER_PORT = 5005; // Port UDP (sesuaikan di server Python)

WiFiUDP udp;
Adafruit_MPU6050 mpu;

// ==============================================================================
// 2. FUNGSI KHUSUS MULTIPLEXER I2C (TCA9548A)
//    Delay dikurangi dari 5ms → 1ms untuk mempercepat switching channel
// ==============================================================================
void selectChannel(uint8_t ch) {
  Wire.beginTransmission(0x70);
  Wire.write(1 << ch);
  Wire.endTransmission();
  delayMicroseconds(500); // 0.5ms, cukup untuk stabilisasi mux
}

// ==============================================================================
// 3. SETUP
// ==============================================================================
void setup() {
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(400000); // I2C Fast Mode (400kHz vs default 100kHz) — 4x lebih cepat

  Serial.println("\n[-] Mencoba menyambungkan ke Wi-Fi...");
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n[✓] Wi-Fi Terhubung!");
  Serial.print("[✓] IP Address ESP32: ");
  Serial.println(WiFi.localIP());

  // Mulai UDP
  udp.begin(4210); // Port lokal ESP32 (bebas)
  Serial.println("[✓] UDP Socket siap.\n");

  // Inisialisasi semua MPU6050
  Serial.println("[-] Konfigurasi sensor MPU6050...");
  for (uint8_t i = 0; i < 4; i++) {
    selectChannel(i);
    Serial.printf("    -> Channel %d: ", i);
    if (!mpu.begin()) {
      Serial.println("[X] GAGAL!");
    } else {
      // Naikkan sample rate untuk data lebih segar
      mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
      mpu.setGyroRange(MPU6050_RANGE_500_DEG);
      mpu.setFilterBandwidth(MPU6050_BAND_94_HZ);
      Serial.println("[✓] OK");
    }
  }
  Serial.println("[✓] Semua sensor siap.\n");
}

// ==============================================================================
// 4. LOOP: SAMPLING + KIRIM LEWAT UDP (NON-BLOCKING, ULTRA-LOW LATENCY)
// ==============================================================================
void loop() {
  // Reconnect guard (non-blocking style)
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[X] Wi-Fi putus, reconnecting...");
    WiFi.reconnect();
    delay(1000);
    return;
  }

  // Buffer char untuk payload — lebih cepat dari String concatenation
  char payload[256];
  int  offset  = 0;
  bool ada_nan = false;

  // --- SAMPLING 4 MPU x 6 SUMBU = 24 NILAI ---
  for (int i = 0; i < 4; i++) {
    selectChannel(i);

    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    float ax = isnan(a.acceleration.x) ? 0.f : a.acceleration.x;
    float ay = isnan(a.acceleration.y) ? 0.f : a.acceleration.y;
    float az = isnan(a.acceleration.z) ? 0.f : a.acceleration.z;
    float gx = isnan(g.gyro.x)         ? 0.f : g.gyro.x;
    float gy = isnan(g.gyro.y)         ? 0.f : g.gyro.y;
    float gz = isnan(g.gyro.z)         ? 0.f : g.gyro.z;

    if (isnan(a.acceleration.x)) ada_nan = true;

    // Tulis langsung ke char buffer (snprintf jauh lebih cepat dari String +)
    offset += snprintf(payload + offset, sizeof(payload) - offset,
                       "%s%.2f,%.2f,%.2f,%.2f,%.2f,%.2f",
                       (i == 0 ? "" : ","), ax, ay, az, gx, gy, gz);
  }

  if (ada_nan) Serial.println("[WARN] Sensor tidak terbaca, pakai 0.00");

  // --- KIRIM PAKET UDP (TIDAK BUTUH HANDSHAKE) ---
  udp.beginPacket(SERVER_IP, SERVER_PORT);
  udp.write((uint8_t*)payload, offset);
  udp.endPacket();

  // Monitoring serial (bisa di-comment untuk performa maksimal)
  Serial.printf("[UDP SENT] %s\n", payload);

  // TIDAK ADA delay() — biarkan loop berjalan secepat hardware mampu
  // Uncomment baris berikut HANYA jika server Python kewalahan menerima data:
  // delayMicroseconds(5000); // 5ms = 200Hz max
}