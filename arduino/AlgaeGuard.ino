/*
 * AlgaeGuard — Arduino Sensor Module
 * Algae-Based Carbon Sequestration Monitoring Platform
 * 
 * Reads environmental sensors and sends data via Serial
 * for real-time monitoring of algae cultivation systems.
 * 
 * Sensors:
 *   - Temperature (DS18B20 / analog)
 *   - pH sensor
 *   - Dissolved Oxygen sensor
 *   - Light sensor (LDR / BH1750)
 *   - CO2 sensor (MQ-135 / MH-Z19)
 *   - Turbidity / Water Quality sensor
 *
 * Built for Hackout'26
 */

// ---- Pin Definitions ----
#define TEMP_PIN        A0
#define PH_PIN          A1
#define DO_PIN          A2
#define LIGHT_PIN       A3
#define CO2_PIN         A4
#define TURBIDITY_PIN   A5

// ---- Calibration Constants ----
const float TEMP_OFFSET     = 0.0;
const float PH_OFFSET       = 0.0;
const float PH_SLOPE        = 3.5;   // mV per pH unit (calibrate for your sensor)
const float DO_OFFSET        = 0.0;
const float CO2_BASELINE     = 400.0; // atmospheric CO2 in ppm

// ---- Timing ----
const unsigned long READ_INTERVAL_MS = 2000; // read every 2 seconds
unsigned long lastReadTime = 0;

// ---- Sensor Reading Struct ----
struct SensorData {
  float temperature_c;
  float ph;
  float dissolved_oxygen_mg_l;
  float light_intensity;
  float co2_ppm;
  float water_quality_index;
};

void setup() {
  Serial.begin(9600);
  while (!Serial) { ; } // wait for serial port

  // Configure analog reference
  analogReference(DEFAULT);

  Serial.println(F("AlgaeGuard Sensor Module v1.0"));
  Serial.println(F("Initializing sensors..."));
  delay(1000);
  Serial.println(F("Ready. Streaming data...\n"));
}

void loop() {
  unsigned long now = millis();
  if (now - lastReadTime < READ_INTERVAL_MS) return;
  lastReadTime = now;

  SensorData data = readAllSensors();
  sendDataJSON(data);
}

// ---- Read All Sensors ----
SensorData readAllSensors() {
  SensorData d;
  d.temperature_c         = readTemperature();
  d.ph                    = readPH();
  d.dissolved_oxygen_mg_l = readDissolvedOxygen();
  d.light_intensity       = readLight();
  d.co2_ppm               = readCO2();
  d.water_quality_index   = readWaterQuality();
  return d;
}

// ---- Individual Sensor Reads ----

float readTemperature() {
  int raw = analogRead(TEMP_PIN);
  // Convert analog to voltage, then to °C
  float voltage = raw * (5.0 / 1023.0);
  float tempC = (voltage - 0.5) * 100.0 + TEMP_OFFSET;
  return tempC;
}

float readPH() {
  int raw = analogRead(PH_PIN);
  float voltage = raw * (5.0 / 1023.0);
  float ph = 7.0 + ((2.5 - voltage) / PH_SLOPE) + PH_OFFSET;
  return constrain(ph, 0.0, 14.0);
}

float readDissolvedOxygen() {
  int raw = analogRead(DO_PIN);
  float voltage = raw * (5.0 / 1023.0);
  // Simplified linear mapping (calibrate with known solution)
  float do_mg_l = voltage * 4.0 + DO_OFFSET;
  return max(do_mg_l, 0.0);
}

float readLight() {
  int raw = analogRead(LIGHT_PIN);
  // Map to a 0–1000 lux approximation
  float lux = map(raw, 0, 1023, 0, 1000);
  return lux;
}

float readCO2() {
  int raw = analogRead(CO2_PIN);
  float voltage = raw * (5.0 / 1023.0);
  // Simplified MQ-135 mapping (rough approximation)
  float ppm = CO2_BASELINE + (voltage - 1.0) * 200.0;
  return max(ppm, 0.0);
}

float readWaterQuality() {
  int raw = analogRead(TURBIDITY_PIN);
  // Map turbidity to a 0–100 water quality index (higher = cleaner)
  float wqi = map(raw, 0, 1023, 100, 0);
  return constrain(wqi, 0.0, 100.0);
}

// ---- Output Data as JSON ----
void sendDataJSON(SensorData &d) {
  Serial.print(F("{"));
  Serial.print(F("\"temperature_c\":"));       Serial.print(d.temperature_c, 2);
  Serial.print(F(",\"ph\":"));                 Serial.print(d.ph, 2);
  Serial.print(F(",\"dissolved_oxygen_mg_l\":")); Serial.print(d.dissolved_oxygen_mg_l, 2);
  Serial.print(F(",\"light_intensity\":"));     Serial.print(d.light_intensity, 1);
  Serial.print(F(",\"co2_ppm\":"));             Serial.print(d.co2_ppm, 1);
  Serial.print(F(",\"water_quality_index\":")); Serial.print(d.water_quality_index, 1);
  Serial.print(F(",\"timestamp_ms\":"));        Serial.print(millis());
  Serial.println(F("}"));
}
