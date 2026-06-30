const int mic1Pin = A0;
const int mic2Pin = A1;

// 샘플링 주기 설정 (125us = 8000Hz)
const unsigned long sampleInterval = 125; 
unsigned long previousMicros = 0;

void setup() {
  Serial.begin(500000); 
  
  // ADC 속도 향상 (분주비를 128에서 32로 변경)
  bitClear(ADCSRA, ADPS0);
  bitClear(ADCSRA, ADPS1);
  bitSet(ADCSRA, ADPS2);
}

void loop() {
  unsigned long currentMicros = micros();
  
  if (currentMicros - previousMicros >= sampleInterval) {
    previousMicros = currentMicros;
    
    int val1 = analogRead(mic1Pin);
    int val2 = analogRead(mic2Pin);
    
    // CSV 형태로 데이터 전송
    Serial.print(val1);
    Serial.print(",");
    Serial.println(val2);
  }
}