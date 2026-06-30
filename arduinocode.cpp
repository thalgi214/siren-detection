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