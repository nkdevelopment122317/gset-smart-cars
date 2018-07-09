#include "configuration.h"

void stopReadingPins() {
  digitalWrite(dir1PinL, LOW);
  digitalWrite(dir2PinL, LOW);
  digitalWrite(dir1PinR, LOW);
  digitalWrite(dir2PinR, LOW);
}


void moveForward(void) {
  digitalWrite(dir1PinL, HIGH);
  digitalWrite(dir2PinL, LOW);
  digitalWrite(dir1PinR, HIGH);
  digitalWrite(dir2PinR, LOW);
}

void initGPIO() {
  pinMode(dir1PinL, OUTPUT); 
  pinMode(dir2PinL, OUTPUT); 
  pinMode(speedPinL, OUTPUT);  
 
  pinMode(dir1PinR, OUTPUT);
  pinMode(dir2PinR, OUTPUT); 
  pinMode(speedPinR, OUTPUT); 
  stopReadingPins();
}

void setMotorSpeed(int leftSpeed, int rightSpeed){
  analogWrite(speedPinL, leftSpeed); 
  analogWrite(speedPinR, rightSpeed);   
}

void setup() {
  initGPIO();
//  moveForward();
//  setMotorSpeed(128, 255);
//  delay(5000);
//  stopReadingPins();
  
}

void loop() {
  Serial.println("starting loop...");
  for (int motorValue = 0; motorValue <= 255; motorValue +=5) {
    moveForward();
    setMotorSpeed(motorValue, motorValue); 
    Serial.println(motorValue);
    delay(30);
  }
  
  for (int motorValue = 255; motorValue >= 0; motorValue -=5) {
    moveForward();
    setMotorSpeed(motorValue, motorValue);
    Serial.println(motorValue);
    delay(30);      
  }

  stopReadingPins();
  
}

