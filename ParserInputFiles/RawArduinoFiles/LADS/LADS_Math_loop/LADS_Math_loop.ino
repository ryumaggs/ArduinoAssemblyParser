void setup() {
  // put your setup code here, to run once:
  // initialize digital pin LED_BUILTIN as an output.
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  // Create trigger
  int randNumber1, randNumber2;
  randomSeed(0);
  noInterrupts();
  digitalWrite(LED_BUILTIN, LOW);   
  digitalWrite(LED_BUILTIN, HIGH);   // Trigger with LED
  //Code goes here
  randNumber1 = random(65535);
  randNumber2 = random(65535);
  interrupts();
}
