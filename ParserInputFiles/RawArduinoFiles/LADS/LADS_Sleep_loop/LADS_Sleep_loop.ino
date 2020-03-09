/*
 * Sleep Loop
 * This sketch is intended to provide a reference execution
 * with low activity. Basically a trigger and then sleep
 */
void setup() {
  // put your setup code here, to run once:
  // initialize digital pin LED_BUILTIN as an output.
  pinMode(LED_BUILTIN, OUTPUT);

}

void loop() {
  // put your main code here, to run repeatedly:
  // Create trigger
  noInterrupts();
  digitalWrite(LED_BUILTIN, LOW);   
  digitalWrite(LED_BUILTIN, HIGH);   // Trigger with LED
  
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay
   asm("nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"); //delay

  interrupts();
}
