
volatile unsigned long a, b; //4 bytes
volatile unsigned int ii;

void setup() {
  // put your setup code here, to run once:
  //Serial.begin(9600);          //  setup serial
  // initialize digital pin LED_BUILTIN as an output.
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  
  // Create trigger
  noInterrupts();
  digitalWrite(LED_BUILTIN, LOW);   
  digitalWrite(LED_BUILTIN, HIGH);   // Trigger with LED
  //Code goes here
  //for(ii=0; ii<15; ii++){
    strong_loop();
  //}
  
  interrupts();
}

void strong_loop(void){
  for(ii=0; ii<15; ii++){
    a = 0x00000000;
    b = 0xffffffff; 
    a |= b;
    b = 0x00000000;
    a &= b;
    b = 0xffffffff; 
    a = a^b;
  }
}

