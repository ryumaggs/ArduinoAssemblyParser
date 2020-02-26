
volatile unsigned long a, b, x; //4 bytes

void setup() {
  // put your setup code here, to run once:
  //Serial.begin(9600);          //  setup serial
  // initialize digital pin LED_BUILTIN as an output.
  pinMode(LED_BUILTIN, OUTPUT);
  x = 1;
  }

void loop() {
  // put your main code here, to run repeatedly:
  
  // Create trigger
  noInterrupts();
  digitalWrite(LED_BUILTIN, LOW);   
  digitalWrite(LED_BUILTIN, HIGH);   // Trigger with LED
  //Code goes here
  for(unsigned int ii=0; ii<15; ii++){
    a = 0x00000000;
    b = 0xffffffff; 
    a |= b;
    b = 0x00000000;
    if(x){
      //a &= b;
      b = 0xffffffff; 
      //a = a^b;
    }
    a &= b;
    b = 0xffffffff; 
    a = a^b;
    //asm("nop\n"); //delay
  
  }
  
  interrupts();
  //Serial.println(a);             // Traying to force the compiler to run the loop
}
