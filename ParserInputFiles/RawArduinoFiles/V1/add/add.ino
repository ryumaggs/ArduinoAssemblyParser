#include <stdio.h>

volatile int a = 0xa1a2;
volatile int b = 0xb1b2;
 

void setup() {
  // put your setup code here, to run once:
  // initialize digital pin LED_BUILTIN as an output.
  pinMode(LED_BUILTIN, OUTPUT);
}
void loop(){
  int randNumber1, randNumber2;
  randomSeed(0);
  noInterrupts();
  digitalWrite(LED_BUILTIN, LOW);   
  digitalWrite(LED_BUILTIN, HIGH);   // Trigger with LED
  asm(
    "jmp __main \n"
    );
  asm(
    "__Noops: \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "ret \n"
    );
  asm volatile(
    "__Add:  \n"
    "add %0, %1 \n"
    "add %0, %1 \n"
    "add %0, %1 \n"
    "ret \n"
    : "=d" (a): "d" (b)
  );
  asm volatile(
    "__Mult: \n"
    "muls %0, %1 \n"
    "muls %0, %1 \n"
    "muls %0, %1 \n"
    "ret \n"
    : "=d" (a): "d" (b)
    );
  asm(
    "__Mov: \n"
    "mov %0, %1 \n"
    "mov %0, %1 \n"
    "mov %0, %1 \n"
    "ret \n"
    : "=d" (a): "d" (b)
    );
  asm(
    "__LDST: \n"
    "ldi %0, %1 \n"
    "ldi %0, %1 \n"
    "ldi %0, %1 \n"
    "ret \n"
    :"=d" (a) : "M" (42): "r26", "r27"
    );
  asm(
    "__main:  \n"
    "add %0, %1 \n"
    "add %0, %1 \n"
    "add %0, %1 \n"
    "call __Noops  \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "add %0, %1 \n"
    "add %0, %1 \n"
    "add %0, %1 \n"     
    "call __Add  \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"  
    "add %0, %1 \n"
    "add %0, %1 \n"
    "add %0, %1 \n"
    "call __Mult  \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"  
    "add %0, %1 \n"
    "add %0, %1 \n"
    "add %0, %1 \n"
    "call __Mov \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"  
    "add %0, %1 \n"
    "add %0, %1 \n"
    "add %0, %1 \n"
    "call __LDST \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "nop \n"  
    "nop \n"
    "nop \n"
    "nop \n"
    : "=d" (a): "d" (b)
  );
  interrupts();
}
