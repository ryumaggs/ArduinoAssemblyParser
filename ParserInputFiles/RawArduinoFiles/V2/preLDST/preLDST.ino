#include <stdio.h>

volatile int a = 0xa1a2;
volatile int b = 0xb1b2;
 

void setup() {
  // put your setup code here, to run once:
  // initialize digital pin LED_BUILTIN as an output.
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(9600);
  Serial.println("in setup");
  asm volatile(
    "__Noops1: \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "jmp __main2 \n"
    );
  asm volatile(
    "__Add:  \n"
    "add %0, %1 \n"
    "add %0, %1 \n"
    "add %0, %1 \n"
    "jmp __Noops1 \n"
    : "=d" (a): "d" (b)
    );

  asm volatile(
    "__Noops2: \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "jmp __main3 \n"
    );
  asm volatile(
    "__Mult: \n"
    "muls %0, %1 \n"
    "muls %0, %1 \n"
    "muls %0, %1 \n"
    "jmp __Noops2 \n"
    : "=d" (a): "d" (b)
    );

  asm volatile(
    "__Noops3: \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "jmp __main4 \n"
    );
  asm(
    "__Mov: \n"
    "mov %0, %1 \n"
    "mov %0, %1 \n"
    "mov %0, %1 \n"
    : "=d" (a): "d" (b)
    );

  asm volatile(
    "__Noops4: \n"
    "nop \n"
    "nop \n"
    "nop \n"
    "jmp __main5 \n"
    );
  asm(
    "__LDST: \n"
    "ldi %0, %1 \n"
    "ldi %0, %1 \n"
    "ldi %0, %1 \n"
    :"=d" (a) : "M" (42): "r26", "r27"
    );
}

void loop(){
  int randNumber1, randNumber2;
  Serial.println("top of loop");
  randomSeed(0);
  noInterrupts();
  
  digitalWrite(LED_BUILTIN, LOW);   
  digitalWrite(LED_BUILTIN, HIGH);   // Trigger with LED
    
    //LEFT OFF HERE. BASICALY U NEED TO BREAK THE MAIN FUNCTION INTO PIECES AND 
    //CONTINUALLY JUMP BETWEEN THEM. "RET" SEEMS TO BREAK THE CODE, EVEN WHEN RCALL 
    // IS VISIBLEIN THE ASSEMBLY
    
    asm volatile(
    "__main1: \n"
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
    "ldi %0, %1 \n"
    "ldi %0, %1 \n"
    "ldi %0, %1 \n"
    "call __Add \n"
    :"=d" (a) : "M" (17): "r26", "r27"
    );
    /*: "=d" (a): "d" (b))*/
    asm volatile(
    "__main2: \n"
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
    "ldi %0, %1 \n"
    "ldi %0, %1 \n"
    "ldi %0, %1 \n"
    "call __Mult \n"
    :"=d" (a) : "M" (17): "r26", "r27"
    );
   asm volatile(
    "__main3: \n"
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
    "ldi %0, %1 \n"
    "ldi %0, %1 \n"
    "ldi %0, %1 \n"
    "call __Mov \n"
    :"=d" (a) : "M" (17): "r26", "r27"
    );
    asm volatile(
    "__main4: \n"
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
    "ldi %0, %1 \n"
    "ldi %0, %1 \n"
    "ldi %0, %1 \n"
    "call __LDST \n"
    :"=d" (a) : "M" (17): "r26", "r27"
    );
    asm volatile(
    "__main5: \n"
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
    "nop \n"
    );
  Serial.println("End of loop");
  interrupts();
}
