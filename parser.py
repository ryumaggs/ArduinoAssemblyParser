import sys
import re

global_regex = re.compile('[0-9a-fA-F]+:')

def load_code(filename):
	curr_pc_int = 0
	prev_pc_int = -1
	code_flag = False
	file = open(filename, "r")

	file_list = []
	line = file.readline()

	while line != "":
		#control statements that will only read the .text (our code) of the input file
		line_split = line.split("\t")
		if line_split[0] == "Disassembly of section .text:\n":
			code_flag = True
		if line_split[0] == "Disassembly of section .bss:\n":
			code_flag = False
		if code_flag == False:
			line = file.readline()
			continue
		potential_pc = line_split[0].strip(' ')
		if global_regex.match(potential_pc) != None:
			#new pc_checker
			string_hex = potential_pc[:-1]
			curr_pc_int = int(string_hex,16)
			# print(curr_pc_int)
			pc_difference = curr_pc_int - prev_pc_int
			if pc_difference > 4:
				num_nops = pc_difference //2
				for i in range(num_nops):
					formated_line = [hex(prev_pc_int+2*(i+1))[2:]+':', '00 00', 'nop']
					curr_pc_int = (prev_pc_int+2*(i+1))
					file_list.append(formated_line)
				prev_pc_int = curr_pc_int
				line = file.readline()
			else:
				line_split[0] = potential_pc
				file_list.append(line_split)
				prev_pc_int = curr_pc_int
				line = file.readline()



			# #old pc checker
			# print(line_split)
			# if ((line_split[0]).strip() == ((hex(pc_hex))[2:]+":")):
			# 	previous_line = (line_split[0]).strip()
			# 	line_split[0] = line_split[0].strip()
			# 	file_list.append(line_split)
			# 	line = file.readline()
			# 	pc_hex += 2
			# elif ((line_split[0]).strip() == ((hex(pc_hex+2))[2:]+":")):
			# 	previous_line = (line_split[0]).strip()
			# 	line_split[0] = line_split[0].strip()
			# 	file_list.append(line_split)
			# 	line = file.readline()
			# 	pc_hex += 4
			# else:
			# 	line = file.readline()
			# 	continue
		else:
			line = file.readline()
			continue
	for item in file_list:
		print(item)

	return file_list

def create_line_dict(code_list):
	# print("creating_line_dict...")
	# for i in code_list:
	# 	print(i)
	ret_d = {}
	index_count = 0
	for line in code_list:
		pc = line[0]
		ret_d[pc] = index_count
		index_count += 1
	return ret_d

def create_instruction_set():
	i_s = {}
	i = ["I"]
	r = ["r"] #use registers
	rc = ["r", "C"] #use register and carry bit
	rs = ["r", "s"] #user register and sram
	rfs = ["r", "f", "s"] #user register, flash, and sram
	zcvh = ["Z", "C", "V", "H"] #zero, carry, ??, ??
	zcvs = ["Z", "C", "V", "S"] #zero, carry, ??, ??
	znvs = ["Z", "N", "V", "S"]
	znv = ["Z", "N", "V"] #zero, ??, ??
	na = [] #none
	#----------------------------------
	#first in the tuple: any hard ware it may use, mainly
	#just checks for register and SRAM

	#second: any flags it may trigger

	#third: number of clock cycles to complete
	i_s["add"] = (r,zcvh, [1])
	i_s["brne"] = (r, [], [1,2])
	i_s["cpc"] = (rc, zcvh, [1])
	i_s["cpi"] = (r,zcvh, [1])
	i_s["sbiw"] = (r,zcvs, [2])
	i_s["eor"] = (r, znv, [1])
	i_s["ori"] = (r, znv, [1])
	i_s["ldi"] = (r, na, [1])
	i_s["in"] = (r, na, [1])
	i_s["out"] = (r, na, [1])
	i_s["lds"] = (rs, na, [2])
	i_s["sts"] = (rs, na, [2])
	i_s["sts"] = (rs, na, [2])
	i_s["jmp"] = (na, na, [3])
	i_s["rjmp"] = (na, na, [2])
	i_s["call"] = (na, na, [5])
	i_s["cli"] = (na, i, [1])
	i_s["sei"] = (na, i, [1])
	i_s["breq"] = (na, na, [1,2])
	i_s["nop"] = (na, na, [2])
	i_s[".word"] = (na, na, [0])
	i_s["sbc"] = (rc, zcvh, [1])
	i_s["and"] = (r, znvs, [1])
	i_s["adc"] = (rc, zcvh, [1])
	i_s["movw"] = (r, na, [1])
	i_s["subi"] = (r, zcvh, [1])
	i_s["sbci"] = (r, zcvh, [1])

	#TODO
	i_s["ld"] = (rfs, na, [2])
	i_s["lpm"] = (rfs,na,[3])
	i_s["st"] = (rfs,na,[1])

	return i_s

def simulate(code_list, line_dict, instruction_set):
	print("---------------------------")
	print("")
	ret_list = []
	index = 0
	register_dict = {"r0":0, "Z":0, "X":0, "Y":0}
	flag_dict = {"Z": 0, "C": 0, "N": 0, "V": 0, "I": 0, "B": 0}
	#Z = zero flag, C = carry bit, N = negative flag, V = twos complement, I = global interrupt
	#B = half carry flag
	sram = {}
	stack = []
	registers = []
	t_n = 0
	# print(instruction_set)
	#user_registers_sram will be a subset of use_register
	while(t_n < 100):
		print("")
		print(t_n,":")
		pc_line = code_list[index]
		command = pc_line[2].replace("\n","")
		print(pc_line)
		ret_list.append(pc_line)
		command_hardware = (instruction_set[command])[0]
		flags_used = (instruction_set[command])[1]
		flash = None
		#separate if statements for commands that dont use registers
		if "r" in command_hardware:
			registers = (pc_line[3]).replace("\n","").replace(" ", "").split(",")

			#initiallizes registers in dict if not already in
			for r in registers:
				if r not in register_dict and r[0] == 'r':
					register_dict[r] = 0


		if "s" in command_hardware:
			mem = (pc_line[3]).replace("\n","").replace(" ", "").split(",")
			for r in mem:
				if r not in sram and r[0] == '0':
					sram[r] = 0

		if "f" in command_hardware:
			args = (pc_line[3]).replace("\n","").replace(" ", "").split(",")
			for f in args:
				if "Z+" in f:
					flash = "Z+"
				elif "-Z" in f:
					flash = "-Z"
				elif "Z" in f:
					flash = "Z"
				elif "Y+" in f:
					flash = "Y+"
				elif "-Y" in f:
					flash = "-Y"
				elif "Y" in f:
					flash = "Y"
				elif "X+" in f:
					flash = "X+"
				elif "-X" in f:
					flash = "-X"
				elif "X" in f:
					flash = "X"
				if not flash is None:
					break

		#--------------individual command code--------------


		#---------- misc --------------
		if command == "nop":
			index += 1

		if command == ".word":
			# Nothing fancy about this directive
			# Allocates space for creating a variable
			index += 1



		#--------------arithmetic---------------------------
		#all but one (SER) arithmetic commands affeect 2 or more registers
		if command == "add":
			result = register_dict[registers[0]] + register_dict[registers[1]]
			register_dict[registers[0]] = result
			check_flags(result, flag_dict,flags_used)
			index += 1

		#ADC
		elif command == "adc":
			result = register_dict[registers[0]] + register_dict[registers[1]]
			check_flags(result, flag_dict,flags_used)
			register_dict[registers[0]] = result + flag_dict["C"]
			index += 1
		#ADIW
		#SUB
		#SUBI
		elif command == "subi":
			result = register_dict[registers[0]] - int(registers[1],0)
			check_flags(result,flag_dict,flags_used)
			register_dict[registers[0]] = result
			index+= 1

		#SBC
		elif command == "sbc":
		#TODO: Check if carry is handled correctly
			result = register_dict[registers[0]] - register_dict[registers[1]]
			check_flags(result, flag_dict,flags_used)
			register_dict[registers[0]] = result - flag_dict["C"]
			index += 1

		#SBCI
		elif command == "sbci":
			result = register_dict[registers[0]] - int(registers[1],0)
			check_flags(result, flag_dict,flags_used)
			register_dict[registers[0]] = result - flag_dict["C"]
			index += 1

		elif command == "sbiw":
			result = register_dict[registers[0]] - int(registers[1],0)
			check_flags(result,flag_dict,flags_used)
			register_dict[registers[0]] = result
			index+= 1

		#AND
		elif command == "and":
				result = register_dict[registers[0]] and register_dict[registers[1]]
				check_flags(result,flag_dict,flags_used)
				register_dict[registers[0]] = result
				index+= 1
		#ANDI
		#OR

		elif command == "ori":
			register_dict[registers[0]] = (register_dict[registers[0]] ^ int(registers[1],0))
			#print(register_dict)
			index += 1

		elif command == "eor":
			register_dict[registers[0]] = (register_dict[registers[0]] ^ register_dict[registers[1]])
			#print(register_dict)
			index += 1

		#com
		#NEG
		#SBR
		#CBR
		#INC
		#DEC
		#TST
		#CLR
		#SER
		#MUL
		#MULS
		#MULSU
		#FMUL
		#FMULS
		#FMULSU

		#--------------branch-------------------------------
		elif command == "rjmp":
			index = rjump(pc_line,line_dict)

		#IJUMP
		#EIJUMP

		elif command == "jmp":
			index = line_dict[(pc_line[3])[2:]+":"]

		#RCALL
		#ICALL
		#EICALL

		elif command == "call":
			index = line_dict[(pc_line[3])[2:]+":"]

		#RET
		#RETI
		#CPSE

		elif command == "cp":
			result = register_dict[registers[0]] - register_dict[registers[1]]
			check_flags(result,flag_dict,flags_used)
			index += 1

		elif command == "cpi":
			result = register_dict[registers[0]] - int(registers[1],0)
			check_flags(result,flag_dict,flags_used)
			index += 1

		elif command == "cpc":
			result = register_dict[registers[0]] - register_dict[registers[1]] - flag_dict["C"]
			check_flags(result,flag_dict,flags_used)
			index += 1

		#SBRC
		#SBRS
		#SBIC
		#SBIS
		#BRBS
		#BRBC
		elif command == "breq":
			if flag_dict["Z"] == 1:
				index = rjump(pc_line,line_dict)
			else:
				index += 1

		elif command == "brne":
			if flag_dict["Z"] == 0:
				index = rjump(pc_line,line_dict)
			else:
				index += 1

		#BRCS
		#BRCC
		#BRSH
		#BRLO
		#BRMI
		#BRPL
		#BRGE
		#BRLT
		#BRHS
		#BRHC
		#BRTS
		#BRTC
		#BRVS
		#BRVC
		#BRIE
		#BRID

		#--------------bit operations-----------------------

		#SBI
		#CBI
		#LSL
		#LSR
		#ROL
		#ROR
		#ASR
		#SWAP
		#BSET
		#BCLR
		#BST
		#BLD
		#SEC
		#CLC
		#SEN
		#CLN
		#SEZ
		#CLZ

		elif command == "sei":
			#this should enable global interrupt flag
			index += 1

		elif command == "cli":
			index += 1

		#ses
		#cls
		#sev
		#clv
		#SET
		#CLT
		#SEH
		#CLH
		#NOP
		elif command == "nop":
			index += 1


		#--------------data transfer------------------------
		#MOV
		#MOVW
		elif command == "movw":
			register_dict[registers[0]] = register_dict[registers[1]]
			rd = int(registers[0][1:])+1
			if rd not in register_dict:
				register_dict[rd] = 0
			rr = int(registers[1][1:])+1
			if rr not in register_dict:
				register_dict[rr] = 0
			register_dict[rd] = register_dict[rr]
			index += 1


		elif command == "ldi":
			register_dict[registers[0]] = int(registers[1],0)
			index += 1

		#LD

		elif command == "ld":
			if flash == "Z+":
				register_dict[registers[0]] = sram[register_dict["Z"]]
				register_dict["Z"] += 1
			elif flash == "-Z":
				register_dict["Z"] -= 1
				register_dict[registers[0]] = sram[register_dict["Z"]]
			elif flash == "Z":
				register_dict[registers[0]] = sram[register_dict["Z"]]
			elif flash == "Y+":
				register_dict[registers[0]] = sram[register_dict["Y"]]
				register_dict["Y"] += 1
			elif flash == "-Y":
				register_dict["Y"] -= 1
				register_dict[registers[0]] = sram[register_dict["Y"]]
			elif flash == "Y":
				register_dict[registers[0]] = sram[register_dict["Y"]]
			elif flash == "X+":
				register_dict[registers[0]] = sram[register_dict["X"]]
				register_dict["X"] += 1
			elif flash == "-X":
				register_dict["X"] -= 1
				register_dict[registers[0]] = sram[register_dict["X"]]
			elif flash == "X":
				register_dict[registers[0]] = sram[register_dict["X"]]
			else:
				#should never happen
				raise ValueError
			# register_dict[registers[0]] = sram[registers[1]]
			index += 1


		#LDD

		elif command == "lds":
			register_dict[registers[0]] = sram[registers[1]]
			index += 1

		#LDS
		#ST
		elif command == "st":
			if flash == "Z+":
				sram[register_dict["Z"]] = register_dict[registers[0]]
				register_dict["Z"] += 1
			elif flash == "-Z":
				register_dict["Z"] -= 1
				sram[register_dict["Z"]] = register_dict[registers[0]]
			elif flash == "Z":
				sram[register_dict["Z"]] = register_dict[registers[0]]
			elif flash == "Y+":
				sram[register_dict["Y"]] = register_dict[registers[0]]
				register_dict["Y"] += 1
			elif flash == "-Y":
				register_dict["Y"] -= 1
				sram[register_dict["Y"]] = register_dict[registers[0]]
			elif flash == "Y":
				sram[register_dict["Y"]] = register_dict[registers[0]]
			elif flash == "X+":
				sram[register_dict["X"]] = register_dict[registers[0]]
				register_dict["X"] += 1
			elif flash == "-X":
				register_dict["X"] -= 1
				sram[register_dict["X"]] = register_dict[registers[0]]
			elif flash == "X":
				sram[register_dict["X"]] = register_dict[registers[0]]
			else:
				#should never happen
				raise ValueError
			index += 1

		#STD

		elif command == "sts":
			sram[registers[0]] = register_dict[registers[1]]
			index += 1

		#LPM
		elif command == "lpm":
			if flash is None:
				register_dict["r0"] = sram[register_dict["Z"]]
			elif flash == "Z+":
				sram[register_dict["Z"]] = register_dict[registers[0]]
				register_dict["Z"] += 1
			elif flash == "Z":
				sram[register_dict["Z"]] = register_dict[registers[0]]
			else:
				#should never happen
				raise ValueError

			index += 1

		#ELPM
		#SPM

		elif command == "in":
			register_dict[registers[0]] = int(registers[1],0)
			index += 1

		elif command == "out":
			index += 1

		#PUSH
		#POP



		print(register_dict)
		t_n += 1
	return ret_list

def rjump(pc_line,line_dict):
	direction_num_no_space = pc_line[3].strip()
	direction = (pc_line[3])[1]
	cur_line_int = ("0x"+(pc_line[0])[0:-1])
	cur_line_int = int(cur_line_int,0)
	amount = int((pc_line[3])[2:])
	new_line_int = 0
	if direction == "-":
		new_line_int = cur_line_int - amount + 2
	elif direction == "+":
		new_line_int = cur_line_int + amount + 2
	else:
		print("unrecognized direction: ", direction)
		exit(1)
	print(hex(new_line_int))
	index = line_dict[hex(new_line_int)[2:]+":"]
	return index


def check_flags(result, flag_dict,flags_used):
	if 'Z' in flags_used:
		if result == 0:
			flag_dict["Z"] = 1
		else:
			flag_dict["Z"] = 0
	if 'C' in flags_used: #since this is a 8-bit micro controller
		if result > 256:
			flag_dict["C"] = 1
		else:
			flag_dict["C"] = 0
	if 'N' in flags_used:
		if result < 0:
			flag_dict["N"] = 1
		else:
			flag_dict["N"] = 0

def save_output(ret_list,filename):
	filename = filename.split('.')[0]
	out_file = open(filename+'OUT.txt','w')
	for i in range(len(ret_list)):
		print(ret_list[i], file=out_file)

def main():
	filename = sys.argv[1]
	print(filename)
	code = load_code(filename)
	d = create_line_dict(code)
	instruction_set = create_instruction_set()
	ret_list = simulate(code,d, instruction_set)
	save_output(ret_list,filename)

main()


#reorganized code for clarity
#WIP: Adding flag edits to commands. began with check_flags() function
