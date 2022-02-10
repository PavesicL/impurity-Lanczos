#!/usr/bin/env python3

from lanczos_diagonalisation import *
import sys
import time

if len(sys.argv)!=2:
	print("Usage: {0} inputFile".format(sys.argv[0]))
	exit()
	
#start timer:
start = time.time()

inputFile = sys.argv[1]
p=params(inputFile)

p.model = "SIAM"

calculate_print_and_save(p)

end = time.time()
print(f"Wall time: {round(end - start, 1)} s")
