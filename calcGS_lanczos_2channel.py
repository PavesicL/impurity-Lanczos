#!/usr/bin/env python3

from lanczos_diagonalisation import *
import sys
import timeit


if len(sys.argv)!=2:
	print("Usage: {0} inputFile".format(sys.argv[0]))
	exit()

#start timer:
start = timeit.timeit()

inputFile = sys.argv[1]
p=params(inputFile)

p.model = "SIAM_2channel"

calculate_print_and_save(p)

end = timeit.timeit()
print(f"Wall time: {end - start} s")