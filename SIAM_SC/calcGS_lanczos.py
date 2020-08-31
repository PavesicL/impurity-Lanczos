#!/usr/bin/env python3

from SIAM_SC import *
import sys


if len(sys.argv)!=2:
	print("Usage: {0} inputFile".format(sys.argv[0]))
	exit()

inputFile = sys.argv[1]
p=params(inputFile)
printGS(p)