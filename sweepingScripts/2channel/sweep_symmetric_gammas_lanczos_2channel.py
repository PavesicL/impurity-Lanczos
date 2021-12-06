#!/usr/bin/env python3

import sys
import numpy as np
sys.path.insert(1, '../')
from lanczos_diagonalisation import getEs, params, saveToFile

#PARSE INPUT:

if len(sys.argv)!=5:
	print("Usage: {0} inputFile gamma_min gamma_max gamma_step".format(sys.argv[0]))
	exit()

inputFile = sys.argv[1]
start = float(sys.argv[2])
stop = float(sys.argv[3])
step = float(sys.argv[4])


#MAKE LIST OF GAMMAS
gammas = [start]
numsteps = int(round((stop-start)/step, 1))

for i in range(numsteps-1):
	gammas.append(round(start+(i+1)*step, 5))


#RUN CALCULATION

print(gammas)
okay = input(f"Run {len(gammas)} calculations? y/n\n")
if okay == "y":
	
	p=params(inputFile)
	p.model = "SIAM_2channel"

	Es, ns, iis = [], [], []
	for gamma in gammas:

		p.gamma1 = gamma
		p.gamma2 = gamma

		p.V1 = np.sqrt(p.gamma1/(np.pi*p.rho))
		p.V2 = np.sqrt(p.gamma1/(np.pi*p.rho))

		res = getEs(p)

		Es.append([gamma] + res[0])
		ns.append([gamma] + res[1])
		iis.append([gamma] + res[2])


	saveName = f"U{p.U}_Ec1{p.Ec1}_Ec2{p.Ec2}_n01{p.n01}_n02{p.n02}"	
	
	saveToFile(Es, "ZBM_energies_"+saveName)
	saveToFile(ns, "ZBM_ns_"+saveName)
	saveToFile(iis, "ZBM_iis_"+saveName)
