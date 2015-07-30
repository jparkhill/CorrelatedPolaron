import numpy, scipy
#from scipy import special
from numpy import array
from numpy import linalg
# standard crap
import __builtin__ 
import os, sys, time, math, re, random, cmath
from time import gmtime, strftime
from types import * 
from itertools import izip
from heapq import nlargest
from multiprocessing import Process, Queue, Pipe
from math import pow, exp, cos, sin, log, pi, sqrt, isnan
# local modules written by Moi.
from TensorNumerics import * 
from Wick import * 
from LooseAdditions import * 
from NumGrad import * 
	   
# algebra and numerics for wavefunction CIS(D) propagation. 
# perturbatively updates a CIS(D) vector. 

class Wavfcn_CISD: 
	def __init__(self):
		print "Getting ready for wavefunction CIS(D) Propagation... "
		# these things must be defined by the algebra. 
		self.VectorShape = None 
		self.ResidualShape = None
#		self.DipoleShape = None
		self.InitAlgebra()
		print "Algebra Initalization Complete... "
		self.V0 = StateVector()
		self.InitalizeNumerics()
		print "CIS Initalization Complete... "
		return 
	
	def Energy(self,AStateVector): 
#		print "E ",AStateVector["r1_ph"][0][1]
		return AStateVector["r1_ph"][0][1]
	
	def InitAlgebra(self): 
		T_Gen = ManyOpStrings()
		T_Gen.Add([OperatorString(1.0,[],[[5,0],[6,1]],"T1")])
		T = T_Gen.ToFermi()
		T.QPCreatePart()
		for Term in T: 
			Term.Print()
			
		LeftVac = T.MyConjugate()
		T.NormalOrder(0)
		T.NormalPart(0)
		T.UnContractedPart()
		
		HT = H_Fermi.clone()
		HT.NormalOrder(0)
		HT.NormalPart(0)
		HT.UnContractedPart()
		HT.Times(T)
		HT.NormalOrder(0,LeftVac.ClassesIContain()) # only those which close against leftvac are kept during this process. 
		LeftVac.Times(HT)
		LeftVac.NormalOrder(0,[[0,0,0,0,0,0,0]], False)		
		LeftVac.FullyContractedPart()
		print "CIS Algebra... "
		for Term in LeftVac: 
			Term.Print()
		
		#
		# End CIS part, begin perturbative part. 
		#
			
		print "begining CIS(D) <T+VDVT>  Algebra"	
		Pterm = V_Fermi.clone()
		LeftDV = V_Fermi.clone()
		LeftDVac = T.MyConjugate()

		# T_conj V(D)V (T)
		LeftDV.Times(T)
		print " Pre Contr VT1"
		print "VT1"
		for Term in LeftDV: 
			Term.Print()
		# only doubly excited space should be included here. 
		LeftDV.NormalOrder(0,[[0,0,0,2,2,0,0]], False)
		LeftDV.ConnectedPart()
		print " connected part V*T1"
		for Term in LeftDV: 
			Term.Print()
		LeftDV.Denominate() # all up indices are put in a denominator string. 
		print "********************************************"		
		# only Singly excited space should be included here. 
		Pterm.Times(LeftDV)
		Pterm.NormalOrder(0,[[0,0,0,1,1,0,0]], False)
		Pterm.ConnectedPart()
		print "V.VT1"
		for Term in Pterm: 
			Term.Print()
		print "********************************************"					
		LeftDVac.Times(Pterm)
		LeftDVac.NormalOrder(0,[[0,0,0,0,0,0,0]], False)
		LeftDVac.FullyContractedPart()	
		LeftDVac.ConnectedPart()
		print "<T+V.VT1>_c"
		for Term in LeftDVac: 
			Term.Print()
		print "*************Algebra Complete... *****************"					
		
		# The <VT2Phi_s> term. Is a subset of these terms, with a redefined denominator. 
		VVTterm = self.VVTPart(VVTterm)
		
		self.VectorShape = T.clone()
		self.ResidualShape = LeftVac.clone()
	
		return 
	
	# This is a bit Hacky... There should be no connection between V and R
	# and the denominator should correspond only to the 
	def VVTPart(self,arg): 
		tore = []
		for Term in arg: 
			tore.append(Term)
		return tore

	def DipoleMoment(self,AVector): 
		Mux = numpy.sum(Integrals["mux_hp"]*AVector["r1_hp"]) + numpy.sum(Integrals["mux_ph"]*AVector["r1_ph"]) 
		Mux += numpy.sum(Integrals["mux_hh"]*AVector["r1_hh"]) + numpy.sum(Integrals["mux_pp"]*AVector["r1_pp"]) 
		Muy = numpy.sum(Integrals["muy_hp"]*AVector["r1_hp"]) + numpy.sum(Integrals["muy_ph"]*AVector["r1_ph"]) 
		Muy += numpy.sum(Integrals["muy_hh"]*AVector["r1_hh"]) + numpy.sum(Integrals["muy_pp"]*AVector["r1_pp"]) 
		Muz = numpy.sum(Integrals["muz_hp"]*AVector["r1_hp"]) + numpy.sum(Integrals["muz_ph"]*AVector["r1_ph"]) 
		Muz += numpy.sum(Integrals["muz_hh"]*AVector["r1_hh"]) + numpy.sum(Integrals["muz_pp"]*AVector["r1_pp"]) 
		return (abs(Mux)+abs(Muy)+abs(Muz))


# Explictly Diagonalize the Liouvillian. 
	def ExplicitlyDiagonalize(self): 
		print "explicitly diagonalizing the singles liouvillian ... "
		# Explicitly construct and diagonalize matrix. 
		InVector = StateVector()
		OutVector = StateVector()
		self.VectorShape.MakeTensorsWithin(InVector)
		self.VectorShape.MakeTensorsWithin(OutVector)		
		IndexVector = InVector.AllElementKeys()
		dim = len(IndexVector)
		ToDiag = numpy.zeros((dim,dim),dtype = complex)
		for I in range(dim): 
			InVector.Fill(0.0)
			OutVector.Fill(0.0)			
			(InVector[((IndexVector[I])[0])])[(IndexVector[I])[1]] = 1.0
			self.VectorShape.EvaluateContributions(self.ResidualShape,OutVector,InVector)
			for J in range(dim): 
				ToDiag[I][J] = (OutVector[((IndexVector[J])[0])])[(IndexVector[J])[1]]
		print "Indices: ", IndexVector
		print "Matrix: ", ToDiag
		ToDiag = ToDiag + ToDiag.transpose()
		ToDiag *= 1.0/2.0 
		w,v = numpy.linalg.eig(ToDiag) 
		print "Energies Au: ", w.real
		print "Energies eV: ", sorted(w.real*EvPerAu)
#		print "Zero Eigenvects:" 
#		for i in range(len(w)):
#			if abs((w.real)[i]) < pow (10.0,-5.0): 
#				print " ----- ", w[i] , v[:i]
#				print " ***** "
		print "Significant Energies eV: ", sorted(list(set(tuple([round(E,4) for E in sorted(w.real*EvPerAu) if E > 0.5]))))
		return 

	def InitalizeNumerics(self): 
		return 
		
	def First(self): 
		return self.V0
		
	# now operate -i[H, ] on rho:
	def Step(self,OldState): 
		NewState = OldState.clone()
		NewState.Fill()
		self.VectorShape.EvaluateContributions(self.ResidualShape,NewState,OldState)
		NewState.MultiplyScalar(complex(0,-1.0))
		return NewState
