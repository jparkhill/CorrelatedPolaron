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
	   
#
# Equations for polaron transformed CIS(D)
#


class Equations_Liouville_CIS: 
	def __init__(self):
		# these things must be defined by the algebra. 
		self.VectorShape = None 
		self.ResidualShape = None
#		self.DipoleShape = None
		self.InitAlgebra()
		self.CISTerms = None  # Guess is provided by CIS. 
		self.PTerms = None		
		return 
	
	def InitAlgebra(self): 
		print "Algebra... "		
		HRho = H_Fermi.clone()
		MaxRho = 1
		Rho = RhosUpTo(MaxRho,2) #second argument is normalization of rho. 
		
		FermiRho = Rho.ToFermi()
		
		#Options for how the equations are derived. 
		SubtractRightClosure = True
		QSpaceProjector = False 
		HermitianTheory = True
		HHandPPTerms = False

		if (not HermitianTheory): 
			FermiRho.QPCreatePart()
		elif (not HHandPPTerms): 
			FermiRhoU = Rho.ToFermi()
			FermiRhoD = Rho.ToFermi()
			FermiRhoU.QPCreatePart()
			FermiRhoD.QPAnnPart()
			FermiRho = FermiRhoU.clone()
			FermiRho.Add(FermiRhoD)
		FermiRhoC = FermiRho.MyConjugate()
				
		if (False): 
			print "------------------------"
			print "CLOSURES of operators!"
			print "------------------------"
			LeftClosure = FermiRhoC.clone()		
			LeftClosure.Times(FermiRho)
			LeftClosure.NormalOrder(0)
			LeftClosure.NormalPart(0)
			LeftClosure.FullyContractedPart()
			print "Left Closure: "
			for Term in LeftClosure: 
				Term.Print()
			RightClosure = FermiRho.clone()				
			RightClosure.Times(FermiRhoC)
			RightClosure.NormalOrder(0)
			RightClosure.NormalPart(0)
			RightClosure.FullyContractedPart()
			print "RightClosure: "
			for Term in RightClosure: 
				Term.Print()
		#------------------------------		
		#PLP#
		#------------------------------		
		
		t01 = H_Fermi.clone()
		t01.Times(FermiRho)
		t02 = FermiRho.clone()
		t02.Times(H_Fermi)
		t01.Subtract(t02) # Hr - rH 
		
		r0 = t01.clone()
		r0.NormalOrderDiag(FermiRhoC.ClassesIContain())
		r0.NoSelfContractPart()
		r0.Times(FermiRhoC)
		r0.NormalOrderDiag([[0,0,0,0,0,0,0]])
		r0.NoSelfContractPart()
		r0.ConnectedPart()
		
		l0 = FermiRhoC.clone()
		t01.NormalOrderDiag(FermiRhoC.ClassesIContain())
		t01.NoSelfContractPart()
		l0.Times(t01)
		l0.NormalOrderDiag([[0,0,0,0,0,0,0]])
		l0.NoSelfContractPart()
		l0.ConnectedPart()
		
		CISTerms = l0.clone()
		if SubtractRightClosure: 
			CISTerms.Subtract(r0)
		else : 
			CISTerms.Add(r0)
		print "-----------------------------"
		print "CIS Terms: ", len(CISTerms)
		for Term in CISTerms: 
			Term.Print()			
		print "-----------------------------"
		
		#------------------------------		
		#[V(t)Q[V(s),\Rho]]#
		#------------------------------
		
		VTt = h2_Fermi.clone()
		VTs = h2_Fermi.clone()
		VTt.GiveTime("t")
		VTs.GiveTime("s")
		
		QVsRho = VTs.clone()
		QVsRho.Times(FermiRho)
		QVsRho.NormalOrderDiag()
		QVsRho.NoSelfContractPart()

		if False: 
			print "QVsRho: "
			for Term in QVsRho: 
				Term.Print()
		
		QRhoVs = FermiRho.clone()
		QRhoVs.Times(VTs)
		QRhoVs.NormalOrderDiag()
		QRhoVs.NoSelfContractPart()					
		
		if False: 
			print "QRhoVs: "
			for Term in QRhoVs: 
				Term.Print()
		
		l1=l2=l3=l4=r1=r2=r3=r4=None 
	
		l1 = FermiRhoC.clone()
		l1.Times(VTt)
		l1.NormalOrderDiag(QVsRho.ClassesIContain())
		l1.NoSelfContractPart()
		l1.Times(QVsRho)
		l1.NormalOrderDiag([[0,0,0,0,0,0,0]])
		l1.NoSelfContractPart()			
		l1.ConnectedPart()
		print "-----------------------------"
		print "l1 Terms (Vt Vs r): ", len(l1)
		for Term in l1: 
			Term.Print()			
		print "-----------------------------"
		
		r1 = VTt.clone()
		r1.Times(QVsRho)
		r1.NormalOrderDiag(FermiRhoC.ClassesIContain())
		r1.NoSelfContractPart()
		r1.Times(FermiRhoC)
		r1.NormalOrderDiag([[0,0,0,0,0,0,0]])
		r1.NoSelfContractPart()		
		r1.ConnectedPart()		
		print "-----------------------------"			
		print "r1 Terms (Vt Vs r): ", len(r1)
		for Term in r1: 
			Term.Print()
		print "-----------------------------"			

		l2 = FermiRhoC.clone()
		l2.Times(VTt)
		l2.NormalOrderDiag(QRhoVs.ClassesIContain())
		l2.NoSelfContractPart()
		l2.Times(QRhoVs)
		l2.NormalOrderDiag([[0,0,0,0,0,0,0]])
		l2.NoSelfContractPart()
		l2.ConnectedPart()
		print "-----------------------------"
		print "l2 Terms (Vt r Vs): ", len(l2)
		for Term in l2: 
			Term.Print()			
		print "-----------------------------"			
			
		r2 = VTt.clone()
		r2.Times(QRhoVs)
		r2.NormalOrderDiag(FermiRhoC.ClassesIContain())
		r2.NoSelfContractPart()
		r2.Times(FermiRhoC)
		r2.NormalOrderDiag([[0,0,0,0,0,0,0]])
		r2.NoSelfContractPart()			
		r2.ConnectedPart()
		print "-----------------------------"
		print "r2 Terms (Vt r Vs): ", len(r2)
		for Term in r2: 
			Term.Print()			
		print "-----------------------------"
		
		l3 = FermiRhoC.clone()
		l3.Times(QVsRho)
		l3.NormalOrderDiag(VTt.ClassesIContain())
		l3.NoSelfContractPart()
		l3.Times(VTt)
		l3.NormalOrderDiag([[0,0,0,0,0,0,0]])
		l3.NoSelfContractPart()
		l3.ConnectedPart()
		print "-----------------------------"
		print "l3 Terms (Vs r Vt): ", len(l3)
		for Term in l3: 
			Term.Print()			
		print "-----------------------------"			

		r3 = QVsRho.clone()
		r3.Times(VTt)
		r3.NormalOrderDiag(FermiRhoC.ClassesIContain())
		r3.NoSelfContractPart()
		r3.Times(FermiRhoC)
		r3.NormalOrderDiag([[0,0,0,0,0,0,0]])
		r3.NoSelfContractPart()			
		r3.ConnectedPart()
		print "-----------------------------"
		print "r3 Terms (Vs r Vt): ", len(r3)
		for Term in r3: 
			Term.Print()			
		print "-----------------------------"

		l4 = FermiRhoC.clone()
		l4.Times(QRhoVs)
		l4.NormalOrderDiag(VTt.ClassesIContain())
		l4.NoSelfContractPart()
		l4.Times(VTt)
		l4.NormalOrderDiag([[0,0,0,0,0,0,0]])
		l4.NoSelfContractPart()			
		l4.ConnectedPart()
		print "-----------------------------"			
		print "l4 Terms (r Vs Vt): ", len(l4)
		for Term in l4: 
			Term.Print()
		print "-----------------------------"
		
		print "Other way... "
		r4 = QRhoVs.clone()
		r4.Times(VTt) 		
		r4.NormalOrderDiag(FermiRhoC.ClassesIContain())
		r4.NoSelfContractPart()
		r4.Times(FermiRhoC)
		r4.NormalOrderDiag([[0,0,0,0,0,0,0]])
		r4.NoSelfContractPart()			
		r4.ConnectedPart()
		print "-----------------------------"
		print "r4 Terms (r Vs Vt): ", len(r4)
		for Term in r4: 
			Term.Print()			
		print "-----------------------------"		
		
		self.PTerms = ManyOpStrings()
		if SubtractRightClosure:
			self.PTerms.Add(l1)
			self.PTerms.Subtract(r1)
			self.PTerms.Subtract(l2)
			self.PTerms.Add(r2)
			self.PTerms.Subtract(l3)
			self.PTerms.Add(r3)
			self.PTerms.Add(l4)
			self.PTerms.Subtract(r4)
		else :
			self.PTerms.Add(l1)
			self.PTerms.Add(r1)
			self.PTerms.Subtract(l2)
			self.PTerms.Subtract(r2)
			self.PTerms.Subtract(l3)
			self.PTerms.Subtract(r3)
			self.PTerms.Add(l4)
			self.PTerms.Add(r4)
		#				
		phph = self.PTerms.clone()
		phph.TermsContainingTensor("r1_ph")
		phph.TermsContainingTensor("r1_Vac_ph")
		print "------------- phph",len(phph),"----------------"
		for Term in phph:
			Term.Print()			
		print "-----------------------------"			
		#
		hphp = self.PTerms.clone()
		hphp.TermsContainingTensor("r1_hp")
		hphp.TermsContainingTensor("r1_Vac_hp")		
		print "------------- hphp",len(hphp),"----------------"
		for Term in hphp: 
			Term.Print()			
		print "-----------------------------"
		#
		phhp = self.PTerms.clone()
		phhp.TermsContainingTensor("r1_hp")
		phhp.TermsContainingTensor("r1_Vac_ph")
		print "------------- phhp",len(phhp),"----------------"
		for Term in phhp: 
			Term.Print()			
		print "-----------------------------"
		#
		hpph = self.PTerms.clone()
		hpph.TermsContainingTensor("r1_ph")
		hpph.TermsContainingTensor("r1_Vac_hp")
		print "------------- hpph",len(hpph),"----------------"
		for Term in hpph: 
			Term.Print()			
		print "-----------------------------"

		# These are the critical things assumed by the numerical part. 
		self.VectorShape = FermiRho.clone()
		self.ResidualShape = CISTerms.clone()
	
		return 

