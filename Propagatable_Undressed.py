import numpy, scipy
#from scipy import special
from numpy import array
from numpy import linalg
# standard crap
import __builtin__ 
import os, sys, time, math, re, random, cmath
import pickle
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
from BosonCorrelationFunction import * 

#
# Does both hp and ph terms. 
#

class UndressedCIS: 
	def __init__(self,Polarization = None):
		print "Getting ready for LIOUVILLE SPACE CIS Propagation... "
		
		if Polarization != None: 
			print "Polarization: ", Polarization
		self.Polarization = Polarization
		
		# these things must be defined by the algebra. 
		self.VectorShape = None
		self.ResidualShape = None
		
		self.SubtractRightClosure = False
		
		self.CISTerms = None  # Guess is provided by CIS. 
		self.UndressedTerms = None
		
#		self.DipoleShape = None
		self.InitAlgebra()
		print "particle-hole 2-TCL Algebra Initalization Complete... "
		self.V0 = StateVector()
		self.Mu0 = None
		self.InitalizeNumerics() # Initalizes BCF. 
		print "particle-hole 2-TCL Initalization Complete... "
		return 
	
	def Energy(self,AStateVector): 
		return AStateVector["r1_ph"][0][1]
	
	def InitAlgebra(self): 
		print "Algebra... "		
		HRho = H_Fermi.clone()
		MaxRho = 1
		Rho = RhosUpTo(MaxRho,2) #second argument is normalization of rho. 
		FermiRho = Rho.ToFermi()
		FermiRho.TermsNotContainingTensor("r1_hh")
		FermiRho.TermsNotContainingTensor("r1_pp")		
		FermiRhoC = FermiRho.MyConjugate()

		print "State vector represented by sectors: "
		for Term in FermiRho: 
			Term.Print()

		print "State vector conjugate represented by sectors: "
		for Term in FermiRhoC: 
			Term.Print()

		#------------------------------		
		#PLP#
		#------------------------------		
		
		rchr = FermiRho.clone()
		rchr.LeftTimes(H_Fermi)
		rchr.LeftTimes(FermiRhoC)
		rchr.NormalOrderDiag([[0,0,0,0,0,0,0]])
		rchr.NoSelfContractPart()
		rchr.ConnectedPart()			
		
		print "r_c hr"
		for term in rchr: 
			term.Print()		
		
		rcrh = H_Fermi.clone()
		rcrh.LeftTimes(FermiRho)
		rcrh.LeftTimes(FermiRhoC)
		rcrh.NormalOrderDiagBigMemory([[0,0,0,0,0,0,0]])
		rcrh.NoSelfContractPart()
		rcrh.ConnectedPart()	
		
		print "rcrh"
		for term in rcrh: 
			term.Print()							
		
		hrrc = FermiRhoC.clone()
		hrrc.LeftTimes(FermiRho)
		hrrc.LeftTimes(H_Fermi)
		hrrc.NormalOrderDiagBigMemory([[0,0,0,0,0,0,0]])
		hrrc.NoSelfContractPart()
		hrrc.ConnectedPart()	
		
		print "hrrc"
		for term in hrrc: 
			term.Print()									

		rhrc = FermiRhoC.clone()
		rhrc.LeftTimes(H_Fermi)
		rhrc.LeftTimes(FermiRho)
		rhrc.NormalOrderDiagBigMemory([[0,0,0,0,0,0,0]])
		rhrc.NoSelfContractPart()
		rhrc.ConnectedPart()	
		
		print "rhrc"
		for term in rhrc: 
			term.Print()				
	
		if (self.SubtractRightClosure):
			self.CISTerms = rchr.clone()
#			self.CISTerms.Subtract(hrrc)
#			self.CISTerms.Subtract(rcrh)
			self.CISTerms.Add(rhrc)
		else: 
			self.CISTerms = rchr.clone()
#			self.CISTerms.Add(hrrc)
#			self.CISTerms.Subtract(rcrh)
			self.CISTerms.Subtract(rhrc)

		print "Overall Phhp"		
		for Term in self.CISTerms: 
			Term.Print()
		
		OutFile = open("./Terms/HermCIS","w")
		pickle.Pickler(OutFile,0).dump(self.CISTerms)
		OutFile.close()		
				
		# UndressedTerms terms ------
		# ------

		if (Params.Undressed):
			self.UndressedTerms = ManyOpStrings() 
			Bt = BathCoupling.clone()
			Bt.GiveTime("t")
			Bs = BathCoupling.clone()
			Bs.GiveTime("s")
			Term1 = None
			Term2 = None 
			Term3 = None 
			Term4 = None
			if (True): # just to scope. 
				print " Pert Term 1 -- Undressed: + B(t)Q{B(s)p}"
				
				cstr = FermiRhoC.clone()
				cstr.Times(Bt)
				cstr.NormalOrderDiag()
				cstr.NoSelfContractPart()		
				cstr.Times(Bs)
				cstr.NormalOrderDiagBigMemory(FermiRho.ClassesIContain())
				cstr.Times(FermiRho)
				cstr.NormalOrderDiagBigMemory([[0,0,0,0,0,0,0]])
				cstr.ConnectedPart()
				cstr.NoSelfContractPart()
				
				strc = Bs.clone()
				strc.Times(Bt)
				strc.NormalOrderDiag()
				strc.NoSelfContractPart()						
				strc.Times(FermiRho)
				strc.NormalOrderDiagBigMemory(FermiRhoC.ClassesIContain())
				strc.Times(FermiRhoC)
				strc.NormalOrderDiagBigMemory([[0,0,0,0,0,0,0]])
				strc.ConnectedPart()
				strc.NoSelfContractPart()
				
				Term1 = cstr.clone()
				if (self.SubtractRightClosure): 
					Term1.Subtract(strc)
				else: 
					Term1.Add(strc)

			if (True): # just to scope. 
			
				ctrs = FermiRhoC.clone()
				ctrs.Times(Bt)
				ctrs.NormalOrderDiag()
				ctrs.NoSelfContractPart()
				ctrs.Times(FermiRho)
				ctrs.NormalOrderDiagBigMemory(Bs.ClassesIContain())
				ctrs.NoSelfContractPart()
				ctrs.Times(Bs)
				ctrs.NormalOrderDiagBigMemory([[0,0,0,0,0,0,0]])
				ctrs.ConnectedPart()
				ctrs.NoSelfContractPart()

				trsc = Bt.clone()
				trsc.Times(FermiRho)
				trsc.NormalOrderDiag()
				trsc.NoSelfContractPart()
				trsc.Times(Bs)
				trsc.NormalOrderDiagBigMemory(FermiRhoC.ClassesIContain())
				trsc.Times(FermiRhoC)
				trsc.NormalOrderDiagBigMemory([[0,0,0,0,0,0,0]])
				trsc.ConnectedPart()
				trsc.NoSelfContractPart()

				Term2 = ctrs.clone()
				if (self.SubtractRightClosure): 
					Term2.Subtract(trsc)
				else: 
					Term2.Add(trsc)
				
			if (True):
				Term3 = ManyOpStrings()
				print " Pert Term 3: + Q{B(s)r}B(t) (just permuting Term2)"
				Term3 = Term2.clone()
				Term3.PermuteTimesOnV()  # This is a total hack. :P 

			if(True):
				print " Pert Term 4: + Q{rB(s)}B(t) "
				crst = FermiRhoC.clone()
				crst.Times(FermiRho) 
				crst.NormalOrderDiag()
				crst.NoSelfContractPart()
				crst.Times(Bs)
				crst.NormalOrderDiagBigMemory(Bt.ClassesIContain())
				crst.Times(Bt)
				crst.NormalOrderDiagBigMemory([[0,0,0,0,0,0,0]])
				crst.ConnectedPart()
				crst.NoSelfContractPart()				

				rstc = FermiRho.clone()
				rstc.Times(Bs)
				rstc.NormalOrderDiag()
				rstc.NoSelfContractPart()
				rstc.Times(Bt)
				rstc.NormalOrderDiagBigMemory(FermiRhoC.ClassesIContain())
				rstc.Times(FermiRhoC)
				rstc.NormalOrderDiagBigMemory([[0,0,0,0,0,0,0]])
				rstc.ConnectedPart()
				rstc.NoSelfContractPart()				

				Term4 = crst.clone()
				if (self.SubtractRightClosure): 
					Term4.Subtract(rstc)
				else: 
					Term4.Add(rstc)				

			self.UndressedTerms = Term1.clone()
			self.UndressedTerms.Subtract(Term2)
			self.UndressedTerms.Subtract(Term3)
			self.UndressedTerms.Add(Term4)
			
			for Term in self.UndressedTerms:
				Term.IncorporateDeltas()
			print "--------------------------------"
			print "undressed terms after"
			print "--------------------------------"
			for Term in self.UndressedTerms:
				Term.Print()

			print " end of undressed terms"

			OutFile = open("./Terms/HermUndressedTerms","w")
			pickle.Pickler(OutFile,0).dump(self.UndressedTerms)
			OutFile.close()

		self.UndressedTerms.AssignBCFandTE() # This caused pickling issues so I'm doing it after.					

		# These are the critical things assumed by the numerical part. 
		self.VectorShape = FermiRho.clone()
		self.ResidualShape = self.CISTerms.clone()
		return 

	def DipoleMoment(self,AVector, Time=0.0): 
		no = Params.nocc
		nv = Params.nvirt
		DressedMu = Integrals["mu_ph"].copy()
		Mu = 0.0
		Mu = numpy.tensordot(AVector["r1_ph"],DressedMu,axes=([0,1],[0,1]))
		if (Params.DirectionSpecific): 
			return Mu
		else : 
			return (1.0/3.0)*numpy.sum(Mu*Mu)

	def AlphaMinusBeta(self,rho1): 
		al = 0.0
		be = 0.0
		for I in Params.occ: 
			if (I in Params.alpha): 
				al += rho1["r1_hh"][I][I]
			else : 
				be += rho1["r1_hh"][I][I]
		for I in Params.virt: 
			if (I in Params.beta): 
				al += rho1["r1_pp"][I][I]
			else : 
				be += rho1["r1_pp"][I][I]			
		return al-be
		
	def InitalizeNumerics(self): 
#		print "Checking Hermitian Symmetry. "
#		Integrals.CheckHermitianSymmetry()
		if (Params.Adiabatic != 2): 
			# ------------------------ 
			# Boson section. 
			# ------------------------ 
			print "Initializing Boson Information... "		
			Bs = BosonInformation(Params)
			Bs.DefaultBosons()
			if (os.path.isfile("./Terms/Bosons")):
				print "Found Existing Bosons, Unpickling them."
				bf = open("./Terms/Bosons","rb")
				UnpickleBos = pickle.Unpickler(bf)
				Bs = UnpickleBos.load()
				bf.close()
				return  
			print "---------------------------------------------"		
			print "Using Boson Correlation Parameters: "
			Bs.Print()
	#		Here we should modify the integrals, but we aren't doing that yet. 
	#		Bs.PlotCorrelationFunction()
			print "Initializing Boson Correlation Functions/Tensors... "
			Temp = BosonInformation(Params)
			Temp.DefaultBosons()
			self.UndressedTerms.MakeBCFs(Temp)
			
		# ------------------------ 
		# prepare some single excitation as the initial guess. 
		# homo->lumo (remembering alpha and beta.)
		# ------------------------ 
		
		# End Perturbative Liouvillian Section. 									
		self.V0=StateVector()
		self.VectorShape.MakeTensorsWithin(self.V0)
		self.V0.Fill(0.0)
		r1_ph = self.V0["r1_ph"]
		r1_hp = self.V0["r1_hp"]		
		if (self.Polarization != None): 
			if self.Polarization == "x":
				r1_ph += Integrals["mu_ph"][:,:,0]
				r1_hp += Integrals["mu_hp"][:,:,0]
			if self.Polarization == "y":
				r1_ph += Integrals["mu_ph"][:,:,1]
				r1_hp += Integrals["mu_hp"][:,:,1]				
			if self.Polarization == "z":
				r1_ph += Integrals["mu_ph"][:,:,2]										
				r1_hp += Integrals["mu_hp"][:,:,2]														
	
		if (self.V0.InnerProduct(self.V0)) != 0.0:
			self.V0.MultiplyScalar(1.0/numpy.sqrt(self.V0.InnerProduct(self.V0)))		

	
		self.Mu0 = self.DipoleMoment(self.V0)
		print self.Mu0
		return 

	def First(self): 
		return self.V0
		
	# now operate -i[H, ] on rho:
	def Step(self,OldState,Time,Field = None, AdiabaticMultiplier = 1.0): 
		NewState = OldState.clone()
		NewState.Fill()
				
		self.VectorShape.EvaluateContributions(self.CISTerms,NewState,OldState)
		NewState.MultiplyScalar(complex(0,-1.0))
		if (Params.Undressed): 
			self.VectorShape.EvaluateContributions(self.UndressedTerms,NewState,self.V0,TimeInformation=Time, MultiplyBy=AdiabaticMultiplier)
		return NewState
