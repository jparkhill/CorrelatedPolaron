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
# A TDA-like approximation to the response matrix (studying only ph->ph transformations) 
# This also includes the bath contributions. 
#

class Liouville_CISD: 
	def __init__(self,Polarization = None):
		print "Getting ready for LIOUVILLE SPACE CIS Propagation... "
		
		if Polarization != None: 
			print "Polarization: ", Polarization
		self.Polarization = Polarization
		
		# these things must be defined by the algebra. 
		self.VectorShape = None
		self.ResidualShape = None

		# options related to specifics of the system-bath model 
		self.RenormalizeIntegrals = True 
		if (Params.Adiabatic == 2 or Params.Adiabatic == 1 or Params.ContBath or Params.Undressed) :
			self.RenormalizeIntegrals = False 
		self.SecondOrderDipole = True
		if (Params.Undressed): 
			self.SecondOrderDipole = False			
		
		self.QSpaceProjector = True 
		self.ForceAntiHermitian = True # adds conjugate perturbative terms. 
		
		self.CISTerms = None  # Guess is provided by CIS. 
		self.PTerms = None
		if(Params.Inhomogeneous):
			self.ITerms = None # Inhomogeneous terms, added by tmarkovich, 7/19/2012
		self.UndressedTerms = None

		self.MuTerms = None # These are mu*rho
		self.DebugTerms = None # useful to toss shit in. 
		
		# The CIS states as obtained by diagonalization
		self.CISStates = None
		self.CISEnergies = None
		# In the Markov limit the correlated states can be made exactly
		self.MarkovStates = None
		self.MarkovEnergies = None
		self.MarkovMatrix = None
		self.NonPertMtrx = None		

#		self.DipoleShape = None
		self.InitAlgebra()
		print "particle-hole 2-TCL Algebra Initalization Complete... "
		self.V0 = StateVector()
		self.Mu0 = None
		self.InitalizeNumerics() # Initalizes BCF. 
		print "particle-hole 2-TCL Initalization Complete... "
		print "Running with parameters: "
		# options related to specifics of the system-bath model 
		print "self.RenormalizeIntegrals: ", self.RenormalizeIntegrals 
		print "self.SecondOrderDipole: ", self.SecondOrderDipole 
		print "self.QSpaceProjector: ", self.QSpaceProjector 	
		print "self.ForceAntiHermitian: ", self.ForceAntiHermitian 
		return 
	
	def Energy(self,AStateVector): 
		return AStateVector["r1_ph"][0][1]
	
	def InitAlgebra(self): 
		print "Algebra... "		
		HRho = H_Fermi.clone()
		MaxRho = 1
		Rho = RhosUpTo(MaxRho,2) #second argument is normalization of rho. 
		FermiRho = Rho.ToFermi()
		# Temporarily only do ph->ph 
		FermiRho.QPCreatePart()
		
		# if the files for the algebra exist, just read them in. 
		if (os.path.isfile("./Terms/CIS") and os.path.isfile("./Terms/phphPterm") and os.path.isfile("./Terms/phphMuTerms") and os.path.isfile("./Terms/UndressedTerms")):
			print "Found Existing Terms, Unpickling them."
			cf = open("./Terms/CIS","rb")
			pf = open("./Terms/phphPterm","rb")			
			mf = open("./Terms/phphMuTerms","rb")						
			udf = open("./Terms/UndressedTerms","rb")						
			UnpickleCIS = pickle.Unpickler(cf)
			UnpicklePT = pickle.Unpickler(pf)			
			UnpickleMT = pickle.Unpickler(mf)			
			UnpickleUD = pickle.Unpickler(udf)			
			self.CISTerms = UnpickleCIS.load()
				
			cf.close()
			del cf
			self.MuTerms = UnpickleMT.load()
			mf.close()	
			del mf

			self.PTerms = UnpicklePT.load()
			pf.close()
			del pf
			if(Params.Undressed):
				self.UndressedTerms = UnpickleUD.load()
				udf.close()
				del udf
			
			# Inhomogenous terms ------
			# ------
			if(Params.Inhomogeneous):
				self.ITerms = self.PTerms.clone()
				for Term in self.ITerms: 
					for T in Term.Tensors: 
						if T.Time == "s": 
							T.Time = "i"
			else: 
				self.ITerms = ManyOpStrings()
				
			if(Params.Correlated):				
				self.PTerms.AssignBCFandTE() # This caused pickling issues so I'm doing it after.	
			if(Params.Inhomogeneous):
				self.ITerms.AssignBCFandTE() # This caused pickling issues so I'm doing it after.	
			if(Params.Undressed):
				self.UndressedTerms.AssignBCFandTE() 
								
			if(Params.Correlated):
				if (self.QSpaceProjector): 
					self.PTerms.QPart()
					if(Params.Inhomogeneous):
						self.ITerms.QPart()
				if (self.ForceAntiHermitian): 
					print "Adding time-reversed terms to force antihermiticity."
					Tmp = self.PTerms.AddAntiHermitianCounterpart()
					Tmp = self.ITerms.AddAntiHermitianCounterpart()

				if (Params.SecularApproximation == 1): 
					print "Applying Algebraic Secular Approximation..."
					for Term in self.PTerms: 
						Term.Secularize()
						Term.AddAntiHermitianCounterpart()
					
			print "Using Non-Perturbative terms: "
			for Term in self.CISTerms:
				Term.Print()
		
			if Params.Undressed: 
				print "Using Undressed terms: "
				for Term in self.UndressedTerms:
					Term.Print()				
			
			if Params.latex:
				print "Printing Latex Terms"
				if Params.Undressed:
					for Term in self.UndressedTerms:
						Term.Print(LatexFormat = True)	
				for Term in self.PTerms:
					Term.Print(LatexFormat = True)	

			if Params.Correlated: 
				print "Using Perturbative terms: "
				for Term in self.PTerms:
					Term.Print()
					
			self.VectorShape = FermiRho.clone()
			self.ResidualShape = self.CISTerms.clone()
			return 
		
		RhoH = FermiRho.clone()
		RhoH.NormalOrder(0)
		RhoH.NormalPart(0)
		RhoH.UnContractedPart()			
		HRho.NormalOrder(0)
		HRho.NormalPart(0)
		HRho.UnContractedPart()			
		LeftVac = RhoH.MyConjugate()
		# Since this is a DM expression we also have to project on the Right 
		# and add the two resulting expressions. 
		# (add or subtract.... !!!!!! - JAP 2011) 
		RightVac = LeftVac.clone()
		RhoTemp = RhoH.clone()
		RhoH.Times(HRho)
		HRho.Times(RhoTemp)
		HRho.Subtract(RhoH)
		print " NormalOrdering HRho - RhoH ... "
		HRho.NormalOrder(0,LeftVac.ClassesIContain()) # only those which close against leftvac are kept during this process. 
		LeftVac.Times(HRho)
		HRho.Times(RightVac) 
		print "NormalOrdering LeftVac*(HRho-RhoH) ... "
		LeftVac.NormalOrder(0,[[0,0,0,0,0,0,0]], False)		
		HRho.NormalOrder(0,[[0,0,0,0,0,0,0]], False)		
		LeftVac.Add(HRho) # Note: Adding instead of subtracting the right vaccum part fucked everything up. 
		LeftVac.FullyContractedPart()
		print "*_*_*_*_*_*_*_*__*_*_*_*_*_*_*_*_*"
		print "Using this expression for LCIS: "
	#	for Term in LeftVac: 
	#		Term.Print()
		print "*_*_*_*_*_*_*_*__*_*_*_*_*_*_*_*_*"
		self.CISTerms = LeftVac.clone()
		
		OutFile = open("./Terms/CIS","w")
		pickle.Pickler(OutFile,0).dump(self.CISTerms)
		OutFile.close()
		del OutFile

		FermiRhoC = FermiRho.MyConjugate()
		print "Doing the Mu Terms."
		t01 = mu_Fermi.clone()
		t01.Times(FermiRho)
		l0 = FermiRhoC.clone()
		t01.NormalOrderDiag(FermiRhoC.ClassesIContain())
		t01.NoSelfContractPart()
		l0.Times(t01)
		l0.NormalOrderDiag([[0,0,0,0,0,0,0]])
		l0.NoSelfContractPart()
		l0.ConnectedPart()
		self.MuTerms = l0.clone()
		print "Mu Terms: "
#		for Term in self.MuTerms: 
	#		Term.Print()
		OutFile = open("./Terms/phphMuTerms","w")
		pickle.Pickler(OutFile,0).dump(self.MuTerms)
		OutFile.close()
		del OutFile
		
		print "Begining the Perturbative terms... "		
		self.PTerms = ManyOpStrings()
		self.ITerms = ManyOpStrings()
		if(Params.Undressed):
			self.UndressedTerms = ManyOpStrings()
		
		VTt = h2_Fermi.clone()
		VTs = h2_Fermi.clone()
		VTt.GiveTime("t")
		VTs.GiveTime("s")
		
		if (True): 
			LeftVac1 = FermiRho.MyConjugate()  
			print " Pert Term 1: + V(t)Q{V(s)p}"
			Temp0 = VTt.clone()
			# knowing the diagrams ahead of time. 
			LeftVac1.Times(Temp0)
			LeftVac1.NormalOrderDiag()
			LeftVac1.NoSelfContractPart()					
			Temp1 = VTs.clone()
			Temp1.Times(FermiRho)
			Temp1.NormalOrderDiag(LeftVac1.ClassesIContain(),False)			
			Temp1.NoSelfContractPart()
			LeftVac1.Times(Temp1)
			LeftVac1.NormalOrderDiag([[0,0,0,0,0,0,0]], False)
			LeftVac1.ConnectedPart()
						
			OutFile = open("./Terms/rVtVsr","w")
			pickle.Pickler(OutFile,0).dump(LeftVac1)
			OutFile.close()
			del OutFile

			
#			print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
#			print len(LeftVac1), " Terms from V(t)Q{V(s)p} - left projection. "
#			for Term in LeftVac1: 
#				Term.Print()
#			print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
			# The Right part isn't done yet (it's zero) if only ph excitations. (Generally)

		if (True): 
			LeftVac2 = FermiRho.MyConjugate()  		
			print " Pert Term 2: + V(t)Q{pV(s)}"
			Temp0 = VTt.clone()
			# knowing the diagrams ahead of time. 
			LeftVac2.Times(Temp0)
			LeftVac2.NormalOrderDiag()
			LeftVac2.NoSelfContractPart()
			
			Temp3 = FermiRho.clone()
			Temp3.Times(VTs)			
			Temp3.NormalOrderDiag(LeftVac2.ClassesIContain(),False)			
			Temp3.NoSelfContractPart()
			
			LeftVac2.Times(Temp3)
			LeftVac2.NormalOrderDiag([[0,0,0,0,0,0,0]], False)
			LeftVac2.ConnectedPart()
												
#			print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
#			print len(LeftVac2), " Terms from V(t)Q{pV(s)} - left projection. "
#			for Term in LeftVac2: 
#				Term.Print()
#			print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

		if (True): 
			LeftVac3 = ManyOpStrings()		
			print " Pert Term 3: + Q{V(s)r}V(t) (just permuting Term2)"
			LeftVac3 = LeftVac2.clone()
			LeftVac3.PermuteTimesOnV()  # This is a total hack. :P 
			OutFile = open("./Terms/rVsrVt","w")
			pickle.Pickler(OutFile,0).dump(LeftVac2)
			OutFile.close()
			del OutFile
#			print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
#			print len(LeftVac3), " Terms from V(s)Q{pV(t)} - left projection. "
#			for Term in LeftVac3: 
#				Term.Print()
#			print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"


		if (True): 
			LeftVac4 = ManyOpStrings()
			print "The Fourth term is zero in the TDA."

		# The perturbative terms are subtracted (i/\hbar)^2 [V,[V,\rho]]
		# this is where the expectation values are taken. 	
		if (True):	
			self.PTerms.Subtract(LeftVac1)		
			self.PTerms.Add(LeftVac2)
			self.PTerms.Add(LeftVac3)
		
		# UndressedTerms terms ------
		# ------

		if (Params.Undressed): 
			Bt = BathCoupling.clone()
			Bt.GiveTime("t")
			Bs = BathCoupling.clone()
			Bs.GiveTime("s")

			if (True): # just to scope. 
				print " Pert Term 1 -- Undressed: + B(t)Q{B(s)p}"
				Term1 = FermiRho.MyConjugate()  
				Term1.Times(Bt)
				Term1.NormalOrderDiag()
				Term1.NoSelfContractPart()		

				Temp1 = Bs.clone()
				Temp1.Times(FermiRho)
				Temp1.NormalOrderDiag(Term1.ClassesIContain(),False)			
				Temp1.NoSelfContractPart()

				Term1.Times(Temp1)
				Term1.NormalOrderDiag([[0,0,0,0,0,0,0]], False)
				Term1.ConnectedPart()
				
		#		for Term in Term1: 
		#			Term.Print()
				OutFile = open("./Terms/rBtBsr","w")
				pickle.Pickler(OutFile,0).dump(Term1)
				OutFile.close()
				del OutFile
				self.UndressedTerms.Add(Term1)

			if (True): # just to scope. 
				Term2 = FermiRho.MyConjugate()  		
				print " Pert Term 2 -- Undressed: + B(t)Q{pB(s)}"
				Temp0 = Bt.clone()
				Term2.Times(Temp0)
				Term2.NormalOrderDiag()
				Term2.NoSelfContractPart()
			
				Temp3 = FermiRho.clone()
				Temp3.Times(Bs)			
				Temp3.NormalOrderDiag(Term2.ClassesIContain(),False)			
				Temp3.NoSelfContractPart()
			
				Term2.Times(Temp3)
				Term2.NormalOrderDiag([[0,0,0,0,0,0,0]], False)
				Term2.ConnectedPart()
				
		#		for Term in Term2: 
		#			Term.Print()
				self.UndressedTerms.Subtract(Term2)

			if (True):
				Term3 = ManyOpStrings()
				print " Pert Term 3: + Q{B(s)r}B(t) (just permuting Term2)"
				Term3 = Term2.clone()
				Term3.PermuteTimesOnV()  # This is a total hack. :P 
				OutFile = open("./Terms/rBsrBt","w")
				pickle.Pickler(OutFile,0).dump(Term2)
				OutFile.close()
				del OutFile
				self.UndressedTerms.Subtract(Term3)

			if(True):
				print "Term 4 is zero according to the TDA"
			for Term in self.UndressedTerms:
				Term.IncorporateDeltas()
			print "--------------------------------"
			print "undressed terms after"
			print "--------------------------------"
			for Term in self.UndressedTerms:
				Term.Print()

			print " end of undressed terms"

			OutFile = open("./Terms/UndressedTerms","w")
			pickle.Pickler(OutFile,0).dump(self.UndressedTerms)
			OutFile.close()
			del OutFile

			#From there, we then need to add the antihermitian counterparts and then print them out.
			
			


		# Inhomogenous terms ------
		# ------
		if (Params.Inhomogeneous): 
			self.ITerms = self.PTerms.clone()
			for Term in self.ITerms: 
				for T in Term.Tensors: 
					if T.Time == "s": 
						T.Time = "i"		
		OutFile = open("./Terms/phphPterm","w")
		pickle.Pickler(OutFile,0).dump(self.PTerms)
		OutFile.close()
		del OutFile

		if(Params.Inhomogeneous):
			OutFile = open("./Terms/phphIterm","w")
			pickle.Pickler(OutFile,0).dump(self.ITerms)
			OutFile.close()
			del OutFile
		if(Params.Undressed):
			OutFile = open("./Terms/UndressedTerms","w")
			pickle.Pickler(OutFile,0).dump(self.UndressedTerms)
			OutFile.close()
			del OutFile

		# This causes pickling issues... Doing pickling after. 
		self.PTerms.AssignBCFandTE()
		# Similarly handle the inhomogenous terms. 
		if(Params.Inhomogeneous):
			self.ITerms.AssignBCFandTE()
		if(Params.Undressed): 
			self.UndressedTerms.AssignBCFandTE() # This caused pickling issues so I'm doing it after.					

		if (self.QSpaceProjector): 
			self.PTerms.QPart()
		if (self.ForceAntiHermitian): 
			print "Adding time-reversed terms to force antihermiticity."
			self.PTerms.AddAntiHermitianCounterpart()
			if(Params.Inhomogeneous):
				self.ITerms.AddAntiHermitianCounterpart()

		print "Complete Homogeneous Term: "
		for Term in self.PTerms: 
			Term.Print()

		print "Complete Inhomogeneous Term: "
		for Term in self.ITerms: 
			Term.Print()

		# These are the critical things assumed by the numerical part. 
		self.VectorShape = FermiRho.clone()
		self.ResidualShape = self.CISTerms.clone()

		import gc
		gc.collect()
		return 

	def DipoleMoment(self,AVector, Time=0.0): 
		no = Params.nocc
		nv = Params.nvirt
		DressedMu = Integrals["mu_ph"].copy()
		Mu = 0.0
		if (Params.Adiabatic == 0 and self.PTerms[0].BosInf != None and not Params.Undressed) : 
			DressedMu = Integrals["mu_ph"].copy()
			BI = self.PTerms[0].BosInf
			MTilde = BI.MTilde
			Freqs = BI.Freqs
			Coths = BI.Coths
			Css = BI.Css
			smax = BI.NBos
			for i in range(DressedMu.shape[0]):
				for j in range(DressedMu.shape[1]):
					TildeSum1 = 0.0
					TildeSum2 = 0.0
					wSum = 0.0 
					BathSum = 0.0
					for s in range(smax): 
						TildeSum1 = (MTilde[s][i + no] - MTilde[s][j]) 
						TildeSum2 = (MTilde[s][i + no] - MTilde[s][j]) 
						BathSum += (TildeSum1)*(TildeSum2)*(self.PTerms[0].BosInf.CAt(Time)[s])
						wSum -= Coths[s]*(TildeSum1+TildeSum2)*(TildeSum1+TildeSum2)/2.0
					try:
						DressedMu[i][j] *= numpy.exp(wSum+BathSum).real
					# Why was there a .real up there. 4-20-2012	
					# Oh... Because the dipole moment should be real. Of course. 
					except Exception as Ex: 
						print "flow error wsum ",wSum, " Bathsum ", BathSum ," : ", (wSum+BathSum)
						raise						
		Mu = numpy.tensordot(AVector["r1_ph"],DressedMu,axes=([0,1],[0,1]))
		if (Params.Correlated and self.SecondOrderDipole and not Params.Undressed and not Params.Adiabatic == 2): 
			for i in range(no):
				for j in range(no):
					for a in range(nv):
						for b in range(nv):	
							Mu += (DressedMu[a][i]*AVector["r1_ph"][b][j]*Integrals["v_pphh"][a][b][i][j])/(Integrals["e_p"][a]+Integrals["e_p"][b]-Integrals["e_h"][i]-Integrals["e_h"][j])
			#Taking the abs totally messes this up. 
		import gc
		gc.collect()
		if (Params.DirectionSpecific): 
			return Mu
		else : 
			return (1.0/3.0)*numpy.sum(Mu*Mu)

	# Serve up an entropy and occupation number spectrum. 
	def BathCorrelationMeasures(self): 
		integral = complex(0.0,0.0)
		integrand = complex(0.0,0.0)
		if (self.PTerms[0].OldTime < 10.0): 
			return 0.0,0.0,0.0
		for Term in self.PTerms: 
			if (Term.WatchBCIndex == None): 
				Term.WatchBCIndex = numpy.unravel_index(Term.IatT0.argmax(), Term.CurrentIntegral.shape)
				integral += Term.CurrentIntegral[Term.WatchBCIndex]
				integrand += Term.IatT0[Term.WatchBCIndex]
			else: 
				integral += Term.CurrentIntegral[Term.WatchBCIndex]
				integrand += Term.IatT0[Term.WatchBCIndex]
		#corrfcn = numpy.sum(self.PTerms[0].BosInf.Css)
		BI = self.PTerms[0].BosInf
		MTilde = BI.MTilde
		Freqs = BI.Freqs
		Coths = BI.Coths
		Css = BI.Css
		smax = BI.NBos
		no = Params.nocc
		i = 0 
		j = 1
		TildeSum1 = 0.0
		TildeSum2 = 0.0
		wSum = 0.0 
		BathSum = 0.0
		for s in range(smax): 
			TildeSum1 = (MTilde[s][i + no] - MTilde[s][j]) 
			TildeSum2 = (MTilde[s][i + no] - MTilde[s][j]) 
			BathSum += (TildeSum1)*(TildeSum2)*Term.BosInf.CAt(Term.OldTime)[0]
			wSum -= Coths[s]*(TildeSum1+TildeSum2)*(TildeSum1+TildeSum2)/2.0		
		corrfcn = numpy.exp(wSum)*numpy.exp(BathSum)
		return integral, integrand, corrfcn

	def CISWeights(self,AVector): 
		if Params.BeginWithStatesOfEnergy == None: 
			return numpy.array([numpy.abs(K.InnerProduct(AVector)) for K in self.CISStates])
		else :
			tore = []
			for i in rlen(self.CISEnergies): 
				if (abs(self.CISEnergies[i] - Params.BeginWithStatesOfEnergy) < Params.PulseWidth): 
					tore.append(numpy.abs((self.CISStates[i]).InnerProduct(AVector)))
			return numpy.array(tore)

	# Serve up an entropy and occupation number spectrum. 
	def EntropyAndONSpectrum(self,AVector): 
		# Check that the input vector is normalized. 
		Nrm = sqrt(AVector.InnerProduct(AVector))
		if Nrm == 0.0: 
			return 0,numpy.zeros(Params.nmo)
		r = AVector["r1_ph"]/Nrm
		no = Params.nocc
		nv = Params.nvirt
		nmo = Params.nocc + Params.nvirt
		Entropy = numpy.vectorize(lambda X: X*numpy.log(X) if X !=0 else 0.0) 		
#
#  I did it MY WAY... 
#		
		if (False): # Transition orbitals. 
			u,S,v = numpy.linalg.svd(r, full_matrices=1, compute_uv=1)
		elif (False):  # Simple Density. 
			ToDiag = numpy.zeros(shape=(nmo,nmo),dtype = complex)
			for i in range(no): 
				for a in range(nv): 
					ToDiag[i][a+no] = r[a][i]
					ToDiag[a+no][i] = r[a][i].conj()
			u,S,v = numpy.linalg.svd(ToDiag, full_matrices=1, compute_uv=1)
		else : # If it were CIS-Amplitudes type density equation. 
			# JPC 1995 99 14261. 
			ToDiag = numpy.zeros(shape=(Params.nmo,Params.nmo),dtype = complex)
			for i in range(no): 
				for j in range(no): 
					for a in range(nv): 
						ToDiag[i][j] -= r[a][i].conj()*r[a][j]
			for a in range(nv): 
				for b in range(nv): 
					for i in range(no): 
						ToDiag[a+no][b+no] += r[a][i].conj()*r[b][i]			
			S,v = numpy.linalg.eig(ToDiag.real)
#			return  numpy.sum(w.real[len(w)/2:]), sorted(w)
		return numpy.sum(Entropy(S*S)), sorted(S) #sorted(S*S)

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

# Explictly Diagonalize the Liouvillian. 
	def ExactCISStates(self): 
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
			self.VectorShape.EvaluateContributions(self.CISTerms,OutVector,InVector)
			for J in range(dim): 
				ToDiag[I][J] = (OutVector[((IndexVector[J])[0])])[(IndexVector[J])[1]]
#		print "Indices: ", IndexVector
#		print "Matrix: ", ToDiag
		ToDiag = ToDiag + ToDiag.transpose()
		ToDiag *= 1.0/2.0 
		w,v = numpy.linalg.eig(ToDiag) 
		print "Energies Au: ", w.real
		print "Energies eV: ", sorted(w.real*EvPerAu)
		print "Significant Energies eV: ", sorted(list(set(tuple([round(E,4) for E in sorted(w.real*EvPerAu) if E > 0.5]))))
		states = [StateVector() for X in range(dim)]
		for I in range(dim): 
			self.VectorShape.MakeTensorsWithin(states[I])
			states[I].Fill(0.0)
			for J in range(dim):
				((states[I])[((IndexVector[J])[0])])[(IndexVector[J])[1]] = v[J,I]
		self.CISStates = states
		self.CISEnergies = w.real 
		return (w.real, states)
		
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
			if(Params.ReOrg):
				print "Correcting Reorganization Energy"
				Bs.CorrectReorganization()
			if (os.path.isfile("./Terms/Bosons")):
				print "Found Existing Bosons, Unpickling them."
				bf = open("./Terms/Bosons","rb")
				UnpickleBos = pickle.Unpickler(bf)
				Bs = UnpickleBos.load()
				bf.close()
				del bf
				return  
			print "---------------------------------------------"		
			print "Using Boson Correlation Parameters: "
			Bs.Print()
	#		Here we should modify the integrals, but we aren't doing that yet. 
	#		Bs.PlotCorrelationFunction()
			print "Initializing Boson Correlation Functions/Tensors... "
			if (not Params.Undressed and Params.Correlated):
				self.PTerms.MakeBCFs(Bs)
			if (Params.Undressed):
				Temp = BosonInformation(Params)
				Temp.DefaultBosons()
				self.UndressedTerms.MakeBCFs(Temp)
			else: 
				if Params.Correlated: 
					overallsize = 0.0
					temsz = self.PTerms[0].CurrentIntegral.itemsize
					for Term in self.PTerms: 
						Tdim = 1
						sh = Term.CurrentIntegral.shape 
						for d in sh : 
							Tdim *= d
						overallsize += 4.0*Tdim*temsz
					print "Total Bath Correlation Tensor Storage: ", overallsize/pow(1024,2), " MB"

				# BATH RENORMALIZATION	
				if (self.RenormalizeIntegrals and Params.Correlated): 
					print "about to... RENORMALIZE INTEGRALS!!!!... "
					print "PRE-Checking Spin Symmetry... "
					if (not Integrals.CheckSpinSymmetry()): 
						raise Exception("BrokeSpin...")
					print "Mean field energy before... "
					Integrals.MakeHFEnergy()
								
					for Tensor in Integrals.iterkeys(): 
						if (not Tensor in Integrals.Types): 
							continue
						if ((len(Integrals[Tensor].shape))==2): 
							typs = Integrals.Types[Tensor]
							if (typs[0] == typs[1]): 
								Ten = Integrals[Tensor]
								Shifts = [0 if X==0 else Params.nocc for X in typs]
								it = numpy.ndindex(Ten.shape)
								for iter in it: 
									if (iter[0]==iter[1]):
										for b in rlen(Bs.Freqs): 
											Ten[iter] -= Bs.Freqs[b]*pow(Bs.MTilde[b][iter[0]+Shifts[0]],2.0)
						elif ((len(Integrals[Tensor].shape))==4):
							typs = Integrals.Types[Tensor]
							if (typs[0] == typs[3] and typs[1] == typs[2]): 
								Ten = Integrals[Tensor]
								Shifts = [0 if X==0 else Params.nocc for X in typs]
								it = numpy.ndindex(Ten.shape)
								for iter in it: 
									if (iter[0]==iter[3] and iter[1]==iter[2] and iter[0] != iter[1]):
										for b in rlen(Bs.Freqs): 
											sfiter = Params.SpinFlipOfIndex(iter,typs)
											Ten[iter] -= (Bs.Freqs[b]*Bs.MTilde[b][iter[0]+Shifts[0]])*(Bs.MTilde[b][iter[1]+Shifts[1]])
											Ten[sfiter] -= (Bs.Freqs[b]*Bs.MTilde[b][iter[0]+Shifts[0]])*(Bs.MTilde[b][iter[1]+Shifts[1]])
					print "Mean field energy after... "
					Integrals.MakeHFEnergy()
					print "Checking Spin Symmetry... "
					if (not Integrals.CheckSpinSymmetry()): 
						raise Exception("BrokeSpin...")
			
		# ------------------------ 
		# prepare some single excitation as the initial guess. 
		# homo->lumo (remembering alpha and beta.)
		# ------------------------ 
		if (not Params.Undressed): 
			self.ExactCISStates()
		
		# In this case the spectrum can be made exactly. 		
		if (Params.Adiabatic == 4):
			pmatrix = self.MakeMarkovLiouvillian()
			dim = len(self.CISEnergies)
			NonPertMatrix = numpy.ndarray(shape = (dim,dim), dtype = complex) 
			NonPertMatrix *= 0.0
			for I in range(dim): 
				NonPertMatrix[I][I] = self.CISEnergies[I].real
			NonPertMatrix *= complex(0.0,-1.0)
			for S1 in rlen(self.CISStates): 
				Rslt = numpy.tensordot(pmatrix,(self.CISStates[S1])["r1_ph"],axes=([2,3],[0,1]))
				for S2 in rlen(self.CISStates): 
					EN = numpy.sum((self.CISStates[S2])["r1_ph"].conj()*(Rslt))
					NonPertMatrix[S2][S1] += EN
			w2,v2 = numpy.linalg.eig(NonPertMatrix) 
			print "Energies of Markovian (eV): ", (w2*EvPerAu)/complex(0.0,-1.0)			
			
			self.MarkovEnergies =  (w2)/complex(0.0,-1.0)
			states = [StateVector() for X in range(dim)]
			for I in range(dim): 
				self.VectorShape.MakeTensorsWithin(states[I])
				states[I].Fill(0.0)
				for J in range(dim):
					states[I].Add(self.CISStates[J],v2[J,I])
			self.MarkovStates = states
		# End Perturbative Liouvillian Section. 									
		self.V0=StateVector()
		self.VectorShape.MakeTensorsWithin(self.V0)
		self.V0.Fill(0.0)
		r1_ph = self.V0["r1_ph"]
		# this is the density emerging from a dipole excitation. 
		if (Params.BeginWithStatesOfEnergy != None): 
			print "Initalizing with a superposition of a few states.... "
			Sts = Ens = None
			if (Params.Adiabatic == 4): 
				Sts, Ens = self.MarkovStates, self.MarkovEnergies
			else: 
				Sts, Ens = self.CISStates, self.CISEnergies
			ToGuess = []
			for i in rlen(Ens): 
				if (abs(Ens[i] - Params.BeginWithStatesOfEnergy) < Params.PulseWidth): 
					ToGuess.append(i)
			# Add weight for only the highest energy 
			GsEns = numpy.array(map(lambda X: Ens[X], ToGuess))
			Me = GsEns.max()
			for i in ToGuess: 
				if (Ens[i] == Me): 
					self.V0.Add(Sts[i],1.0)
				else : 	
					self.V0.Add(Sts[i],1.0)
				print "Adding Contribution at energy:", Ens[i]*27.2113 
				vec = (Sts[i])["r1_ph"]
				for o in range(Params.nocc): 
					for v in range(Params.nvirt):
						if (abs(vec[v][o]) > pow(10.0,-3.0)):
							print "o:", o, " v: ", v, " : ", vec[v][o]
			self.V0.MultiplyScalar(1.0/numpy.sqrt(self.V0.InnerProduct(self.V0)))						
		elif (Params.DipoleGuess):
			if (self.Polarization != None): 
				if self.Polarization == "x":
					r1_ph += Integrals["mu_ph"][:,:,0]
				if self.Polarization == "y":
					r1_ph += Integrals["mu_ph"][:,:,1]
				if self.Polarization == "z":
					r1_ph += Integrals["mu_ph"][:,:,2]										
			else: 
				print "Exciting with dipole guess"
				r1_ph += Integrals["mu_ph"][:,:,0]
				if (Params.InitialDirection == -1):
					r1_ph += Integrals["mu_ph"][:,:,1]
					r1_ph += Integrals["mu_ph"][:,:,2]
				else : 
					print "Anisotropic dipole guess !!!!!!!!!"
			if (self.V0.InnerProduct(self.V0)) != 0.0:
				self.V0.MultiplyScalar(1.0/numpy.sqrt(self.V0.InnerProduct(self.V0)))		
		else :
			# r1_hp = self.V0["r1_hp"]
			# Try just propagating ph->ph 
			r1_hp = numpy.zeros(shape=(Params.nocc,Params.nvirt) , dtype=complex)
			if (False): 
				print "Checking Spin Symmetry ... "
				self.V0.CheckSpinSymmetry() # I don't think this was working (at all) Jan 20-2012.)
				print "Projecting onto exact eigenstates... "
				Ens,Sts = self.CISEnergies, self.CISStates
				# UGH! The degenerate states are not orthogonal. 
				#SymmetricOrthogonalize(Sts)
				print [ [(Sts[k]).InnerProduct(Sts[j]).real.round(5) for j in range(len(Sts)) ] for k in range(len(Sts))]
				SumSq = 0.0
				for s in range(len(Ens)): 
					Wght = Sts[s].InnerProduct(self.V0)
					print "En: ", Ens[s]," (ev) ", Ens[s]*EvPerAu ,  " Weight ", Wght, " dipole: ", self.DipoleMoment(Sts[s])
					SumSq += Wght*numpy.conjugate(Wght)
				print "SumSq: ", SumSq
			Ens,Sts = self.CISEnergies, self.CISStates
			# simply make an even mix bright states. 
			self.V0.Fill(0.0)
			for s in range(len(Ens)): 
				if (numpy.sum(self.DipoleMoment(Sts[s])*self.DipoleMoment(Sts[s])) > 0.001):
	#			if (abs(Ens[s]-11.54/EvPerAu) < .1): 
					self.V0.Add(Sts[s])
			self.V0.MultiplyScalar(1.0/numpy.sqrt(self.V0.InnerProduct(self.V0)))		
			SumSq = 0.0
			for s in range(len(Ens)): 
				Wght = Sts[s].InnerProduct(self.V0)
				print "En: ", Ens[s]," (ev) ", Ens[s]*EvPerAu ,  " Weight ", Wght, " dipole: ", self.DipoleMoment(Sts[s])
#				print "Contributions: "
#				Sts[s].Print()
				SumSq += Wght*numpy.conjugate(Wght)
			# The exact states are not orthogonal. 
			print "SumSq: ", SumSq
			
		self.Mu0 = self.DipoleMoment(self.V0)
		print self.Mu0
		import gc
		gc.collect()
		return 

	def First(self): 
		return self.V0
		
	def MakeMarkovLiouvillian(self): 
		self.NonPertMtrx = numpy.zeros(shape=(Params.nvirt,Params.nocc,Params.nvirt,Params.nocc),dtype = complex)		
		Inp = StateVector()
		self.VectorShape.MakeTensorsWithin(Inp)
		Rslt = StateVector()
		self.VectorShape.MakeTensorsWithin(Rslt)
		for a in range(Params.nvirt): 
			for i in range(Params.nocc): 		
				Inp.Fill()
				Rslt.Fill()
				Inp["r1_ph"][a][i] += 1.0
				self.VectorShape.EvaluateContributions(self.CISTerms,Rslt,Inp)
				Rslt.MultiplyScalar(complex(0,-1.0))				
				for b in range(Params.nvirt): 
					for j in range(Params.nocc):
						self.NonPertMtrx[b][j][a][i] = Rslt["r1_ph"][b][j]

		OverallMtrx = numpy.ndarray(shape=(Params.nvirt,Params.nocc,Params.nvirt,Params.nocc),dtype = complex)		
		OverallMtrx *= 0.0
		numpy.seterr(under='ignore')
		for Term in self.PTerms: 
			print "Generating Markovian Rate Matrix for Term: "
			Term.Print()
			self.V0 = StateVector()
			self.VectorShape.MakeTensorsWithin(self.V0)
			Mtrx = Term.EvaluateMatrixForm(self.V0)
			HPart = (Mtrx + numpy.transpose(Mtrx,(2,3,0,1)).conj())*0.5
			AHPart = (Mtrx - numpy.transpose(Mtrx,(2,3,0,1)).conj())*0.5
			print "This Term's contribution to the rates: "
			for S1 in rlen(self.CISStates): 
				Rslt = numpy.tensordot(HPart,(self.CISStates[S1])["r1_ph"],axes=([2,3],[0,1]))
				for S2 in rlen(self.CISStates): 
					EN = numpy.sum((self.CISStates[S2])["r1_ph"].conj()*(Rslt))
					if (abs(EN)>pow(10.0,-5.0)):
						print S1," ", self.CISEnergies[S1]*27.2113, "->" , S2 ," ", self.CISEnergies[S2]*27.2113, " : " , EN , " and in time: ", 1.0/EN
			print " ------------------------------------ "			
			OverallMtrx += Mtrx		
			dex = numpy.ndindex(OverallMtrx.shape) 
			for I in dex: 
				if abs(OverallMtrx[I]) < pow(10.0,-10.): 
					OverallMtrx[I] = 0.0
		print "OVERALL in Basis of CIS States: ----- "
		HPart = (OverallMtrx + numpy.transpose(OverallMtrx,(2,3,0,1)).conj())*0.5
		AHPart = (OverallMtrx - numpy.transpose(OverallMtrx,(2,3,0,1)).conj())*0.5
		print "HPart: ", numpy.sum(HPart*HPart.conj()), " ahpart ", numpy.sum(AHPart*AHPart.conj())		
		for S1 in rlen(self.CISStates): 
			Rslt = numpy.tensordot(OverallMtrx,(self.CISStates[S1])["r1_ph"],axes=([2,3],[0,1]))
			D = numpy.ndindex(Rslt.shape)
			for i in D: 
				if (abs(Rslt[i]) < pow(10.0,-10.0)): 
					Rslt[i] = 0.0
			for S2 in rlen(self.CISStates): 
				if (Params.SecularApproximation == 1 and S1 != S2): 
					continue 			
				EN = numpy.sum((self.CISStates[S2])["r1_ph"].conj()*(Rslt))
				if (abs(EN)>pow(10.0,-5.0)):
					print S1," ", self.CISEnergies[S1]*27.2113, "->" , S2 ," ", self.CISEnergies[S2]*27.2113, " : " , EN , " and in time: ", 1.0/EN

		self.MarkovMatrix = OverallMtrx
		Params.ExponentialStep = True
		import gc
		gc.collect()
		return OverallMtrx
		
	# now operate -i[H, ] on rho:
	def Step(self,OldState,Time,Field = None, AdiabaticMultiplier = 1.0): 
		NewState = OldState.clone()
		NewState.Fill()
				
		self.VectorShape.EvaluateContributions(self.CISTerms,NewState,OldState)
		if (Field != None):
			if (sqrt(Field[0]*Field[0]+Field[1]*Field[1]+Field[2]*Field[2]) > Params.FieldThreshold):
				NewState["r1_ph"] += Integrals["mux_ph"]*Field[0]
				NewState["r1_ph"] += Integrals["muy_ph"]*Field[1]
				NewState["r1_ph"] += Integrals["muz_ph"]*Field[2]			
				self.VectorShape.EvaluateContributions(self.MuTerms,NewState,OldState, FieldInformation = Field)		
		NewState.MultiplyScalar(complex(0,-1.0))
		if (Params.Undressed): 
			self.VectorShape.EvaluateContributions(self.UndressedTerms,NewState,self.V0,TimeInformation=Time, MultiplyBy=AdiabaticMultiplier)
		if (Params.Correlated):
			self.VectorShape.EvaluateContributions(self.PTerms,NewState,OldState,TimeInformation=Time, MultiplyBy=AdiabaticMultiplier)
		if (Params.Inhomogenous): 
			self.VectorShape.EvaluateContributions(self.ITerms,NewState,self.V0,TimeInformation=Time, MultiplyBy=AdiabaticMultiplier)
		import gc
		gc.collect()
		return NewState
