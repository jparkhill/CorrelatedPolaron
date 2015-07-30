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
#
# A TDA-like approximation to the response matrix (studying only ph->ph transformations) 
# Also splitting up the terms and building on LCISD 
#

class CISMatrix: 
	def __init__(self):
		print "Getting ready for LIOUVILLE SPACE CIS Propagation... "
		# these things must be defined by the algebra. 
		self.VectorShape = None 
		self.ResidualShape = None

		self.CISTerms = None  # Guess is provided by CIS. 
		self.InitAlgebra()
		print "Algebra Initalization Complete... "
		self.V0 = StateVector()
		self.Mu0 = None
		self.InitalizeNumerics() 
		print "CIS Initalization Complete... "
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
		if (os.path.isfile("./Terms/CIS") and os.path.isfile("./Terms/phphPterm")):
			print "Found Existing Terms, Unpickling them."
			cf = open("./Terms/CIS","rb")
			pf = open("./Terms/phphPterm","rb")			
			UnpickleCIS = pickle.Unpickler(cf)
			UnpicklePT = pickle.Unpickler(pf)			
			self.CISTerms = UnpickleCIS.load()
			cf.close()
			self.PTerms = UnpicklePT.load()
			pf.close()
			self.PTerms.AssignBCFandTE() # This caused pickling issues so I'm doing it after. 			
						
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
		for Term in LeftVac: 
			Term.Print()
		print "*_*_*_*_*_*_*_*__*_*_*_*_*_*_*_*_*"
		self.CISTerms = LeftVac.clone()

		OutFile = open("./Terms/CIS","w")
		pickle.Pickler(OutFile,0).dump(self.CISTerms)
		OutFile.close()

		# These are the critical things assumed by the numerical part. 
		self.VectorShape = FermiRho.clone()
		self.ResidualShape = self.CISTerms.clone()
		return 

	def DipoleMoment(self,AVector, Time=0.0): 
		Mu = numpy.tensordot(AVector["r1_ph"],Integrals["mu_ph"],axes=([0,1],[0,1]))
		return (1.0/3.0)*numpy.sum(Mu*Mu)

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
			self.VectorShape.EvaluateContributions(self.CISTerms,OutVector,InVector)
			for J in range(dim): 
				ToDiag[I][J] = (OutVector[((IndexVector[J])[0])])[(IndexVector[J])[1]]
		numpy.savetxt("./Output/CISMatrix",ToDiag,fmt='%.18e')		
		# Create a summarized version of the matrix. 
		IndexTypes = list(set(tuple(map(lambda X: X[0],IndexVector))))
		print "index types: ", IndexTypes
		TypedMatrix = numpy.zeros((len(IndexTypes),len(IndexTypes)),dtype = complex)
		# make a typed summary of the matrix. 
		for I in range(len(IndexVector)): 
			for J in range(len(IndexVector)): 		
				SI = IndexTypes.index(IndexVector[I][0])
				SJ = IndexTypes.index(IndexVector[J][0])
				TypedMatrix[SI][SJ] += ToDiag[I][J]
		print "Typed Matrix: ", TypedMatrix
		
		print "Indices: ", IndexVector
		print "Matrix: ", ToDiag
		ToDiag = ToDiag + ToDiag.transpose()
		ToDiag *= 1.0/2.0 
		w,v = numpy.linalg.eig(ToDiag) 
		print "Energies Au: ", w.real
		print "Energies eV: ", sorted(w.real*EvPerAu)
		print "Significant Energies eV: ", sorted(list(set(tuple([round(E,4) for E in sorted(w.real*EvPerAu) if E > 0.5]))))		
		return 

	def InitalizeNumerics(self): 
	# prepare some single excitation as the initial guess. 
	# homo->lumo (remembering alpha and beta.)
		self.VectorShape.MakeTensorsWithin(self.V0)
		r1_ph = self.V0["r1_ph"]
		# r1_hp = self.V0["r1_hp"]
		# Try just propagating ph->ph 
		r1_hp = numpy.zeros(shape=(Params.nocc,Params.nvirt) , dtype=complex)

		self.ExplicitlyDiagonalize()

		print "Initializing Boson Correlation Functions/Tensors... "
		self.Mu0 = self.DipoleMoment(self.V0)
		return 
