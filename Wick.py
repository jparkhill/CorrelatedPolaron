#
# Fermion Wick's theorem in Python. 
# For Genuine and Fermi Vacuums.
#
# Also Provides "EvaluateContributions()" 
# Which does the numerical work of evaluating a term. 
# Given An Integrals  (See TensorNumerics.py)
# And State Vectors
#
# John Parkhill Nov 21st. 2011
#

import numpy, scipy
from numpy import array
from numpy import linalg
import numpy as np
from scipy.special import gamma, zeta
from scipy.integrate import quad, quadrature, fixed_quad
# standard crap
import scipy.weave as weave # To the cloud!
import os, sys, time, math, re, random
from time import gmtime, strftime
from types import * 
from itertools import izip
from heapq import nlargest
from multiprocessing import Process, Queue, Pipe
from math import pow, exp, cos, sin, sinh, cosh, log, pi, sqrt, isnan
from TensorNumerics import * 
from LooseAdditions import * 
import copy, cmath

# Just keeps track of which indices map to which in the data. 
# and holds a pointer to that data. 
class TensorIndexer: 
	def __init__(self,nam,ops,Vac = False,ToFerm = False): 
		self.name = nam
		# "Which time is this operator at."
		self.Time = ""
		self.vac = ""
		# A useful catch-all for things like hole-particle character.  
		self.suffix = ""
		self.ops = copy.deepcopy(ops) 
		self.indices = map(lambda X:X[1],ops)
		self.indextypes = map(lambda X:X[0],ops)
		# stupid stupid <pq||rs> +p+qsr nonsense. 
		if (not Vac): 
			if (len(self.indices) >= 4 and len(self.indices)%2 == 0):
				crea = [1,4,5]
				anni = [2,3,6]
				fh = self.indextypes[:len(self.indextypes)/2]
				sh = self.indextypes[len(self.indextypes)/2:]
				for X in fh: 
					if (not X in crea): 
						print "You're not creating a normal operator. RuhRoh.1", ops
						raise Exception	
				for Y in sh: 
					if (not Y in anni): 
						print "You're not creating a normal operator. RuhRoh.2",ops
						raise Exception								
				l1 = copy.copy(self.indices[:len(self.indices)/2])
				l2 = copy.copy(self.indices[len(self.indices)/2:])
				l2.reverse()
				l1.extend(l2)
				self.indices = copy.copy(l1)
				l1 = copy.copy(self.indextypes[:len(self.indextypes)/2])
				l2 = copy.copy(self.indextypes[len(self.indextypes)/2:])
				l2.reverse()
				l1.extend(l2)
				self.indextypes = copy.copy(l1)	
				# if it's a rank 2 tensor then these are already properly assigned. 
		# if ToFerm = True then suffix the name with my types: 
		if (ToFerm):
			HPChar = lambda XX:  "h" if (XX == 2 or XX == 4) else "p"
			sufx = "_"+"".join(map(HPChar,self.indextypes))	
			self.suffix = self.suffix + sufx
		return 
	def TypeOfDummy(self,D): 
		for O in self.ops: 
			if O[1] == D:
				return O[0]
		raise Exception("Dummy Not Found... ")
		return 
	def ReDummy(self,Mep): 
		for I in range(len(self.indices)): 
			self.indices[I] = Mep[self.indices[I]]
		for I in range(len(self.ops)):
			self.ops[I][1] = Mep[(self.ops[I][1])]
		return 
	def FlipVac(self): 
		if (self.vac == "_Vac"): 
			self.vac = ""
		else: 
			self.vac = "_Vac"
		return 
	def NameConjugate(self): 
		stmp = list(self.suffix)
		stmp.pop(0)
		stmp = stmp[::-1]
		self.suffix = '_'+''.join(stmp)
		return 
	def OrderHP(self): 
		if (len(self.ops) != 4): 
			return 1			
		tore = 1
		if (self.indextypes[0] == 1 and self.indextypes[1] == 4): 
			self.Swap(0,1)
			tore *= -1
		if (self.indextypes[2] == 3 and self.indextypes[3] == 2): 	
			self.Swap(2,3)
			tore *= -1
		return tore
	def Swap(self,n,m):
		SFX = list(self.suffix)
		SFX[n+1], SFX[m+1] = SFX[m+1], SFX[n+1]  
		self.suffix = ''.join(SFX)
		self.ops[n], self.ops[m] = self.ops[m], self.ops[n]
		self.indices[n], self.indices[m] = self.indices[m], self.indices[n]
		self.indextypes[n], self.indextypes[m] = self.indextypes[m], self.indextypes[n]
		return	
	def CreateWithin(self,VectorOfTensors): 
		if (not self.Name() in VectorOfTensors): 
	#		print "Creating:", self.Name(), self.indextypes
			if ((1 in self.indextypes) or (2 in self.indextypes)or(3 in self.indextypes)or (4 in self.indextypes)): 
				XX = lambda XX: 0 if (XX==4 or XX==2) else 1 
			#	print "Creating with Index Types: ", map(XX,self.indextypes)
				VectorOfTensors.Create(self.Name(),len(self.indices),map(XX,self.indextypes))
			else:
				VectorOfTensors.Create(self.Name(),len(self.indices))
		return
	def Order(self): 
		if (self.Name().count("Vac") > 0): 
			return 0 
		else :
			return 1
	def IsVac(self): 
		return (self.vac != "")
	def IsLike(self,Other): 
		return (self.indextypes == Other.indextypes)
	def GiveTime(self,sufx): 
		self.Time+=sufx
		return
	def Name(self): 
		return (self.name+self.vac+self.suffix)
	def NonVacName(self): 
		return self.name+self.suffix
	def RootName(self): 
		return self.name
	def MyDataFrom(self,VectorOfTensors):
		if (self.IsVac()): 
			print "Error, Looking for Vaccuum"
			raise Exception("VacData From") 
		TimeIndependentVersion = None
		if (self.Name() in Integrals): 
			TimeIndependentVersion = Integrals[self.Name()]
		elif (self.Name() in VectorOfTensors):
			TimeIndependentVersion = VectorOfTensors[self.Name()]
		else : 
			print "Error, Tensor not found, self: "
			self.Print()
			print "Contents of VectorOfTensors: "
			for k in VectorOfTensors.iterkeys(): 
				print "k: ", k
			raise Exception("Tensor Not Found")
		return TimeIndependentVersion
		if (False) : 
			TimeToRotate = [X[1] for X in TimeInformation if X[0] == self.Time][0]		
			TimeDependentVersion = TimeIndependentVersion.copy()
			# Signs depend on whether they are creation or annihilation operators. 
			# Also the p or h indices. 
			dim = len(self.indextypes)
			Mapto01 = lambda XX:  0 if (XX == 2 or XX == 4) else 1
			Typs = map(Mapto01,self.indextypes)			
			TimeDependentVersion = Integrals[self.Name()].copy()
			# Fuck this and do broadcasted version. 
			for Axis in range(dim): 
				TimeSign = 1.0
				if (Axis >= (dim/2)): 
					TimeSign = -1.0
				ToRotate = None
				if (Typs[Axis] == 0):
					ToRotate = Integrals["e_h"]
				elif (Typs[Axis] == 1):
					ToRotate = Integrals["e_p"]
				else : 
					print "Can't Fock Rotate, wrong type :( "
					raise Exception("Fuuuuucccc.... ")
				Rotator = numpy.exp(ToRotate*TimeSign*TimeToRotate*complex(0.0,1.0))
				TimeDependentVersion = numpy.apply_along_axis(lambda X: X*Rotator , Axis , TimeDependentVersion )
			return TimeDependentVersion
		return 
	def	OpIndices(self): 
		return map(lambda X:X[1], self.ops )
	def Contains(self,i): 
		return (i in self.indices)
	def Print(self): 
		print self.Name()+"("+self.Time+")["+str(self.indices)+"]"+"Ops:"+str(self.ops)	
	def Shift(self,amount): 
		dim = range(len(self.indices))
		for I in dim: 
			self.indices[I] += amount
		for O in self.ops:
			O[1] += amount
		return 

# Operator String - ----------
# Scalar multiplier (float)
# delta functions occur between operators with dummy indices: [[i,j],[k,l],[c,d], ... ]
# if they've already been contracted over their types are made negative to speed contraction
# genuine vacuum notation: ---------
# Creation (hole and particle) = 5 
# Annihilation (hole and particle) = 6
# fermi vacuum ---------
# hole creation (a^\dagger_i) = 4 
# particle annihilation (a_a) = 3
# hole ann. (a_i) = 2
# particle creation (a^\dagger_a) = 1
class OperatorString: 
	def __init__(self,fac=1.0,dels=[],ops=[],nam="N"): 
		self.PreFactor = fac
		self.Deltas = copy.copy(dels)
		self.Ops=copy.deepcopy(ops)
		self.Tensors = [TensorIndexer(nam,ops)] # The prefactor tensor
		
		# things specific to Polarization propagator. 
		self.EDeltat = []
		self.EDeltas = []
		self.EDeltai = []
		self.EDTTensor = None
		self.EDSTensor = None
		self.EDITensor = None
		self.TimeReversed = False

		# things specific to the system-bath model. 
		self.BosInf = None
		self.BCIndices = None
		self.BCIndexShifts = None
		self.UniqueBCIndices = None
		
		# This is a stupid thing. A particular index to watch the BCF and integral. 
		self.WatchBCIndex=None
		
		# These are used to numerically integrate the boson correlation function with Gaussian Quadrature. 
		self.IatT0 = None 
		self.IatT1 = None 
		self.IatT2 = None 		
		self.OldTime = 0.0
		self.CurrentIntegral = None	
		return 

	def DeltasAsDict(self): 
		tore = dict()
		for d in self.Deltas: 
			tore[d[0]] = d[1]
			tore[d[1]] = d[0]
		return tore
	def clone(self): 
		return copy.deepcopy(self)
	def AllReferencedTensors(self):
		tore = []
		for T in self.Tensors: 
			tore.append(T.Name())
		return tore
	def MultiplyScalar(self,AScalar):
		self.PreFactor *= AScalar
		return
	def MakeTensorsWithin(self,AVectorOfTensors): 
		for T in self.Tensors: 
			T.CreateWithin(AVectorOfTensors)	
		return 
	def GiveTime(self, sufx): 
		if (not self.IsSimple()): 
			print "GiveTime, should be simple."
		self.Tensors[0].GiveTime(sufx)
		return
	def MyConjugate(self): 
		if len(self.Tensors) > 1: 
			print "Conjugating multiple Tensors.... RUH ROH."
			raise Exception("MultipleConju")
		self.PreFactor = 1.0
		self.Tensors[0].FlipVac()
		# Recursively apply (ab)+ = b+a+ return conjugate operator. 
		NewOps = []
		while (len(self.Ops) > 0) : 
			tmp1 = copy.deepcopy(self.Ops.pop())
			NewOps.append([self.Conjugate(tmp1[0]),tmp1[1]])
		self.Ops = []
		self.Ops = copy.deepcopy(NewOps)
 		# Stupidly this wasn't conjugating the ops of the tensor as well.. 
		self.Tensors[0].ops = copy.deepcopy(NewOps)
		return 
		
	def AntiHermitianCounterpart(self): 
		#print "AHC, Term: "
		#self.Print()
		tore = self.clone()
		if (tore.EDSTensor == None): 
			print "Error... Must assign BCF to do this Antihermitian Counterpart thing."
			raise Exception("AntiHermCounterpart.")
		RhoIndices = None
		VacIndices = None 
		AllInd = None 
		for T in tore.Tensors: 
			if (T.IsVac()):  
				Vacn=T.NonVacName()
				VacIndices = T.indices
			elif (T.name == "r1"): 
				Rhon = T.Name()
				RhoIndices = T.indices
		if (Vacn != Rhon): 
			print "This hack only works if Vacn == Rhon"
			raise Exception("Wrong Hack used.") 	
		AllInd = self.AllIndices()
		Mep = dict()
		for i in AllInd: 
			Mep[i] = i 
		for i in range(2): 
			Mep[VacIndices[i]], Mep[RhoIndices[i]] = Mep[RhoIndices[i]], Mep[VacIndices[i]]
		# Finally replace the vaccuum and rho indices. 
		for T in tore.Tensors: 
			if (T.IsVac()):  
				T.ReDummy(Mep)
			elif (T.name == "r1"): 
				T.ReDummy(Mep)
		tore.PreFactor *= -1.0
		tore.TimeReversed = True
		# in the case of our model. the integral of the conjugate is the conjugate of the integral
		# so we can do the time-reversal in "EvaluateMyself."
		#print "After: "
		#tore.Print()
		return tore
		
	#conjugates the vacuum and rho
	def ConjugateTerm(self): 
		print "Conjugate for Term: "
		self.Print()
		tore = self.clone()
		ToFlip = []
		for T in rlen(tore.Tensors): 
			if (tore.Tensors[T].name == "r1"): 
				ToFlip.append(T)
		for T in ToFlip: 
			tore.Tensors[T].FlipVac()
			tore.Tensors[T].NameConjugate()
		tore.PreFactor *= -1.0
		tore.TimeReversed = True
		print "After: "
		tore.Print()
		return tore		
		

	def CheckConsistency(self): 	
		for D in self.Deltas: 
			if abs(abs(self.TypeOfDummy(D[0])) - abs(self.TypeOfDummy(D[1]))) != 2: 
				print "Inconsistent Delta. "
				self.Print()
				raise Exception("Fuckk.... ")
		return

	# total hack-shortcut. 
	def	PermuteTimesOnV(self): 
		for T in self.Tensors: 
			if T.Time == "t": 
				if len(T.ops) > 2: 
					T.Time ="s"
			elif T.Time == "s":  
				if len(T.ops) > 2: 
					T.Time ="t"	
		return
		
		# Construct an adjacency matrix and make sure that population in one tensor 
		# ends up in all other tensors. 
	def IsConnected(self): 
		if (len(self.Tensors) < 2): 
			return True
		M = self.AdjacencyMatrix()
		N = numpy.zeros(shape=(len(M),1))
		N[-1][0] = 1.0
		for i in range(len(M)+2): 
			N=numpy.dot(M,N)
		for i in range(len(M)): 			
			if N[i] == 0.0: 
				return False
		return True	
	def IsNotSelfContracted(self):
		if (len(self.Deltas) == 0):
			return True
		for D in self.Deltas: 
			D1 = self.LocationOfDummyIndex(D[0])
			D2 = self.LocationOfDummyIndex(D[1])
			if (D1[0] == D2[0]):
				return False
		return True
	def HPTypeDummy(self,I): 
		for O in self.Ops: 
			if O[1] == I: 
				if (abs(O[0]) == 2 or abs(O[0]) == 4) : 
					return 0 
				else :
					return 1					
	# for establishing equivalence	
	def TypeOfDummy(self,I): 
		for O in self.Ops: 
			if O[1] == I: 
				return O[0]
		else : 
			return 0 
	def TensorWithName(self,nm): 
		for T in self.Tensors: 
			if T.Name() == nm: 
				return T
		return None
		
	# an integral is "ordered" when holes preceed particles
	# in the integral's name. 
	# permute to get there if any integrals in this term are unordered.  
	def OrderHP(self):
		newsign = 1.0
		for T in self.Tensors: 
			newsign *= T.OrderHP()
		self.PreFactor *= newsign
		
	# Gives this a term a "Canonical "set of dummy indices
	# which are always increasing when the operators are written in order for each tensor. 
	def ReDummy(self):
		if ( self.BCIndices != None): 
			print "Shouldn't redummy when I have BCF... "
			raise Exception("Redummy")
		OldIndices = []
		# Jap Feb 26 2012. 
		# Vacuum operator indices need to be conjugated. 
		for T in self.Tensors: 
			OldIndices.extend(T.OpIndices())
		# Only do the UNIQUE indices
		NewOldIndices = []
		for X in OldIndices: 
			if NewOldIndices.count(X) == 0: 
				NewOldIndices.append(X)
		OldIndices = NewOldIndices
		#OldIndices = list(set(OldIndices))   # This Fucks up the indices of the vacuum. 
		Mep = dict()
		for i in range(len(OldIndices)):
			Mep[OldIndices[i]] = i
		Pos = 0 
		NewOps = []
		NewDeltas = []
		for T in self.Tensors: 
			T.ReDummy(Mep)
		for i in rlen(self.Ops):
			NewOps.append([self.Ops[i][0],Mep[self.Ops[i][1]]])
		for i in rlen(self.Deltas):
			NewDeltas.append([Mep[self.Deltas[i][0]],Mep[self.Deltas[i][1]]])				
		self.Ops = NewOps
		self.Deltas = NewDeltas				
		return 

	# Keep only summations where the external and r indices match.  
	def Secularize(self):
		OldIndices = []
		VacIndices = None	
		RhoIndices = None
		for T in self.Tensors: 
			OldIndices.extend(T.OpIndices())
			if T.IsVac():
				VacIndices=T.OpIndices()
			elif (T.name == "r1"): 
				RhoIndices=T.OpIndices()
		NewOldIndices = []
		for X in OldIndices: 
			if NewOldIndices.count(X) == 0: 
				NewOldIndices.append(X)
		OldIndices = NewOldIndices
		Mep = dict()
		for i in range(len(OldIndices)):
			Mep[OldIndices[i]] = i
		# this is where the secular bit comes in. 
		MepNew = dict()
		for O in OldIndices:
			if (O in RhoIndices): 
				if (RhoIndices.index(O) == 0): 
					MepNew[O] = Mep[VacIndices[1]] 
				else:
					MepNew[O] = Mep[VacIndices[0]] 
			else: 
				MepNew[O] = Mep[O]		
		Pos = 0 
		NewOps = []
		NewDeltas = []
		for T in self.Tensors: 
			T.ReDummy(MepNew)
		for i in rlen(self.Ops):
			NewOps.append([self.Ops[i][0],MepNew[self.Ops[i][1]]])
		for i in rlen(self.Deltas):
			NewDeltas.append([MepNew[self.Deltas[i][0]],MepNew[self.Deltas[i][1]]])
		NewEDeltat = []
		NewEDeltas = []
		# Also fix: 
		for E in self.EDeltat:
			NewEDeltat.append([E[0],MepNew[E[1]]])
		for E in self.EDeltas:
			NewEDeltas.append([E[0],MepNew[E[1]]])
		NewBCIndices=[]
		for I in self.BCIndices: 
			NewBCIndices.append(MepNew[I])
		self.BCIndices = NewBCIndices
		self.UniqueBCIndices = list(set(NewBCIndices))
		self.Ops = NewOps
		self.Deltas = NewDeltas				
		self.EDeltat = NewEDeltat
		self.EDeltas = NewEDeltas
		return 
				
	# Construct the sign from the number of intersections Inbetween the deltas. 
	# Term has been reDummy'd and the deltas have been sorted. 
	def ConstructSign(self):
		I=0
		d = len(self.Deltas)
		for i in range(d): 
			for j in range(i+1,d):
				if (((self.Deltas[i][0] < self.Deltas[j][0]) and (self.Deltas[j][0] < self.Deltas[i][1])) and ((self.Deltas[j][0] < self.Deltas[i][1]) and (self.Deltas[i][1] < self.Deltas[j][1]))): 
					I += 1 
					continue 
				if (((self.Deltas[j][0] < self.Deltas[i][0]) and (self.Deltas[i][0] < self.Deltas[j][1])) and ((self.Deltas[i][0] < self.Deltas[j][1]) and (self.Deltas[j][1] < self.Deltas[i][1]))): 
					I += 1 
					continue 
		return pow(-1,I) 
	def NEquivalentLines(self): 
		MyT = self.AllReferencedTensors()
		dim = len(MyT)
		Neq = 0 
		for T1 in range(dim): 
			for T2 in range(T1+1,dim): 
				if (T1 != T2):
					Tmp = self.ConnectionsBetween(T1,T2)
					for J in Tmp: 
						if J == 2: 
							Neq += 1 
						elif J > 2: 
							print "This is the wrong way to get the factor for this term. needs to be \prod 1/(neq!)"
							raise Exception("Algebra") 
		return Neq
	# This constructs a sign and factor based on diagrammatic arguements. 
	def DiagrammaticSignAndFactor(self): 
		NewSign = self.ConstructSign()
		NewFactor = pow((1./2.),self.NEquivalentLines())
		if NewSign*NewFactor == 0.0: 
			raise Exception("Sign Error..")
		self.PreFactor = NewSign*NewFactor
		return 		
	def DeltasBetween(self,t1,t2): 
		i1 = self.Tensors[t1].indices
		i2 = self.Tensors[t2].indices
		tore = []
		for D in self.Deltas:
			if ((D[0] in i1) and (D[1] in i2)):
				tore.append(D)
			elif ((D[1] in i1) and (D[0] in i2)):
				tore.append([D[1],D[0]])
		return tore
	def ConnectionsBetween(self,T1,T2): 
		tmp = [0,0,0,0,0,0,0]
		for D in self.DeltasBetween(T1,T2): 
			tmp[self.TypeOfDummy(D[0])] = tmp[self.TypeOfDummy(D[0])]+1
		return tmp

	def ConnectionsBetweenWithInd(self,T1,T2): 
		tmp = [0,0,0,0,0,0,0]
		i1 = T1.indices
		i2 = T2.indices
		tore = []
		for D in self.Deltas:
			if ((D[0] in i1) and (D[1] in i2)):
				tore.append(D)
			elif ((D[1] in i1) and (D[0] in i2)):
				tore.append([D[1],D[0]])

		for D in tore: 
			tmp[self.TypeOfDummy(D[0])] = tmp[self.TypeOfDummy(D[0])]+1
		return tmp

	def occNumber(self,TempTensors): 
#		print type(TempTensors)
		tmp = [0,0,0,0,0,0,0]
		for T in rlen(TempTensors.indices): 
		#	print TempTensors.indices[T]
			tmp[self.TypeOfDummy(T)] += 1
		return tmp
		
	def AdjacencyMatrix(self): 
		MyT = self.AllReferencedTensors()
		dim = len(MyT)
		AdjM = numpy.eye(dim,dim)
		for T1 in rlen(MyT): 
			for T2 in rlen(MyT): 
				if (T1 != T2): 
				 AdjM[T1][T2] = numpy.sum(self.ConnectionsBetween(T1,T2))
		return 0.1*AdjM

	def ContainsTensor(self,Tnam): 
		return (Tnam in self.AllReferencedTensors())
		
	# if self and other are related by permutations, 
	# then change the prefactor of self and Other.  
	def CombineIfLike(self,Other,Debug = False, Signed = True): 
		if Other.PreFactor == 0.0: 
			print "How the fuck did this come in? "
			raise Exception("wtf.")
	
		if (len(self.Deltas)*len(Other.Deltas) == 0): 
			return
		# JAP Feb 23 2012 - Removed sorting. 
		MyT = self.AllReferencedTensors()
		MyT.sort()   # Why in the name of god would I sort the tensors? 
		YourT = Other.AllReferencedTensors()
		YourT.sort()
	#	MyT = self.AllReferencedTensors()
	#	MyT.sort()   # Why in the name of god would I sort the tensors? 
	#	YourT = Other.AllReferencedTensors()
	#	YourT.sort()
		if (MyT != YourT): 
			return
		for tensor in rlen(MyT): 
			if self.Tensors[tensor].Time != Other.Tensors[tensor].Time : 
				return 
		else: 
			if Debug: 
				self.Print()
				Other.Print()
				print "MyT, YourT: ", MyT, YourT			
			for T1 in rlen(MyT): 
				if (self.OpenTypesInTensor(T1) != Other.OpenTypesInTensor(T1)):
					if Debug: 
						print "Fails Open Types...: ", self.OpenTypesInTensor(T1), " VS ", Other.OpenTypesInTensor(T1)
					return 
				for T2 in range(T1+1,len(MyT)): 	
					if self.ConnectionsBetween(T1,T2) != Other.ConnectionsBetween(T1,T2):
						return
			Sign1 = 1
			Sign2 = 1
			if (Signed): 
				Sign1 = self.ConstructSign()
				Sign2 = Other.ConstructSign()
			if Debug: 
				print "COMBINING !!!"
				self.Print()
				print "with: "
				Other.Print()			
			if (Signed): 
				self.PreFactor += Other.PreFactor*Sign1*Sign2
			Other.PreFactor = 0.0
		return 
	# -----------------------------------
	# for contraction. 
	# returns (which tensor, which index)
	def LocationOfDummyIndex(self,I):
		rng = range(len(self.Tensors))
		for T in rng: 
			if (self.Tensors[T].Contains(I)): 
				return (T,self.Tensors[T].indices.index(I))
		print "Couldn't find dummy ", I," in: " 
		self.Print()
		
	def SummedBetween(self,ainds,binds):
		SummedinA = []
		SummedinB = []
		for A in ainds: 
			for B in binds: 
				for D in self.Deltas: 
					if (D[0] == A and D[1] == B): 
						SummedinA.append(A)
						SummedinB.append(B)
					elif (D[0] == B and D[1] == A): 
						SummedinA.append(A)
						SummedinB.append(B)
		return (SummedinA,SummedinB)
	def MyDataFrom(self,VectorOfTensors): 
		if (not self.IsSimple()): 
			print "Error... MyData, but I'm not simple."
			return
		else :  
			return self.Tensors[0].MyDataFrom(VectorOfTensors)
	def IsSimple(self): 
		return (len(self.Deltas) == 0 and (len(self.Tensors) == 1))
	def ContractsToMe(self,Other): 
		return (self.NumberOfEachType() == Other.NumberOfEachType()) 
	def ContainsVacOf(self, Other): 
		# Other should be simple 
		# and I should have a vac with that name. 
		if ( not Other.IsSimple()): 
			print "Error, Other should be simple"
			raise Exception("fuuuuu...... ")			
		# Make sure the vac type is my type.
		for T in self.Tensors: 
			if (T.NonVacName().count(Other.Tensors[0].Name()) > 0): 
				if (T.IsVac()):
					return True
		return False
	def AllIndices(self): 
		tore=[]
		for T in self.Tensors: 
			tore.extend(T.indices) 
		return tore
	def ContractedIndices(self): 
		tore = []
		for D in self.Deltas: 
			tore.extend(D)
		return tore
	def OpenIndices(self): 
		return ListComplement(self.AllIndices(),self.ContractedIndices())
	def OpenTypesInTensor(self,T1):
		tmp = [0,0,0,0,0,0,0]
		Os = self.Tensors[T1].ops
		Oo = self.OpenIndices()
		for X in Os: 
			if (Oo.count(X[1]) > 0): 
				tmp[abs(X[0])] = 1 + tmp[abs(X[0])]
		return tmp

	def OpIndices(self): 
		tore = []
		dim = range(len(self.Ops))
		for d in dim: 
			if (self.Ops[d][0] > 0):
				tore.append(self.Ops[d][1])
		return tore

	def TensorProduct(self, other):
		self.PreFactor *= other.PreFactor
		OtherDeltas = copy.deepcopy(other.Deltas)
		#shift other dummy indices above the max index of this tensor. 
		OtherShift = max(map(lambda ZZ: ZZ[1],self.Ops))+1
		tmp = copy.deepcopy(other.Tensors)
		for UU in tmp: 
			UU.Shift(OtherShift)
		self.Tensors.extend(tmp) 
		DDIM  = range(len(OtherDeltas))
		for d in DDIM: 
			OtherDeltas[d][0] += OtherShift
			OtherDeltas[d][1] += OtherShift
		self.Deltas.extend(OtherDeltas) 
		OtherOps = copy.deepcopy(other.Ops)
		for x in OtherOps: 
			x[1] += OtherShift
		self.Ops.extend(OtherOps)
		return self
		
	def Conjugate(self,ii): 
		if (ii == 1): 
			return 3
		elif (ii == 3) :
			return 1 
		elif (ii == 2): 
			return 4
		elif (ii == 4): 
			return 2
		elif (ii == 5): 
			return 6
		elif (ii == 6): 
			return 5
		else : 
			print "Unknown type: ", ii 
		return 0 

	# number of external indices of each type. 
	def NumberOfEachType(self,AlsoContracted = False):
		tore = [0 for x in range(7)]
		for x in self.Ops: 
			if (AlsoContracted): 
				tore[abs(x[0])] += 1
			elif (x[0] > 0): 
				tore[x[0]] += 1
		return tore
	def QPRank(self):
		tmp = self.NumberOfEachType()
		return (tmp[1]+tmp[2]-(tmp[3]+tmp[4]))/2
	def ConservesParticleNumber(self):
		tmp = self.NumberOfEachType()
		return (tmp[5] == tmp[6] and ((tmp[4] == tmp[2]) and (tmp[3] == tmp[1])))
	# if this term can be closed by OtherTypes (at all, before contraction) 
	def MayCloseWith(self,OtherTypes):
		if (len(OtherTypes) == 0): 
			return True
		SelfTypes = self.NumberOfEachType()
		for T in OtherTypes: 
			if (SelfTypes[1]>=T[3] and SelfTypes[2]>=T[4] and SelfTypes[5]>=T[6] and SelfTypes[3]>=T[1] and SelfTypes[4]>=T[2] and SelfTypes[6]>=T[5]):
				return True
		return False
	# this is useful if we accelerate wick. 
	def MayCloseAfterDelta(self,op1,op2,OtherTypes)	:
		if (len(OtherTypes) == 0): 
			return True
		SelfTypes = self.NumberOfEachType()
		SelfTypes[self.Ops[op1][0]] = SelfTypes[self.Ops[op1][0]] - 1
		SelfTypes[self.Ops[op2][0]] = SelfTypes[self.Ops[op2][0]] - 1		
		for T in OtherTypes: 
			if (SelfTypes[1]>=T[3] and SelfTypes[2]>=T[4] and SelfTypes[5]>=T[6] and SelfTypes[3]>=T[1] and SelfTypes[4]>=T[2] and SelfTypes[6]>=T[5]):
				return True
		return False
	# if this term closes with othertypes (after contraction) 
	def ClosesWith(self,OtherTypes):
		if (len(OtherTypes) == 0): 
			return True	
		SelfTypes = self.NumberOfEachType()
		for T in OtherTypes: 
			if (SelfTypes[1]==T[3] and SelfTypes[2]==T[4] and SelfTypes[5]==T[6] and SelfTypes[3]==T[1] and SelfTypes[4]==T[2] and SelfTypes[6]==T[5]):
				return True
#		print "Rejecting ", SelfTypes, " versus: ", OtherTypes
		return False
	def TracesWith(self, Other):
		return self.NumberOfEachType() == Other.NumberOfEachType()			
	def IsNormal(self,Vac): 
		dim = range(len(self.Ops))
		if (Vac == 0): 
			for it1 in dim: 			
				# if 1 isn't contracted. 
				if (self.type(it1) > 0) :
					it2 = it1 + 1
					while (it2 < len(self.Ops)):
						if (self.type(it2) > 0) : 
							# hole creation (a^\dagger_i) = 4 
							# particle annihilation (a_a) = 3
							# hole ann. (a_i) = 2
							# particle creation (a^\dagger_a) = 1
							# The four anticommutators:
							if ((self.type(it1) == 4)  and (self.type(it2) == 2) ): 
								return False
							elif ((self.type(it1) == 3)  and (self.type(it2) == 1) ): 
								return False
							elif ((self.type(it1) == 3)  and (self.type(it2) == 2) ): 
								return False
							elif ((self.type(it1) == 4)  and (self.type(it2) == 1) ): 
								return False
						it2 += 1
			return True
		elif (Vac == 1):
			for it1 in dim: 			
				# if 1 isn't contracted. 
				if (self.type(it1) > 0) :
					it2 = it1 + 1
					while (it2 < len(self.Ops)):
						if (self.type(it2) > 0) : 
							if ((self.type(it1) == 6)  and (self.type(it2) == 5) ): 
								return False
						it2 += 1
		return True
	# pass a string of 1's and 0's the same length as number of operators. 
	# returns a copy of self(general operator string) with those specified. 
	def SpecifyHP(self,atypes): 
		tmp = copy.deepcopy(self)
		Rng = range(len(self.Ops))
		for R in Rng: 
			if self.Ops[R][0] == 5: 
				if (atypes[R] == 0): # hole = occ. 
					tmp.Ops[R][0] = 4
				elif (atypes[R] == 1): # part = virt. 
					tmp.Ops[R][0] = 1
			if self.Ops[R][0] == 6: 
				if (atypes[R] == 0): # hole = occ. 
					tmp.Ops[R][0] = 2
				elif (atypes[R] == 1): # part = virt. 
					tmp.Ops[R][0] = 3	
		# Replace my tensor with hole-particle tensors. 
		#( I should be simple)
		if (not tmp.IsSimple()): 
			print "Can't Specify HP for product tensors. "
		# Get a map between dummy indices and O or V
		HoleOrP=dict()
		for R in Rng: 
			HoleOrP[self.Ops[R][1]] = atypes[R]
		# fix the tensors 
		oldname = tmp.Tensors[0].name
		tmp.Tensors = []
		tmp.Tensors.append(TensorIndexer(oldname, tmp.Ops, False, True)) 
		return tmp

	# Returns a list where each general index of 1 operator 
	# is replaced with hole or particle indices 
	# using binary encoding trick. 
	def ToFermi(self): 
		itemp = len(self.Ops)
		TmpLst = ManyOpStrings()
		maxstr = "0b"+"".join(["1" for i in range(itemp)])
		for i in range(0,int(maxstr,2)+1): 
			s = bin(i)[2:]
			ss = list(s) 
			# pad 0 left such that len(s) = itemp
			while (len(ss) < itemp):
				ss.insert(0,'0')
			G = map(int, ss)
			TmpLst.Add([self.SpecifyHP(G)])
		return TmpLst
		
		
	# For Wick's Theorem ---------------------------
	# Just places all the operators from the second 
	# string after these and muliplies the factors.
	# This will probably get Normal ordered next. 
		
	# These return the results of applying anticommutator. 
	# always the d's are in order the tensors appear. 
	def delta(self,n1,n2):
		Z = copy.deepcopy(self)
		Z.Ops[n1][0] *= -1
		Z.Ops[n2][0] *= -1
		if n1<n2: 
			Z.Deltas.append([Z.Ops[n1][1],Z.Ops[n2][1]])
		else:
			Z.Deltas.append([Z.Ops[n2][1],Z.Ops[n1][1]]) 
		return Z
	def swap(self,n1,n2):
		Z = copy.deepcopy(self)
		Z.Ops[n1], Z.Ops[n2] = Z.Ops[n2], Z.Ops[n1]
		Z.PreFactor *= -1.0
		return Z
	def type(self,n1):
		return self.Ops[n1][0]			
	# apply one anticommutator if any pair of operators 
	# are (Genuine Vac) out-of-order 
	# returns (If Applied, and a list of between 1-2 strings of operators depending on result)
	# (Summation is impled on that list)
	# Creation (hole and particle) = 5 
	# Annihilation (hole and particle) = 6
	def AntiCommuteGenuine(self, Closing = []): 
		dim = range(len(self.Ops))
		for it1 in dim: 			
			# if 1 isn't contracted. 
			if (self.type(it1) > 0) :
				it2 = it1 + 1
				while (it2 < len(self.Ops)):
					if (self.type(it2) < 0) : 
						it2 += 1
					elif (self.type(it2) > 0) : 
						if ((self.type(it1) == 6)  and (self.type(it2) == 5) ): 
							if (self.MayCloseAfterDelta(it1,it2,Closing)):
								return (True, [self.delta(it1,it2),self.swap(it1,it2)] )
							else:
								return (True, [self.swap(it1,it2)] ) 
						break
					it2 += 1
		return (False, [])
	# apply one anticommutator if any pair of operators 
	# are (FermiVac) out-of-order 
	def AntiCommuteFermi(self, Closing = []): 
		dim = range(len(self.Ops))
		for it1 in dim: 			
			# if 1 isn't contracted. 
			if (self.type(it1) > 0) :
				it2 = it1 + 1
				while (it2 < len(self.Ops)):
					if (self.type(it2) < 0) : 
						it2 += 1
					elif (self.type(it2) > 0) : 
						# hole creation (a^\dagger_i) = 4 
						# particle annihilation (a_a) = 3
						# hole ann. (a_i) = 2
						# particle creation (a^\dagger_a) = 1
						# Normal rule (+a,i)<(+h,p) = (1,2)<(4,3)
						# The four anticommutators:
						if ((self.type(it1) == 4)  and (self.type(it2) == 2) ): 
							if (self.MayCloseAfterDelta(it1,it2,Closing)):
								return (True, [self.delta(it1,it2),self.swap(it1,it2)] )
							else:
								return (True, [self.swap(it1,it2)] ) 
						elif ((self.type(it1) == 3)  and (self.type(it2) == 1) ): 
							if (self.MayCloseAfterDelta(it1,it2,Closing)):
								return (True, [self.delta(it1,it2),self.swap(it1,it2)] )
							else:
								return (True, [self.swap(it1,it2)] ) 								
						elif ((self.type(it1) == 3)  and (self.type(it2) == 2) ): 
							return (True, [self.swap(it1,it2)])
						elif ((self.type(it1) == 4)  and (self.type(it2) == 1) ): 
							return (True, [self.swap(it1,it2)])
						# This is just a matter of taste. 
						# it's nice to have a+p before a_h, and a+h before a_p 
						#	to align with amplitudes
						elif ((self.type(it1) == 2)  and (self.type(it2) == 1) ): 
							return (True, [self.swap(it1,it2)])
						elif ((self.type(it1) == 3)  and (self.type(it2) == 4) ): 
							return (True, [self.swap(it1,it2)])
						break
					it2 += 1
		return (False, [])	
		
#	Printing routines and whatnot. 
	def OpAsString(self,oo):
		TempStrings = [" ","(a+_p","(a_h","(a_p","(a+_h","(a+","(a"]
		NegTempStrings = [" ","(A+_p","(A_h","(A_p","(A+_h","(A+","(A"]
		if (oo[0] < 0):
			return NegTempStrings[abs(oo[0])]+str(abs(oo[1]))+")"
		else :
			return TempStrings[abs(oo[0])]+str(abs(oo[1]))+")"	
	def Print(self, LatexFormat = False): 
		if (LatexFormat): 
			if(self.PreFactor == -1.0):
				PreFactorString = "-"
			elif(self.PreFactor == 0.5):
				PreFactorString = "\\frac{1}{2}"
			elif(self.PreFactor == -0.5):
				PreFactorString = "\\frac{-1}{2}"
			elif(self.PreFactor == 0.25):
				PreFactorString = "\\frac{1}{4}"
			elif(self.PreFactor == -0.25):
				PreFactorString = "\\frac{-1}{4}"
			elif(self.PreFactor == 1.0):
				PreFactorString = ""
			else:
				print self.PreFactor
			TensorString=" "
			LHS=" "
			TensorString = TensorString + PreFactorString
			for UU in self.Tensors:
				if(str(UU.Name()) == 'r1_Vac_ph'):
					LHS = "o^"+ str(UU.indices[0]) + "_" + str(UU.indices[1]) + " \leftarrow "
				elif(str(UU.Name()) == 'B_hh'):
					TensorString = TensorString+"B_{hh}^" + str(UU.indices) +"(" + UU.Time + ") "
				elif(str(UU.Name()) == 'B_pp'):
					TensorString = TensorString+"B_{pp}^" + str(UU.indices) +"(" + UU.Time + ") "
				elif(str(UU.Name()) == 'B_hp'):
					TensorString = TensorString+"B_{hp}^" + str(UU.indices) +"(" + UU.Time + ") "
				elif(str(UU.Name()) == 'B_ph'):
					TensorString = TensorString+"B_{ph}^" + str(UU.indices) +"(" + UU.Time + ") "
				elif(str(UU.Name()) == 'v_hhhh'):
					TensorString = TensorString+"V_{hhhh}^" + str(UU.indices) +"(" + UU.Time + ") "
				elif(str(UU.Name()) == 'v_hhhp'):
					TensorString = TensorString+"V_{hhhp}^" + str(UU.indices) +"(" + UU.Time + ") "
				elif(str(UU.Name()) == 'v_hhpp'):
					TensorString = TensorString+"V_{hhpp}^" + str(UU.indices) +"(" + UU.Time + ") "
				elif(str(UU.Name()) == 'v_hppp'):
					TensorString = TensorString+"V_{hppp}^" + str(UU.indices) +"(" + UU.Time + ") "
				elif(str(UU.Name()) == 'v_pppp'):
					TensorString = TensorString+"V_{pppp}^" + str(UU.indices) +"(" + UU.Time + ") "
				elif(str(UU.Name()) == 'v_ppph'):
					TensorString = TensorString+"V_{ppph}^" + str(UU.indices) +"(" + UU.Time + ") "
				elif(str(UU.Name()) == 'v_pphh'):
					TensorString = TensorString+"V_{pphh}^" + str(UU.indices) +"(" + UU.Time + ") "
				elif(str(UU.Name()) == 'v_phhh'):
					TensorString = TensorString+"V_{phhh}^" + str(UU.indices) +"(" + UU.Time + ") "
				elif(str(UU.Name()) == 'v_phpp'):
					TensorString = TensorString+"V_{phpp}^" + str(UU.indices) +"(" + UU.Time + ") "
				elif(str(UU.Name()) == 'v_pphp'):
					TensorString = TensorString+"V_{pphp}^" + str(UU.indices) +"(" + UU.Time + ") "
				elif(str(UU.Name()) == 'v_hphh'):
					TensorString = TensorString+"V_{hphh}^" + str(UU.indices) +"(" + UU.Time + ") "
				elif(str(UU.Name()) == 'r1_ph'):
					TensorString = TensorString+"o^"+ str(UU.indices[0]) + "_" + str(UU.indices[1]) +" " 
				else:
					print "FIXXXXXX THIIIISSSSSSSSSSSS"
					print UU.Name()
				TensorString = TensorString.replace("[", "{")
				TensorString = TensorString.replace("]", "}")
			DeltaString = ""
			for d in self.Deltas:
				DeltaString = DeltaString+"d("+str(d[0])+","+str(d[1])+")"
			OpString = ""	
			for o in self.Ops:
				OpString = OpString+self.OpAsString(o)
			if (len(self.EDeltat) > 0) :
				Tup=""
				Tdown=""
				Sup=""
				Sdown=""
				for i in rlen(self.EDeltat): 
					if self.EDeltat[i][0] > 0: 
						Tup = Tup+str(self.EDeltat[i][1])
					else:
						Tdown = Tdown+str(self.EDeltat[i][1]) 
				for i in rlen(self.EDeltas): 
					if self.EDeltas[i][0] > 0: 
						Sup = Sup+str(self.EDeltas[i][1])
					else:
						Sdown = Sdown+str(self.EDeltas[i][1]) 
				Texp = " e^{i(\Delta^{"+Tup+"}_{" + Tdown +"})t}"	
				Sexp = " e^{i(\Delta^{"+Sup+ "}_{" + Sdown +"})s}"	
				Texp = Texp.replace("[", "{")
				Sexp = Sexp.replace("[", "{")
				Texp = Texp.replace("]", "}")
				Sexp = Sexp.replace("]", "}")
				print LHS + DeltaString+TensorString+Texp+" \int_{t_0}^t "+Sexp+"C(t-s) ds" + "  \\notag \\\\"
			else : 
				print LHS+DeltaString+TensorString+"\int_{t_0}^t "+"C(t-s) ds" + " \\notag \\\\"
			return
		print "--- Operator ---" 
		PreFactorString = "("+str(self.PreFactor)+")*"
		TensorString=" "
		for UU in self.Tensors: 
			TensorString = TensorString+str(UU.Name())+UU.Time+"("+str(UU.indices)+")*"
		DeltaString = ""
		for d in self.Deltas:
			DeltaString = DeltaString+"d("+str(d[0])+","+str(d[1])+")*"
		OpString = ""	
		for o in self.Ops:
			OpString = OpString+self.OpAsString(o)
		if (len(self.EDeltat) > 0) :
			Texp = "*e^(i("+str(self.EDeltat)+")t)"	
			Sexp = "*e^(i("+str(self.EDeltas)+")s)"	
			BosonCF = "B("+str(self.BCIndices)+")"
			print PreFactorString+DeltaString+TensorString+"\n"+OpString+Texp+"\n"+Sexp+BosonCF
		else : 
			print PreFactorString+DeltaString+TensorString+"\n"+OpString
		return


#   ----------------------------------------------------------------
#   EVERYTHING BELOW HERE is specific to the Polarization Propagator
#   ----------------------------------------------------------------

	def IncorporateDeltas(self): 
		OldIndices = []
		for T in self.Tensors: 
			OldIndices.extend(T.OpIndices())
		Mep = dict()
		for i in range(len(OldIndices)):
			Mep[OldIndices[i]] = OldIndices[i]
			for j in self.Deltas: 
				if (OldIndices[i] == j[1]): 
					Mep[OldIndices[i]] = j[0]
					break
		Pos = 0 
		NewOps = []
		for T in self.Tensors: 
			T.ReDummy(Mep)
		for i in range(len(self.Ops)):
			NewOps.append([self.Ops[i][0],Mep[self.Ops[i][1]]])
		self.Ops = NewOps
		return 
		
	# this is a hack to gather the boson correlation function and interaction exponential
	# I say that because the notation for the term is changed after this has been called. 
	# and only really useful for evaluating the term, not 
	# it should be called after Normal ordering is complete, and is specific to 2nd order polarization propagator.
	def AssignBCFandTE(self): 	
		self.IncorporateDeltas()
		self.Deltas = []
		self.ReDummy()
		Deltas = []
		Deltat = []
		Deltai = []
		IsAnInhomogeneousTerm = False
		IsAnUndressedTerm = False		
		for T in self.Tensors:
			#print T.Time
			if T.Time == "i":
				IsAnInhomogeneousTerm = True
		for T in self.Tensors:
			if T.name == "B":
				IsAnUndressedTerm = True				
		BCFU = []
		BCFL = []
		for T in self.Tensors:
			if (T.Time == "t"): 
				for O in T.ops:
					if abs(O[0]) == 1:
						Deltat.append([1,O[1]]) # meaning + epsilon_O[1]
						# I'm abusing the fact that integrals should be create,create, ann ann... 
						if (T.name == "v"):
							BCFU.append(O[1]) # meaning it's [V, dummy index1]
					elif abs(O[0]) == 2: 
						Deltat.append([-1,O[1]])
						if (T.name == "v"):							
							BCFU.append(O[1]) # meaning it's [O, dummy index1]
					elif abs(O[0]) == 3:
						Deltat.append([-1,O[1]])
						if (T.name == "v"):										 
							BCFU.append(O[1]) # meaning it's [V, dummy index1]
					elif abs(O[0]) == 4: 
						Deltat.append([1,O[1]])
						if (T.name == "v"):							
							BCFU.append(O[1]) # meaning it's [O, dummy index1]
					else : 
						print "Can't assign BCF, unknown type."
			elif (T.Time == "s"): 
				for O in T.ops: 
					if abs(O[0]) == 1:
						Deltas.append([1,O[1]])
						if (T.name == "v"):
							BCFL.append(O[1])
					elif abs(O[0]) == 2: 
						Deltas.append([-1,O[1]])
						if (T.name == "v"):
							BCFL.append(O[1])
					elif abs(O[0]) == 3:
						Deltas.append([-1,O[1]])			 
						if (T.name == "v"):
							BCFL.append(O[1]) 
					elif abs(O[0]) == 4: 
						Deltas.append([1,O[1]])
						if (T.name == "v"):
							BCFL.append(O[1]) 
					else : 
						print "Can't assign BCF, unknown type."
			elif (T.Time == "i"): 
				for O in T.ops: 
					if abs(O[0]) == 1:
						Deltai.append([1,O[1]])
					elif abs(O[0]) == 2: 
						Deltai.append([-1,O[1]])
					elif abs(O[0]) == 3:
						Deltai.append([-1,O[1]])			 
					elif abs(O[0]) == 4: 
						Deltai.append([1,O[1]])
					else : 
						print "Can't assign BCF, unknown type."
			elif (T.Time == ""): 
				if (IsAnInhomogeneousTerm): 
					continue 
				# Wow... So wow... 
				# I totally fucked up the way the deltas were being calculated! Seriously! Wow! 
				for O in T.ops:
					if abs(O[0]) == 1:
						Deltat.append([1,O[1]]) # meaning + epsilon_O[1]
					elif abs(O[0]) == 2: 
						Deltat.append([-1,O[1]])
					elif abs(O[0]) == 3:
						Deltat.append([-1,O[1]])			 
					elif abs(O[0]) == 4: 
						Deltat.append([1,O[1]])
					else : 
						print "Can't assign HCF, unknown type."
			else: 
				print "UNKNOWN TIME: ", T.Time
				raise Exception("BadTime")
		# Cancel duplicates from exp(i\Deltat) 
		# this assumes they have opposite signs. 
		tmp = [X[1] for X in Deltat]
		DeltatNew=[]
		for X in Deltat: 
			if tmp.count(X[1]) == 1 : 
				DeltatNew.append(X)
		self.EDeltat = copy.deepcopy(DeltatNew)
		self.EDeltas = copy.deepcopy(Deltas)
		self.EDeltai = copy.deepcopy(Deltai)
		EDTIndices = [X[1] for X in self.EDeltat]
		EDTTypes = map(self.HPTypeDummy,EDTIndices)
		EDSIndices = [X[1] for X in self.EDeltas]		
		EDSTypes = map(self.HPTypeDummy,EDSIndices)
		EDIIndices = [X[1] for X in self.EDeltai]		
		EDITypes = map(self.HPTypeDummy,EDIIndices)		
		self.EDTTensor = numpy.zeros(shape = (len(self.EDeltat),max(Params.nocc,Params.nvirt)),dtype=float)
		self.EDSTensor = numpy.zeros(shape = (4,max(Params.nocc,Params.nvirt)),dtype=float)
		self.EDITensor = numpy.zeros(shape = (4,max(Params.nocc,Params.nvirt)),dtype=float)
		eo = Integrals["h_hh"].diagonal().copy()
		ev = Integrals["h_pp"].diagonal().copy()
		if (len(eo) != len(ev)): 
			eo = numpy.append(eo,numpy.zeros(max(Params.nocc,Params.nvirt) - min(Params.nocc,Params.nvirt)))
		try: 
			for i in rlen(self.EDeltat): 
				if (EDTTypes[i] == 0): 
					self.EDTTensor[i] += (self.EDeltat[i][0])*eo
				elif (EDTTypes[i] == 1): 
					self.EDTTensor[i] += (self.EDeltat[i][0])*ev
			if (not IsAnInhomogeneousTerm and not IsAnUndressedTerm): 
				for i in range(4): 				
					if (EDSTypes[i] == 0):
						self.EDSTensor[i] += (self.EDeltas[i][0])*eo
					elif (EDSTypes[i] == 1): 
						self.EDSTensor[i] += (self.EDeltas[i][0])*ev
			elif (IsAnUndressedTerm): 
				for i in range(2): 				
					if (EDSTypes[i] == 0):
						self.EDSTensor[i] += (self.EDeltas[i][0])*eo
					elif (EDSTypes[i] == 1): 
						self.EDSTensor[i] += (self.EDeltas[i][0])*ev	
			elif (IsAnInhomogeneousTerm): 
				for i in range(4): 				
					if (EDITypes[i] == 0):
						self.EDITensor[i] += (self.EDeltai[i][0])*eo
					elif (EDITypes[i] == 1): 
						self.EDITensor[i] += (self.EDeltai[i][0])*ev
		except Exception as Ex:
			self.Print()
			raise
		BCFU.extend(BCFL)
		self.BCIndices = copy.deepcopy(BCFU) 
		self.BCIndexShifts = numpy.array([ 0 if X == 0 else Params.nocc for X in map(self.HPTypeDummy,self.BCIndices) ])
		self.UniqueBCIndices = list(set(self.BCIndices))
		return	

	# To get a vague idea of the effects of the bath correlation function 
	def MakeMatrixMarkov(self): 
		print "Making this term's Markovian Relaxation Matrix... "
		self.Print()
		CI = self.CurrentIntegral
		EDSTensor = self.EDSTensor
		MTil = self.BosInf.MTilde
		Freqs = self.BosInf.Freqs
		Coths = self.BosInf.Coths
		Css = self.BosInf.Css
		smax = self.BosInf.NBos
		Unique = self.UniqueBCIndices		
		# 4-19-2012 It can happen that the unique indices2 aren't contiguous
		# Check to make sure that isn't broken here. 
		LoopDim = len(self.UniqueBCIndices)
		try : 
			RedIndicies = [self.UniqueBCIndices.index(self.BCIndices[i]) for i in range(8)]
			RedTypes = map(self.HPTypeDummy, RedIndicies)
			RedShifts = [ Params.nocc if X == 1 else 0 for X in RedTypes]
			SIndices = [ self.UniqueBCIndices.index(X[1]) for X in self.EDeltas ] 													
			TypesOfSummed = map(self.HPTypeDummy,self.UniqueBCIndices)
			ShapeOfSummed = [ Params.nocc if X==0 else Params.nvirt for X in TypesOfSummed]
		except Exception as Ex: 
			self.Print()
			raise		
		Adia = Params.Adiabatic 
		UVCut = Params.UVCutoff		
		
		TmpPiece = numpy.zeros(shape=CI.shape,dtype = complex)
		DeltaPiece = numpy.zeros(shape=CI.shape,dtype = complex)
		EqPiece = numpy.zeros(shape=CI.shape,dtype = float)
		# The dynamic piece is separated for each s
		tmp = list(CI.shape)
		tmp.insert(0,smax)
		DynPiece = numpy.zeros(shape=tuple(tmp),dtype = complex)
		code =	"""
				#line 1214 "Wick.py"
				using namespace std;
				int ri[8]; 
				int rs[8];
				int si[4]; 
				int ci[6];
				for (int k = 0 ; k<8; ++k)
				{
					ri[k] = (int)RedIndicies[k];
					rs[k] = (int)RedShifts[k]; // Picks out Occupied or V
				}
				for (int k = 0 ; k<4; ++k)
				{
					si[k] = (int)SIndices[k];
				}
				for (int k = 0 ; k<(int)LoopDim; ++k)
				{
					ci[k] = (int)ShapeOfSummed[k]; 					
				}	
				complex<double> NewBCF; 
				complex<double> imi(0.0,1.0); 
				complex<double> mimi(0.0,-1.0); 				
				double TildeSum1 = 0.0; 
				double TildeSum2 = 0.0;
				double es;  				
				int It[6];
				if (LoopDim == 4)
				{
					for (It[0] = 0 ; It[0] < ci[0] ; It[0]=It[0]+1)
					{
						for (It[1] = 0 ; It[1] < ci[1] ; It[1]=It[1]+1)
						{
							for (It[2] = 0 ; It[2] < ci[2] ; It[2]=It[2]+1)
							{
								for (It[3] = 0 ; It[3] < ci[3] ; It[3]=It[3]+1)
								{
									es = EDSTensor(0,It[si[0]]) + EDSTensor(1,It[si[1]]) + EDSTensor(2,It[si[2]]) + EDSTensor(3,It[si[3]]); 
									DeltaPiece(It[0],It[1],It[2],It[3]) = es;
									if (abs(es) > UVCut)
										continue;
									for (int s = 0; s<smax ; ++s)
									{
										TildeSum1 = (MTil(s,It[ri[0]]+rs[0])+MTil(s,It[ri[1]]+rs[1])-MTil(s,It[ri[2]]+rs[2])-MTil(s,It[ri[3]]+rs[3]));
										TildeSum2 = (MTil(s,It[ri[4]]+rs[4])+MTil(s,It[ri[5]]+rs[5])-MTil(s,It[ri[6]]+rs[6])-MTil(s,It[ri[7]]+rs[7]));
										DynPiece(s,It[0],It[1],It[2],It[3]) -= (TildeSum1)*(TildeSum2);
										EqPiece(It[0],It[1],It[2],It[3]) -= Coths(s)*(TildeSum1+TildeSum2)*(TildeSum1+TildeSum2)/2.0;											
									}
								}								
							}					
						}
					}
				}
				else if (LoopDim == 5)
				{
					for (It[0] = 0 ; It[0] < ci[0] ; It[0]=It[0]+1)
					{
						for (It[1] = 0 ; It[1] < ci[1] ; It[1]=It[1]+1)
						{
							for (It[2] = 0 ; It[2] < ci[2] ; It[2]=It[2]+1)
							{
								for (It[3] = 0 ; It[3] < ci[3] ; It[3]=It[3]+1)
								{
									for (It[4] = 0 ; It[4] < ci[4] ; It[4]=It[4]+1)
									{
										es = EDSTensor(0,It[si[0]]) + EDSTensor(1,It[si[1]]) + EDSTensor(2,It[si[2]]) + EDSTensor(3,It[si[3]]); 
										DeltaPiece(It[0],It[1],It[2],It[3],It[4]) = es;
										if (abs(es) > UVCut)
											continue;
										for (int s = 0; s<smax ; ++s)
										{
											TildeSum1 = (MTil(s,It[ri[0]]+rs[0])+MTil(s,It[ri[1]]+rs[1])-MTil(s,It[ri[2]]+rs[2])-MTil(s,It[ri[3]]+rs[3]));
											TildeSum2 = (MTil(s,It[ri[4]]+rs[4])+MTil(s,It[ri[5]]+rs[5])-MTil(s,It[ri[6]]+rs[6])-MTil(s,It[ri[7]]+rs[7]));
											DynPiece(s,It[0],It[1],It[2],It[3],It[4]) -= (TildeSum1)*(TildeSum2);
											EqPiece(It[0],It[1],It[2],It[3],It[4]) -= Coths(s)*(TildeSum1+TildeSum2)*(TildeSum1+TildeSum2)/2.0;						
										}
									}															
								}								
							}					
						}
					}
				}
				else if (LoopDim == 6) 
				{
					for (It[0] = 0 ; It[0] < ci[0] ; It[0]=It[0]+1)
					{
						for (It[1] = 0 ; It[1] < ci[1] ; It[1]=It[1]+1)
						{
							for (It[2] = 0 ; It[2] < ci[2] ; It[2]=It[2]+1)
							{
								for (It[3] = 0 ; It[3] < ci[3] ; It[3]=It[3]+1)
								{
									for (It[4] = 0 ; It[4] < ci[4] ; It[4]=It[4]+1)
									{
										for (It[5] = 0 ; It[5] < ci[5] ; It[5]=It[5]+1)
										{
											es = EDSTensor(0,It[si[0]]) + EDSTensor(1,It[si[1]]) + EDSTensor(2,It[si[2]]) + EDSTensor(3,It[si[3]]); 
											DeltaPiece(It[0],It[1],It[2],It[3],It[4],It[5]) = es;
											if (abs(es) > UVCut)
												continue;											
											for (int s = 0; s<smax ; ++s)
											{
												TildeSum1 = (MTil(s,It[ri[0]]+rs[0])+MTil(s,It[ri[1]]+rs[1])-MTil(s,It[ri[2]]+rs[2])-MTil(s,It[ri[3]]+rs[3]));
												TildeSum2 = (MTil(s,It[ri[4]]+rs[4])+MTil(s,It[ri[5]]+rs[5])-MTil(s,It[ri[6]]+rs[6])-MTil(s,It[ri[7]]+rs[7]));
												DynPiece(s,It[0],It[1],It[2],It[3],It[4],It[5]) -= (TildeSum1)*(TildeSum2);
												EqPiece(It[0],It[1],It[2],It[3],It[4],It[5]) -= Coths(s)*(TildeSum1+TildeSum2)*(TildeSum1+TildeSum2)/2.0;						
											}
										} 
									}															
								}								
							}					
						}
					}
				}	
				else 
				{
					cout << "ERROR... WRONG NUM BOSON INDICES " << endl; 
				}
				"""
		weave.inline(code,
					['CI','EDSTensor','MTil','Freqs','ShapeOfSummed', 'DeltaPiece', 'EqPiece', 'DynPiece',
					'Coths', 'smax','Unique','LoopDim','RedIndicies','RedTypes','RedShifts','SIndices','UVCut'],
					headers = ["<complex>","<iostream>"],
					global_dict = dict(), 					
					type_converters = scipy.weave.converters.blitz, 
					compiler = Params.Compiler,
					verbose = 1)
		
		Equil = numpy.exp(EqPiece)
		print "Integrating ... Average Delta: ", numpy.average(DeltaPiece), " Stdev: ", numpy.std(DeltaPiece), "Maximum Frequency: ", numpy.max(numpy.abs(DeltaPiece)) , "Minimum Frequency: ", numpy.min(numpy.abs(DeltaPiece))
		T3 = numpy.array([-sqrt(3.0/5.0),0.0,sqrt(3.0/5.0)])
		W3 = numpy.array([5.0/9.0,8.0/9.0,5.0/9.0])

		TMax = Params.TMax
		Eta = 0.0
		if (Params.ContBath): 
			print "Continuous and Markov Bath is being used." 
			TMax = Params.Tc*numpy.sqrt(2.0/(self.BosInf.beta*self.BosInf.OmegaCThree))
			print "TMax:", TMax 			
			
		else:
			Eta = Params.MarkovEta
			TMax = 9.0/Params.MarkovEta 

		Step = Params.TStep*5.0
		print "3rd-Order Gaussian Quadrature, Step: ", Step
		CI *= 0.0
		f0 = numpy.zeros(TmpPiece.shape)
		f1 = numpy.zeros(TmpPiece.shape)
		f2 = numpy.zeros(TmpPiece.shape)		
		TR = int(self.TimeReversed)
		if TR: 
			DeltaPiece *= -1.0
		for T in numpy.arange(0.0,TMax,Step):
		
			a = T
			b = T + Step
			ts = T3*((b-a)/2.0) + ((a+b)/2.0)

			TNow = ts[0]
			self.BosInf.SetTime(TNow)
			TmpPiece = (DeltaPiece*complex(0.0,-1.0) - Eta)*TNow
			for s in range(smax): 
				TmpPiece += self.BosInf.Css[s]*DynPiece[s,:]
			f0 = numpy.exp(TmpPiece)

			TNow = ts[1]
			self.BosInf.SetTime(TNow)
			TmpPiece = (DeltaPiece*complex(0.0,-1.0) - Eta)*TNow
			for s in range(smax): 
				TmpPiece += self.BosInf.Css[s]*DynPiece[s,:]
			f1 = numpy.exp(TmpPiece)

			TNow = ts[2]
			self.BosInf.SetTime(TNow)
			TmpPiece = (DeltaPiece*complex(0.0,-1.0) - Eta)*TNow
			for s in range(smax): 
				TmpPiece += self.BosInf.Css[s]*DynPiece[s,:]
			f2 = numpy.exp(TmpPiece)			
			
			CI += (f0*W3[0]+f1*W3[1]+f2*W3[2])*((b-a)/2.0)
			
			if (T+Step >= TMax ): 
				print " Returning: ", numpy.average(CI)/CI.size 
		if (Params.ContBath): 
			# This is the integral from the MaximumTime to T is Infinity. 
			print "Truncating Bath part at: ", self.BosInf.Css[s]
			print "Performing the remainder of the integral to infinity"
			CI += complex(0.0,-1.0)*numpy.exp(complex(0.0,-1.0)*DeltaPiece*TMax)/DeltaPiece
		CI *= Equil
		return

	def MakeBCF(self, ABosonInformation): 
		# For now just evaluate the integral using the trapezoidal rule. 
		# Meaning that we store two tensors. Old-Non-Integrated, and new integrated. 
		# values of e^{i\Deltas}*B
		if (not Params.Undressed):
			self.BosInf = ABosonInformation
			print ABosonInformation
			self.BosInf.SetTime(0.0)
			UniqueDims = [ Params.nocc if X==0 else Params.nvirt for X in map(self.HPTypeDummy,self.UniqueBCIndices)]
			self.OldTime = 0.0
			self.CurrentIntegral = numpy.zeros(shape=tuple(UniqueDims),dtype = complex)
			self.IatT0 = numpy.zeros(shape=tuple(UniqueDims),dtype = complex)
			self.IatT1 = numpy.zeros(shape=tuple(UniqueDims),dtype = complex)
			self.IatT2 = numpy.zeros(shape=tuple(UniqueDims),dtype = complex)	
		else:
			self.BosInf = ABosonInformation
			self.BosInf.SetTime(0.0)		
		return 

#		
#	Other is another operator string. 
#	This does the index mapping overhead of contraction. 
#	The numerical work is done in "EvaluateMyself"
#
	def Contract(self,Other,NewState,OldState,TimeInformation = None, MultiplyBy = 1.0, FieldInformation = None):
		selft = self.MyDataFrom(NewState)
		Multiplier = MultiplyBy
		if (FieldInformation != None): 
			if ( "mux" in [X.RootName() for X in Other.Tensors]): 
				Multiplier *= FieldInformation[0]				
			elif ( "muy" in [X.RootName() for X in Other.Tensors]): 
				Multiplier *= FieldInformation[1]							
			elif ( "muz" in [X.RootName() for X in Other.Tensors]): 
				Multiplier *= FieldInformation[2]
		if (cmath.isnan(cmath.sqrt(numpy.sum(selft*selft)))): 
			print "Pre Contraction Is nan... "
			raise Exception("Nan.")				
		if (TimeInformation != None): 
			if (Params.Undressed): 
				#print "undressed. "			
				for T in Other.Tensors:
					if (T.name == "v"):
						selft += Multiplier*Other.EvaluateMyselfAdiabaticallyNew(OldState,TimeInformation)
						#print "Adiabatically", cmath.sqrt(numpy.sum(selft*selft))
						return
					elif (T.name == "B"):
						selft += Multiplier*Other.EvaluateMyselfUndressed(OldState,TimeInformation)
						#print "Undressed", cmath.sqrt(numpy.sum(selft*selft))
						return
			if ( Params.Inhomogeneous): 
				for T in Other.Tensors: 
					if (T.Time == "i"):
						selft += Multiplier*Other.EvaluateMyselfInhomogeneous(OldState,TimeInformation)
						return 			
			if (Params.Adiabatic == 3): 
				selft += Multiplier*Other.EvaluateMyselfAdiabaticallyAntiHerm(OldState,TimeInformation)
			elif (Params.Adiabatic == 2):
				selft += Multiplier*Other.EvaluateMyselfAdiabaticallyNew(OldState,TimeInformation)
			# The below even works for the Markov case. 
			elif (Params.Adiabatic == 1 or Params.Adiabatic == 0 or Params.Adiabatic == 4): 
				selft += Multiplier*Other.EvaluateMyselfTrivially(OldState,TimeInformation)
			else : 
				print "Unknown adiabatic setting... "
			if (cmath.isnan(cmath.sqrt(numpy.sum(selft*selft)))): 
				print "Post contraction Is nan... "
				self.Print()
				Other.Print()
				raise Exception("Nan.")	
			return 			
		# Here we obtain the positions of the vac dummy indices in the simple 
		# contraction result. 
		VacIndices = []
		for T in Other.Tensors: 
			if (T.IsVac()): 
				VacIndices = copy.copy(T.indices)
				if (len(VacIndices) != len(self.Tensors[0].indices)):
					print "Error... Should already check that there is same num of vac indices!"
					raise Exception("Fuuuuu.... ")
				else : 
					break 
		if (len(VacIndices) == 0): 
			print "Error... No Vacuum indices. "
			self.Print()
			Other.Print()
		# Make sure that the vac indices are my indices. 
		ToObtain = self.Tensors[0].indices # I should be the vacuum.
		for I in ToObtain:
			if (not (I in ToObtain)): 
				print "Vac and Me don't match :( ", ToObtain, VacIndices
				raise Exception("ruhroh")
		#
		dim = range(len(ToObtain))		
		VacIndicesComeFrom = range(len(dim))
		DeltaMap = Other.DeltasAsDict()
		# These are the dummy indices that the vaccum is delta-d to. 
		try: 
			for I in dim:
				VacIndicesComeFrom[I] = DeltaMap[VacIndices[I]]		
		except: 
			print "Failed to assign vaccuum."
			self.Print()
			Other.Print()
			print "VacIndices: ", VacIndices
		# Now establish the position of each of these dummy indices in the result
		# of the contraction (we have to remove all the other summed over indices) 
		RealSums = []
		UnOrderedResultIndices = []
		for D in Other.Deltas: 
			if (D[0] in VacIndices): 
				UnOrderedResultIndices.append(D[1])
			elif (D[1] in VacIndices): 
				UnOrderedResultIndices.append(D[0])			
			else : 
				RealSums.append(D) 		
		OrderedResultIndices = []
		for I in Other.AllIndices(): 
			if (I in UnOrderedResultIndices): 
				OrderedResultIndices.append(I)
		#Finally the positions the vacuum indices come from. 
		Map = range(len(ToObtain))
		for I in dim: 
			Map[I] = OrderedResultIndices.index(VacIndicesComeFrom[I])
		try:
			# Note that evaluate myself includes the prefactor for this term. 
			selft += complex(Multiplier)*numpy.transpose(Other.EvaluateMyself(OldState),axes=Map)
		except Exception as Ex:
			print "Transpose or evaluate failure. Axes: ", Map
			print "Strings: "
			print "Self: "
			self.Print()
			print "Other: "
			Other.Print()
			print Other.EvaluateMyself(OldState)
			raise Ex
		if (cmath.isnan(cmath.sqrt(numpy.sum(selft*selft)))): 
			print "Post contraction Is nan... "
			self.Print()
			raise Exception("Nan.")
		return 

	def CorrelationInt(x,t,n,beta,xc,delta,alpha):
		sum = 0.0
		pre1 = pow((1-x)/x, n)*pow((1-xc)/xc,1-n)*alpha*(np.cosh(((1-x)*xc)/(x*(1-xc))) - np.sinh(((1-x)*xc)/(x*(1-xc))))/(((pow(1-x,2)/pow(x,2)) - pow(delta,2))*np.sinh((beta*(1-x))/(2*x)))
		pre2 = -i1*delta*np.cos((t*(1-x))/x-(t0*(1-x))/x-(1j*(1-x)*beta)/(2*x)) + ((1-x)/x)*np.sin((t*(1-x))/x-(t0*(1-x))/x-(1j*(1-x)*beta)/(2*x))
		sum1 = np.cos(t0 * delta) + 1j*np.sin(t0*delta)
		sum2 = 1j*(np.cos(t*delta) + 1j*np.sin(t*delta))*(delta*np.cosh(((1-x)*beta)/(2*x)) + ((1-x)/x)*np.sinh( ((1-x)*beta)/(2*x)) )
		return pre1*(pre2*sum1 + sum2)

	def CorrelationIntDiag(x,t,n,beta,xc,alpha): #This was originally added to the project in the CH4 directory. Oops.
		return np.exp((-(((1-x)*xc)/(x*1-xc))))*pow(((1-x)/x),n-1)*pow(((1-xc)/xc),1-n)*alpha*(1j*1/np.sinh(((1 - x)*beta)/(2*x)) * np.sin(  (t*(1 - x))/x - (t0*(1 - x))/x - (1j*(1 - x)*beta)/(2*x) ) )

	def EvaluateMyself(self,OldState):
		if (len(self.Tensors) <= 2): 
			print "error, I'm simple."
			return
		if (len(self.Deltas) == 0): 
			print "Error, No summation. "
			raise Exception("NoSummation") 
		else: 
			TempTensors = copy.copy(self.Tensors) 
			# locate the vaccum and pop it. 
			a = None
			for T in rlen(TempTensors): 
				if TempTensors[T].IsVac():					 
					a = TempTensors.pop(T)
					break 
			if (not a.IsVac()): 
				print "Error, First tensor should be vacuum."
				raise Exception
			a = TempTensors.pop(0)	
			adata = a.MyDataFrom(OldState)
			ainds = copy.copy(a.indices)
			stuffina = [a.name]
			while (len(TempTensors) > 0): 
				b = TempTensors.pop(0)
				bdata = b.MyDataFrom(OldState)
				binds = copy.copy(b.indices)
				(SummedIn1,SummedIn2) = self.SummedBetween(ainds,binds)
				# tensordot needs the positions of those indices. 
				c1 = tuple(map(lambda X:ainds.index(X),SummedIn1))
				c2 = tuple(map(lambda X:binds.index(X),SummedIn2))
				# Now make the new list of indices. 
				t1 = ListComplement(ainds,SummedIn1)
				t2 = ListComplement(binds,SummedIn2)
				t1.extend(t2)
				ainds = []
				ainds = copy.copy(t1)
				# perform the numerical work. 
				try: 
					cc = numpy.tensordot(adata,bdata,axes=(c1,c2))
			#		print "Adata, ", adata.mean(), ", Bdata, ", bdata.mean(), " Cdata ", cc.mean()
					adata = cc
					stuffina.append(b.name)				
				except Exception as Inst: 
					print "tensorsdot failed for term: "
					self.Print()
					print "Tensors: ",stuffina,b.name 
					print adata.shape
					print bdata.shape				
					print SummedIn1,SummedIn2,c1,c2
					raise Inst 
				# replace a with c 
			adata *= self.PreFactor
			#print " Result With PreFactor ", adata.mean()			
			return adata	

	def EvaluateMyselfAdiabaticallyNew(self,OldState,Tt):
		if (len(self.Tensors) <= 2): 
			print "error, I'm simple."
			return
		else: 
			TempTensors = copy.copy(self.Tensors) 
			a = None
			# Generate Transformed integral once-and for all. 
			SIntegral = None				
			TimeFactor = None
			for T in rlen(TempTensors):
				if (TempTensors[T].name == "v" and TempTensors[T].Time == "s"):
					SIntegral = TempTensors[T].MyDataFrom(OldState)
					TimeFactor = numpy.zeros(SIntegral.shape, dtype=complex)
			EDSTensor = self.EDSTensor
			ShapeOfSummed = TimeFactor.shape
			TR = int(self.TimeReversed)
			code =	"""
					#line 1633 "Wick.py"
					using namespace std;
					int SOS[4];				
					for (int k = 0 ; k<4; ++k)
						SOS[k] = (int)ShapeOfSummed[k];
					double es; 
					complex<double> imi(0.0,1.0);
					int It[4];	
					for (It[0] = 0 ; It[0] < SOS[0] ; It[0]=It[0]+1)
					{
						for (It[1] = 0 ; It[1] < SOS[1] ; It[1]=It[1]+1) 
						{
							for (It[2] = 0 ; It[2] < SOS[2] ; It[2]=It[2]+1)
							{
								for (It[3] = 0 ; It[3] < SOS[3] ; It[3]=It[3]+1)
								{					
									es = 0.0; 
									for (int j = 0 ; j < 4; ++j) 
										es += EDSTensor(j,It[j]); 
									if (TR)
										TimeFactor(It[0],It[1],It[2],It[3]) += conj((1.0-( cos(Tt*es) - imi*sin(Tt*es)))/(imi*es));
									else 
										TimeFactor(It[0],It[1],It[2],It[3]) += (1.0-( cos(Tt*es) - imi*sin(Tt*es)))/(imi*es);						
								}							
							}			 		
						}
					}
					"""
			weave.inline(code,
						['ShapeOfSummed', 'TimeFactor','TR','EDSTensor','Tt'],
						global_dict = dict(), 
						headers = ["<complex>","<iostream>"],
						type_converters = scipy.weave.converters.blitz, 
						compiler = Params.Compiler, extra_compile_args = Params.Flags,
						verbose = 1)	
			if False: 
				ShapeOfSummed = TimeFactor.shape
				for It[0] in range(ShapeOfSummed[0]):
					for It[1] in range(ShapeOfSummed[1]):
						for It[2] in range(ShapeOfSummed[2]):
							for It[3] in range(ShapeOfSummed[3]):
								es = 0.0
								for i in xrange(4):
									es += EDSTensor[i, It[i]]
								if es == 0:
									print "Denominator is 0"
								# be sure to calculate the new correlation function
								if(self.TimeReversed):
									TimeFactor[It[0]][It[1]][It[2]][It[3]] = np.conj( (1.0-(np.cos(Tt*es) - 1j*np.sin(Tt*es)))/(1j*es) )
								else:
									TimeFactor[It[0]][It[1]][It[2]][It[3]] = ((1.0-(np.cos(Tt*es) - 1j*np.sin(Tt*es)))/(1j*es))
								
								
			a = None
			for T in rlen(TempTensors): 
				if TempTensors[T].IsVac():					 
					a = TempTensors.pop(T)
					break 
			if (not a.IsVac()): 
				print "Error, First tensor should be vacuum."
				raise Exception
			a = TempTensors.pop(0)	
			adata=None
			if (a.name == "v" and a.Time == "s"):
				adata = a.MyDataFrom(OldState)*TimeFactor			
			else: 
				adata = a.MyDataFrom(OldState)
			ainds = copy.copy(a.indices)
			stuffina = [a.name]
			weight = np.array([0, Params.nvirt, Params.nocc, Params.nvirt, Params.nocc, Params.nvirt + Params.nocc, Params.nvirt + Params.nocc])
			#print weight
			while (len(TempTensors) > 0): 
				#logic to pick the tensor contraction that has the smallest memory footprint
				aselfCont = np.array(self.occNumber(a))
			#	print aselfCont
				maxDiff = 0
				maxInd = 0
				for T in rlen(TempTensors):
			#		print T
					bselfCont = np.array(self.occNumber(TempTensors[T]))
			#		print bselfCont
					sharedInd = np.array(self.ConnectionsBetweenWithInd(a,TempTensors[T]))
			#		print sharedInd
					tmpDiff = numpy.dot(aselfCont + bselfCont - sharedInd, weight)
					if(tmpDiff > maxDiff):
						maxInd = T
						maxDiff = tmpDiff
				b = TempTensors.pop(maxInd)
				bdata = None
				if (b.name == "v" and b.Time == "s"):
					bdata = b.MyDataFrom(OldState)*TimeFactor 
				else :
					bdata = b.MyDataFrom(OldState)
				binds = copy.copy(b.indices)
			#	print ainds
			#	print binds
			#	print sys.getsizeof(adata)
				#New Logic for repeated indices.
				c1 = [] #just changed this so that c1 has the append method
				c2 = [] 
				t1 = [] 
				t2 = []
				# tensordot needs the positions of those indices.
				for ind1 in range(len(ainds)): 
					if ainds[ind1] in binds: 
						c1.append(ind1)
						c2.append(binds.index(ainds[ind1]))
					else: 
						t1.append(ainds[ind1]) # Because these indices will remain and be contracted later. 
				for ind2 in range(len(binds)): 
					if not binds[ind2] in ainds: 
						t2.append(binds[ind2])						
				t1.extend(t2)
				ainds = []
				ainds = copy.copy(t1)
				# perform the numerical work. 
				try: 
					adata = numpy.tensordot(adata,bdata,axes=(c1,c2))
					stuffina.append(b.name)				
				except Exception as Inst: 
					print "tensorsdot failed for term: "
					self.Print()
					print "Tensors: ",stuffina,b.name 
					print adata.shape
					print bdata.shape				
					print SummedIn1,SummedIn2,c1,c2
					raise Inst 
				# replace a with c 
			adata *= self.PreFactor
			
			# Figure out if the result needs to be transposed. 
			VacIndices = None
			for T in self.Tensors: 
				if (T.IsVac()):  
					VacIndices = T.indices
			
			if (ainds[0] == VacIndices[0] and ainds[1] == VacIndices[1]): 
				return adata
			elif (ainds[0] == VacIndices[1] and ainds[1] == VacIndices[0]): 
				return adata.transpose()
			else: 
				print "Bad Indexing. "
				raise Exception("Bad Indexing.")
			exit()
			return 


#	Reorder summation loop to do the bs bosons first?
#
	global integrandCont
	#np.cos(es*t) = coet
	#np.sin(es*t) = siet
	def integrandCont(w, wc, t, es, beta, alpha, coet, siet):
		return (alpha*w*(-1j*w*coet + 1j*w*np.cos(t*w) - 1j*es*coet*np.cosh((beta*w)/2.)/np.sinh((beta*w)/2.) + 1j*es*np.cos(t*w)*np.cosh((beta*w)/2.)/np.sinh((beta*w)/2.) + w*siet + es*np.cosh((beta*w)/2.)/np.sinh((beta*w)/2.)*siet - es*np.sin(t*w) - w*np.cosh((beta*w)/2.)/np.sinh((beta*w)/2.)*np.sin(t*w))*(np.cos(es*t - (1j*w)/wc) - 1j*np.sin(es*t - (1j*w)/wc)))/((es - w)*(es + w)*wc*wc)

	def EvaluateMyselfUndressed(self, OldState, t): 
		SummedIndices = list(set(self.AllIndices())) 
		TypesOfSummed = map(self.HPTypeDummy,SummedIndices)
		ShapeOfSummed = range(4)
		for i in rlen(ShapeOfSummed): 
			if i in SummedIndices: 
				if TypesOfSummed[SummedIndices.index(i)] == 0 : #If this comes out to be zero, it gives it hole dimension otherwise, it gives it the particle dimension
					ShapeOfSummed[i] = Params.nocc
				else :
					ShapeOfSummed[i] = Params.nvirt				
			else :
				ShapeOfSummed[i] = 1
		Deltas = []
		Deltat = []
		Btn = None
		Vsn = None
		BtIndices = None
		BsIndices = None
		RhoIndices = None
		VacIndices = None
		r1 = None
		VacShape = None
		for T in self.Tensors:
			if (T.Time == "t"): 
				BtIndices = T.indices
				for O in T.ops:
					if abs(O[0]) == 1:
						Deltat.append([1,O[1]]) # meaning + epsilon_O[1]
					elif abs(O[0]) == 2: 
						Deltat.append([-1,O[1]])
					elif abs(O[0]) == 3:
						Deltat.append([-1,O[1]])
					elif abs(O[0]) == 4: 
						Deltat.append([1,O[1]])
					else : 
						print "Can't assign BCF, unknown type."
			elif (T.Time == "s"): 
				for O in T.ops: 
					if abs(O[0]) == 1:
						Deltas.append([1,O[1]])
					elif abs(O[0]) == 2: 
						Deltas.append([-1,O[1]])
					elif abs(O[0]) == 3:
						Deltas.append([-1,O[1]])			 
					elif abs(O[0]) == 4: 
						Deltas.append([1,O[1]])
					else : 
						print "Can't assign BCF, unknown type."
		for T in self.Tensors: 
			if (T.IsVac()):  
				VacIndices = T.indices
				VacShape = OldState[T.NonVacName()].shape
			elif T.Time == "t" : # This is V at time t. 
				Btn=T.Name()
				BtIndices=T.indices
			elif T.Time == "s":  
				Bsn=T.Name()
				BsIndices=T.indices
			elif (T.name == "r1"):
				r1=OldState[T.NonVacName()]				
				RhoIndices = T.indices
			elif T.Time == "i": 
				raise Exception("Error, Contained i") 	
		if (RhoIndices == None or VacShape == None or r1 == None): 
			raise Exception("Error, didn't find rho") 	
		Ext = numpy.zeros(shape=VacShape, dtype = complex)
		self.EDeltat = copy.deepcopy(Deltat)
		self.EDeltas = copy.deepcopy(Deltas)
		EDTIndices = [X[1] for X in self.EDeltat]
		EDTTypes = map(self.HPTypeDummy,EDTIndices)
		EDSIndices = [X[1] for X in self.EDeltas]
		EDSTypes = map(self.HPTypeDummy,EDSIndices)	
		EDTTensor = self.EDTTensor
		EDSTensor = self.EDSTensor
		
		Bt = self.BosInf.GeneralCouplings[Btn]
		Bs = self.BosInf.GeneralCouplings[Bsn] 
		
		if (numpy.sum(Bt*Bt) == 0.0 or numpy.sum(Bs*Bs) == 0.0): 
			return Ext

		Coths = self.BosInf.Coths
		It = np.zeros(4)
		beta = self.BosInf.beta
		TR = int(self.TimeReversed)
		n = self.BosInf.ohmicParam
		wc = self.BosInf.OmegaCThree
		alpha = self.BosInf.a
	
		ints = numpy.zeros(shape = Bs.shape,dtype = complex)
		ContBath = int(Params.ContBath)
		if(Params.ContBath):
			for i in xrange(ints.shape[0]):
				for j in xrange(ints.shape[1]):
					# CHEAT HACK CHEAT HACK
					# if Bs is zero at this index we don't need to integrate. 					
					if (Bs[i][j][0] == 0.0):
						continue 
					es = EDSTensor[0,i] + EDSTensor[1,j]
					coet=np.cos(es*t)
					siet=np.sin(es*t)					
					#ints[i,j] = quadrature(integrandCont, 0, 15*wc, args=(wc,t,es,beta,alpha,coet,siet), tol=1e-8)[0]
					ints[i,j] = fixed_quad(integrandCont, 0, 15*wc, args=(wc,t,es,beta,alpha,coet,siet), n=45)[0]
		KMax = self.BosInf.NBos
		w = self.BosInf.Freqs
		code =	"""  
				#line 1944 "Wick.py"  
				using namespace std;
				int bti[2]; 
				int bsi[2]; 
				int eti[2];
				int esi[2];
				int SOS[4];				
				int ri[2];
				int ei[2];
				for (int k = 0 ; k<2; ++k)
				{
					bti[k] = (int)BtIndices[k];
					bsi[k] = (int)BsIndices[k]; 
					esi[k] = (int)EDSIndices[k];   
					eti[k] = (int)EDTIndices[k];					
				}
				for (int k = 0 ; k<4; ++k)
					SOS[k] = (int)ShapeOfSummed[k];
				for (int k = 0 ; k<2; ++k)
				{
					ri[k] = (int)RhoIndices[k];
					ei[k] = (int)VacIndices[k];	
				}
				complex<double> r,ext, tmp1, tmp2; 
				double bs,bt;
				double es, wk; 
				complex<double> imi(0.0,1.0);  
				int It[4];
				// these loops can improve efficiency significantly if Bt is sparse. 
				if (bti[0] == 2 && bti[1] == 0)
				{
					for(int k = 0; k < KMax; k++)
					{
						wk = w(k);
						for (It[0] = 0 ; It[0] < SOS[0] ; It[0]=It[0]+1)
						{
							for (It[2] = 0 ; It[2] < SOS[2] ; It[2]=It[2]+1)
							{
								bt = Bt(It[2], It[0],k);
								if(bt == 0.0)
									continue; 						
								for (It[1] = 0 ; It[1] < SOS[1] ; It[1]=It[1]+1)
								{
									for (It[3] = 0 ; It[3] < SOS[3] ; It[3]=It[3]+1)
									{
										bs = Bs(It[bsi[0]], It[bsi[1]],k); 
										if(bs == 0.0)
											continue; 									
										r = r1(It[ri[0]], It[ri[1]]);
										es = EDSTensor(0, It[esi[0]]) + EDSTensor(1, It[esi[1]]);
										if(es == 0.0)
										{  
											tmp1 = -imi/wk + (imi*cos(t*wk))/wk + (1./tanh((beta*wk)/2.)*sin(t*wk))/wk;
											Ext(It[ei[0]], It[ei[1]]) += -1.0*(bt*bs*r*tmp1);
											continue;
										} 
										tmp1 = ((cos(es*t) - imi*sin(es*t))*(-imi*wk*cos(es*t) + imi*wk*cos(t*wk) - imi*es*cos(es*t)*Coths(k) + imi*es*cos(t*wk)*Coths(k) + wk*sin(es*t) + es*Coths(k)*sin(es*t) - es*sin(t*wk) - wk*Coths(k)*sin(t*wk)))/((es - wk)*(es + wk));
										Ext(It[ei[0]], It[ei[1]]) += -1.0*(bt*bs*r*tmp1);
										if ((int)ContBath && k==0)
											Ext(It[ei[0]], It[ei[1]]) += -1.0*(bt*bs*r*ints(It[esi[0]], It[esi[1]]));
									}																																	
								}							   	
							}					
						}
					}          																								
				}
				else if (bti[0] == 2 && bti[1] == 3)
				{
					for(int k = 0; k < KMax; k++)
					{
						wk = w(k);
						for (It[2] = 0 ; It[2] < SOS[2] ; It[2]=It[2]+1)
						{
							for (It[3] = 0 ; It[3] < SOS[3] ; It[3]=It[3]+1)
							{
								bt = Bt(It[2], It[3],k);
								if(bt == 0.0)
									continue; 						
								for (It[1] = 0 ; It[1] < SOS[1] ; It[1]=It[1]+1)
								{
									for (It[0] = 0 ; It[0] < SOS[0] ; It[0]=It[0]+1)
									{
										bs = Bs(It[bsi[0]], It[bsi[1]],k);
										if(bs == 0.0)
											continue; 
										r = r1(It[ri[0]], It[ri[1]]);
										es = EDSTensor(0, It[esi[0]]) + EDSTensor(1, It[esi[1]]);
										if(es == 0.0)
										{  
											tmp1 = -imi/wk + (imi*cos(t*wk))/wk + (1./tanh((beta*wk)/2.)*sin(t*wk))/wk;
											Ext(It[ei[0]], It[ei[1]]) += -1.0*(bt*bs*r*tmp1);
											continue;
										} 
										tmp1 = ((cos(es*t) - imi*sin(es*t))*(-imi*wk*cos(es*t) + imi*wk*cos(t*wk) - imi*es*cos(es*t)*Coths(k) + imi*es*cos(t*wk)*Coths(k) + wk*sin(es*t) + es*Coths(k)*sin(es*t) - es*sin(t*wk) - wk*Coths(k)*sin(t*wk)))/((es - wk)*(es + wk));
										Ext(It[ei[0]], It[ei[1]]) += -1.0*(bt*bs*r*tmp1);
										if ((int)ContBath && k==0)
											Ext(It[ei[0]], It[ei[1]]) += -1.0*(bt*bs*r*ints(It[esi[0]], It[esi[1]]));										
									}																																	
								}							   	
							}					
						}
					}          																								
				}																		
				else if (bti[0] == 1 && bti[1] == 2)
				{
					for(int k = 0; k < KMax; k++)
					{
						wk = w(k);
						for (It[1] = 0 ; It[1] < SOS[1] ; It[1]=It[1]+1)
						{
							for (It[2] = 0 ; It[2] < SOS[2] ; It[2]=It[2]+1)
							{
								bt = Bt(It[1], It[2],k);
								if(bt == 0.0)
									continue; 						
								for (It[0] = 0 ; It[0] < SOS[0] ; It[0]=It[0]+1)
								{
									for (It[3] = 0 ; It[3] < SOS[3] ; It[3]=It[3]+1)
									{
										bs = Bs(It[bsi[0]], It[bsi[1]],k);
										if(bs == 0.0)
											continue; 									
										r = r1(It[ri[0]], It[ri[1]]);
										es = EDSTensor(0, It[esi[0]]) + EDSTensor(1, It[esi[1]]);
										if(es == 0.0)
										{  
											tmp1 = -imi/wk + (imi*cos(t*wk))/wk + (1./tanh((beta*wk)/2.)*sin(t*wk))/wk;
											Ext(It[ei[0]], It[ei[1]]) += -1.0*(bt*bs*r*tmp1);
											continue;
										} 
										tmp1 = ((cos(es*t) - imi*sin(es*t))*(-imi*wk*cos(es*t) + imi*wk*cos(t*wk) - imi*es*cos(es*t)*Coths(k) + imi*es*cos(t*wk)*Coths(k) + wk*sin(es*t) + es*Coths(k)*sin(es*t) - es*sin(t*wk) - wk*Coths(k)*sin(t*wk)))/((es - wk)*(es + wk));
										Ext(It[ei[0]], It[ei[1]]) += -1.0*(bt*bs*r*tmp1);
										if ((int)ContBath && k==0)
											Ext(It[ei[0]], It[ei[1]]) += -1.0*(bt*bs*r*ints(It[esi[0]], It[esi[1]]));										
									}																																	
								}							   	
							}					
						}
					}          																								
				}
				else
				{																																										
					for (It[0] = 0 ; It[0] < SOS[0] ; It[0]=It[0]+1)
					{
						for (It[1] = 0 ; It[1] < SOS[1] ; It[1]=It[1]+1)
						{
							for (It[2] = 0 ; It[2] < SOS[2] ; It[2]=It[2]+1)
							{
								for (It[3] = 0 ; It[3] < SOS[3] ; It[3]=It[3]+1)
								{
									r = r1(It[ri[0]], It[ri[1]]);
									es = EDSTensor(0, It[esi[0]]) + EDSTensor(1, It[esi[1]]);
									for(int k = 0; k < KMax; k++)
									{
										wk = w(k);
										bt = Bt(It[bti[0]], It[bti[1]],k);
										bs = Bs(It[bsi[0]], It[bsi[1]],k);
										if(bt == 0.0 || bs == 0.0)
											continue; 
										if(es == 0.0)
										{  
											tmp1 = -imi/wk + (imi*cos(t*wk))/wk + (1./tanh((beta*wk)/2.)*sin(t*wk))/wk;
											Ext(It[ei[0]], It[ei[1]]) += -1.0*(bt*bs*r*tmp1);
											continue;
										} 
										tmp1 = ((cos(es*t) - imi*sin(es*t))*(-imi*wk*cos(es*t) + imi*wk*cos(t*wk) - imi*es*cos(es*t)*Coths(k) + imi*es*cos(t*wk)*Coths(k) + wk*sin(es*t) + es*Coths(k)*sin(es*t) - es*sin(t*wk) - wk*Coths(k)*sin(t*wk)))/((es - wk)*(es + wk));
										Ext(It[ei[0]], It[ei[1]]) += -1.0*(bt*bs*r*tmp1);
										if ((int)ContBath && k==0)
											Ext(It[ei[0]], It[ei[1]]) += -1.0*(bt*bs*r*ints(It[esi[0]], It[esi[1]]));										
									}																																	
								}							   	
							}					
						}
					}  
				}               
				"""
		weave.inline(code,
					['Ext','r1','Bt','Bs','BtIndices','EDTIndices','EDSIndices', 'BsIndices',
					'RhoIndices','ShapeOfSummed', 'VacIndices','EDTTensor','EDSTensor',
					'KMax','w','Coths',
					't', 'beta', 'ints','ContBath'],
					 global_dict = dict(), 
					 headers = ["<complex>","<iostream>"], 
					 type_converters = scipy.weave.converters.blitz, extra_compile_args = Params.Flags,
					 verbose = 0)
		Ext *= self.PreFactor
		if (cmath.isnan(cmath.sqrt(numpy.sum(Ext*Ext)))): 
			print "Post contraction Is nan... "
			self.Print()
			raise Exception("Nan.")
		#print Ext added to check that Ext wasn't empty when hasbosons is false. This works. We're good
		return Ext	


	def EvaluateMyselfInhomogeneous(self,OldState,Tt): #Note that all the subscript s stuff is actually subscript i
		SummedIndices = list(set(self.AllIndices())) # These are then sorted. 
		TypesOfSummed = map(self.HPTypeDummy,SummedIndices)		
		ShapeOfSummed = range(6)
		for i in rlen(ShapeOfSummed): 
			if i in SummedIndices: 
				if TypesOfSummed[SummedIndices.index(i)] == 0 : 
					ShapeOfSummed[i] = Params.nocc
				else :
					ShapeOfSummed[i] = Params.nvirt				
			else :
				ShapeOfSummed[i] = 1
		Vtn = None 
		Vsn = None 
		VtIndices = None
		VsIndices = None 
		RhoIndices = None
		EDTIndices = [X[1] for X in self.EDeltat]
		EDSIndices = [X[1] for X in self.EDeltai]		
		RhoIndices = None
		VacIndices = None 
		r1 = None 
		VacShape = None	
		for T in self.Tensors: 
			if (T.IsVac()):  
				VacIndices = T.indices
				VacShape = OldState[T.NonVacName()].shape
			elif T.Time == "t" : # This is V at time t. 
				Vtn=T.Name()
				VtIndices=T.indices
			elif T.Time == "i":  
				Vsn=T.Name()
				VsIndices=T.indices
			elif (T.name == "r1"):
				r1=OldState[T.NonVacName()]				
				RhoIndices = T.indices
			elif T.Time == "s": 
				raise Exception("Error, Contained s") 	
		if (RhoIndices == None or VacShape == None or r1 == None): 
			raise Exception("Error, didn't find rho") 	
		Ext = numpy.zeros(shape=VacShape, dtype = complex)
		Vt = Integrals[Vtn]
		Vs = Integrals[Vsn]
		EDTTensor = self.EDTTensor
		EDSTensor = self.EDITensor #Defines EDSTensor as EDITensor to simplify things quite a bit and allow us to use mostly old code
		DeltaTLen = len(self.EDeltat)
		TR = int(self.TimeReversed)
		code =	"""
				#line 1995 "Wick.py"
				using namespace std;
				int vti[4]; 
				int vsi[4]; 
				int eti[6];
				int esi[4];
				int SOS[6];				
				int ri[2];
				int ei[2];				
				for (int k = 0 ; k<4; ++k)
				{
					vti[k] = (int)VtIndices[k];
					vsi[k] = (int)VsIndices[k];
					esi[k] = (int)EDSIndices[k];
				}
				for (int k = 0 ; k<(int)DeltaTLen; ++k)
					eti[k] = (int)EDTIndices[k];					
				for (int k = 0 ; k<6; ++k)
					SOS[k] = (int)ShapeOfSummed[k];
				for (int k = 0 ; k<2; ++k)
				{
					ri[k] = (int)RhoIndices[k];
					ei[k] = (int)VacIndices[k];	
				}
				complex<double> r,ext;
				double vs,vt;
				double es, et; 
				complex<double> imi(0.0,1.0);
				int It[6];																
				for (It[0] = 0 ; It[0] < SOS[0] ; It[0]=It[0]+1)
				{
					for (It[1] = 0 ; It[1] < SOS[1] ; It[1]=It[1]+1)
					{
						for (It[2] = 0 ; It[2] < SOS[2] ; It[2]=It[2]+1)
						{
							for (It[3] = 0 ; It[3] < SOS[3] ; It[3]=It[3]+1)
							{
								for (It[4] = 0 ; It[4] < SOS[4] ; It[4]=It[4]+1)
								{
									for (It[5] = 0 ; It[5] < SOS[5] ; It[5]=It[5]+1)
									{																					
										r = r1(It[ri[0]],It[ri[1]]);										
										if ( r == 0.0)
											continue; 										
										vt = Vt(It[vti[0]],It[vti[1]],It[vti[2]],It[vti[3]]);
										if ( vt == 0.0 )
											continue; 
										vs = Vs(It[vsi[0]],It[vsi[1]],It[vsi[2]],It[vsi[3]]);
										if ( vs == 0.0 )
											continue; 
										et = 0.0;
										es = 0.0;
										for (int j = 0 ; j < (int)DeltaTLen; ++j) 
											et += EDTTensor(j,It[eti[j]]);	
										for(int j = 0; j < 4; j++)
											es += EDSTensor(j,It[esi[j]]); 																				
										if (es == 0.0)
										{
											cout << "Error. Denominator is zero." << endl; 
										}
										if (TR)
											Ext(It[ei[0]],It[ei[1]]) += vt*vs*r*conj( ( -sin(Tt*et) - imi*cos(Tt*et))/(es-imi)) ; //Should have an e^i \Delta t
										else 
											Ext(It[ei[0]],It[ei[1]]) += vt*vs*r*( -sin(Tt*et) - imi*cos(Tt*et) )/(es-imi); //should just be the unconjugated partner to the above

//											Ext(It[ei[0]],It[ei[1]]) += vt*vs*r*conj( ( -sin(Tt*et) + imi*cos(Tt*et))/(es*imi)) ; //Should have an e^i \Delta t
//										else 
//											Ext(It[ei[0]],It[ei[1]]) += vt*vs*r*( -sin(Tt*et) + imi*cos(Tt*et) )/(es*imi); //should just be the unconjugated partner to the above										
									}								
								}															
							}								
						}					
					} 
				}
				"""
		weave.inline(code,
					['Ext','r1','Vt','Vs','VtIndices','EDTIndices','EDSIndices', 
					'VsIndices','RhoIndices','ShapeOfSummed', 'VacIndices','TR',
					'EDTTensor','EDSTensor','Tt', 'DeltaTLen'],
					global_dict = dict(), 
					headers = ["<complex>","<iostream>"],
					type_converters = scipy.weave.converters.blitz, 
					compiler = Params.Compiler,
					verbose = 1)	
		Ext *= self.PreFactor
		if (cmath.isnan(cmath.sqrt(numpy.sum(Ext*Ext)))): 
			print "Post contraction Is nan... "
			self.Print()
			raise Exception("Nan.")
		return Ext	

	# this is the version which is actually called if adiabatic == 0 
	# Uses third order Gaussian Quadrature. 
	def UpdateTermBCF(self,T):
		if (T == self.OldTime): 
			return
		T3 = numpy.array([-sqrt(3.0/5.0),0.0,sqrt(3.0/5.0)])
		W3 = numpy.array([5.0/9.0,8.0/9.0,5.0/9.0])		
		a = self.OldTime
		b = T
		ts = T3*((b-a)/2.0) + ((a+b)/2.0)
		t0 = float(ts[0])
		t1 = float(ts[1])
		t2 = float(ts[2])
		C0 = self.BosInf.CAt(ts[0])
		C1 = self.BosInf.CAt(ts[1])
		C2 = self.BosInf.CAt(ts[2])
		CI = self.CurrentIntegral
		f0 = self.IatT0
		f1 = self.IatT1
		f2 = self.IatT2
		f0*=0.0
		f1*=0.0
		f2*=0.0		
		EDSTensor = self.EDSTensor
		MTil = self.BosInf.MTilde
		Freqs = self.BosInf.Freqs
		Coths = self.BosInf.Coths
		smax = self.BosInf.NBos
		Unique = self.UniqueBCIndices		
		# 4-19-2012 It can happen that the unique indices2 aren't contiguous
		# Check to make sure that isn't broken here. 
		LoopDim = len(self.UniqueBCIndices)
		try : 
			RedIndicies = [self.UniqueBCIndices.index(self.BCIndices[i]) for i in range(8)]
			RedTypes = map(self.HPTypeDummy, RedIndicies)
			RedShifts = [ Params.nocc if X == 1 else 0 for X in RedTypes]
			SIndices = [ self.UniqueBCIndices.index(X[1]) for X in self.EDeltas ] 													
			TypesOfSummed = map(self.HPTypeDummy,self.UniqueBCIndices)
			ShapeOfSummed = [ Params.nocc if X==0 else Params.nvirt for X in TypesOfSummed]
		except Exception as Ex: 
			self.Print()
			raise		
		Adia = Params.Adiabatic 
		UVCut = Params.UVCutoff
		TR = int(self.TimeReversed)
		code =	"""
				#line 2132 "Wick.py"
				using namespace std;
				int ri[8]; 
				int rs[8];
				int si[4]; 
				int ci[6];
				for (int k = 0 ; k<8; ++k)
				{
					ri[k] = (int)RedIndicies[k];
					rs[k] = (int)RedShifts[k]; // Picks out Occupied or V
				}
				for (int k = 0 ; k<4; ++k)
				{
					si[k] = (int)SIndices[k];
				}
				for (int k = 0 ; k<(int)LoopDim; ++k)
				{
					ci[k] = (int)ShapeOfSummed[k]; 					
				}	
				complex<double> imi(0.0,1.0); 
				complex<double> mimi(0.0,-1.0); 				
				complex<double> BathSum0,BathSum1,BathSum2;				
				double TildeSum1 = 0.0; 
				double TildeSum2 = 0.0;
				double es;  				
				double wSum = 0.0; // Sum of equilibrium piece. 
				int It[6];
				if (LoopDim == 4)
				{
					for (It[0] = 0 ; It[0] < ci[0] ; It[0]=It[0]+1)
					{
						for (It[1] = 0 ; It[1] < ci[1] ; It[1]=It[1]+1)
						{
							for (It[2] = 0 ; It[2] < ci[2] ; It[2]=It[2]+1)
							{
								for (It[3] = 0 ; It[3] < ci[3] ; It[3]=It[3]+1)
								{
									es = EDSTensor(0,It[si[0]]) + EDSTensor(1,It[si[1]]) + EDSTensor(2,It[si[2]]) + EDSTensor(3,It[si[3]]);
									if (abs(es) > UVCut && Adia == 0)
									{
										if (TR) 
											CI(It[0],It[1],It[2],It[3]) = conj((1.0-( cos(t1*es) - imi*sin(t1*es)))/(imi*es));
										else 
											CI(It[0],It[1],It[2],It[3]) = (1.0-( cos(t1*es) - imi*sin(t1*es)))/(imi*es);
										continue; 
									}
									BathSum0 = 0.0;
									BathSum1 = 0.0;
									BathSum2 = 0.0;
									wSum = 0.0;
									if (Adia == 0)
									{
										for (int s = 0; s<smax ; ++s)
										{
											TildeSum1 = (MTil(s,It[ri[0]]+rs[0])+MTil(s,It[ri[1]]+rs[1])-MTil(s,It[ri[2]]+rs[2])-MTil(s,It[ri[3]]+rs[3]));
											TildeSum2 = (MTil(s,It[ri[4]]+rs[4])+MTil(s,It[ri[5]]+rs[5])-MTil(s,It[ri[6]]+rs[6])-MTil(s,It[ri[7]]+rs[7]));
											/*
											//	HACK this should lead to integrals with a negative real part. 
											if ( (TildeSum1)*(TildeSum2) > 0.0 && es > 0.0)
											{
												TildeSum1 *= -1.0;
											}
											else if ( (TildeSum1)*(TildeSum2) < 0.0 && es < 0.0)
											{
												TildeSum1 *= -1.0;
											}	
											*/
											BathSum0 -= (TildeSum1)*(TildeSum2)*C0(s);
											BathSum1 -= (TildeSum1)*(TildeSum2)*C1(s);
											BathSum2 -= (TildeSum1)*(TildeSum2)*C2(s);
											wSum -= Coths(s)*(TildeSum1+TildeSum2)*(TildeSum1+TildeSum2)/2.0;						
										}
									}
									if (TR)
										es *= -1.0;		
									f0(It[0],It[1],It[2],It[3]) = exp(wSum + BathSum0 + mimi*t0*es);
									f1(It[0],It[1],It[2],It[3]) = exp(wSum + BathSum1 + mimi*t1*es);
									f2(It[0],It[1],It[2],It[3]) = exp(wSum + BathSum2 + mimi*t2*es);
								}								
							}					
						}
					}
				}
				else if (LoopDim == 5)
				{
					for (It[0] = 0 ; It[0] < ci[0] ; It[0]=It[0]+1)
					{
						for (It[1] = 0 ; It[1] < ci[1] ; It[1]=It[1]+1)
						{
							for (It[2] = 0 ; It[2] < ci[2] ; It[2]=It[2]+1)
							{
								for (It[3] = 0 ; It[3] < ci[3] ; It[3]=It[3]+1)
								{
									for (It[4] = 0 ; It[4] < ci[4] ; It[4]=It[4]+1)
									{
										es = EDSTensor(0,It[si[0]]) + EDSTensor(1,It[si[1]]) + EDSTensor(2,It[si[2]]) + EDSTensor(3,It[si[3]]);
										if (abs(es) > UVCut && Adia == 0)
										{
											if (TR) 
												CI(It[0],It[1],It[2],It[3],It[4]) = conj((1.0-( cos(t1*es) - imi*sin(t1*es)))/(imi*es));
											else 
												CI(It[0],It[1],It[2],It[3],It[4]) = (1.0-( cos(t1*es) - imi*sin(t1*es)))/(imi*es);
											continue; 
										}										
										BathSum0 = 0.0;
										BathSum1 = 0.0;
										BathSum2 = 0.0;
										wSum = 0.0;
										if (Adia == 0)
										{
											for (int s = 0; s<smax ; ++s)
											{
												TildeSum1 = (MTil(s,It[ri[0]]+rs[0])+MTil(s,It[ri[1]]+rs[1])-MTil(s,It[ri[2]]+rs[2])-MTil(s,It[ri[3]]+rs[3]));
												TildeSum2 = (MTil(s,It[ri[4]]+rs[4])+MTil(s,It[ri[5]]+rs[5])-MTil(s,It[ri[6]]+rs[6])-MTil(s,It[ri[7]]+rs[7]));
											/*
											//	HACK this should lead to integrals with a negative real part. 
											if ( (TildeSum1)*(TildeSum2) > 0.0 && es > 0.0)
											{
												TildeSum1 *= -1.0;
											}
											else if ( (TildeSum1)*(TildeSum2) < 0.0 && es < 0.0)
											{
												TildeSum1 *= -1.0;
											}	
											*/
												BathSum0 -= (TildeSum1)*(TildeSum2)*C0(s);
												BathSum1 -= (TildeSum1)*(TildeSum2)*C1(s);
												BathSum2 -= (TildeSum1)*(TildeSum2)*C2(s);											
												wSum -= Coths(s)*(TildeSum1+TildeSum2)*(TildeSum1+TildeSum2)/2.0;						
											}
										}
										if (TR)
											es *= -1.0;											
										f0(It[0],It[1],It[2],It[3],It[4]) = exp(wSum + BathSum0 + mimi*t0*es);
										f1(It[0],It[1],It[2],It[3],It[4]) = exp(wSum + BathSum1 + mimi*t1*es);
										f2(It[0],It[1],It[2],It[3],It[4]) = exp(wSum + BathSum2 + mimi*t2*es);
									}															
								}								
							}					
						}
					}
				}
				else if (LoopDim == 6) 
				{
					for (It[0] = 0 ; It[0] < ci[0] ; It[0]=It[0]+1)
					{
						for (It[1] = 0 ; It[1] < ci[1] ; It[1]=It[1]+1)
						{
							for (It[2] = 0 ; It[2] < ci[2] ; It[2]=It[2]+1)
							{
								for (It[3] = 0 ; It[3] < ci[3] ; It[3]=It[3]+1)
								{
									for (It[4] = 0 ; It[4] < ci[4] ; It[4]=It[4]+1)
									{
										for (It[5] = 0 ; It[5] < ci[5] ; It[5]=It[5]+1)
										{										
											es = EDSTensor(0,It[si[0]]) + EDSTensor(1,It[si[1]]) + EDSTensor(2,It[si[2]]) + EDSTensor(3,It[si[3]]);
											if (abs(es) > UVCut && Adia == 0)
											{
												if (TR) 
													CI(It[0],It[1],It[2],It[3],It[4],It[5]) = conj((1.0-( cos(t1*es) - imi*sin(t1*es)))/(imi*es));
												else 
													CI(It[0],It[1],It[2],It[3],It[4],It[5]) = (1.0-( cos(t1*es) - imi*sin(t1*es)))/(imi*es);
												continue; 
											}
											BathSum0 = 0.0;
											BathSum1 = 0.0;
											BathSum2 = 0.0;
											wSum = 0.0;
											if (Adia == 0)
											{
												for (int s = 0; s<smax ; ++s)
												{
													TildeSum1 = (MTil(s,It[ri[0]]+rs[0])+MTil(s,It[ri[1]]+rs[1])-MTil(s,It[ri[2]]+rs[2])-MTil(s,It[ri[3]]+rs[3]));
													TildeSum2 = (MTil(s,It[ri[4]]+rs[4])+MTil(s,It[ri[5]]+rs[5])-MTil(s,It[ri[6]]+rs[6])-MTil(s,It[ri[7]]+rs[7]));
											/*
											//	HACK this should lead to integrals with a negative real part. 
											if ( (TildeSum1)*(TildeSum2) > 0.0 && es > 0.0)
											{
												TildeSum1 *= -1.0;
											}
											else if ( (TildeSum1)*(TildeSum2) < 0.0 && es < 0.0)
											{
												TildeSum1 *= -1.0;
											}	
											*/
													BathSum0 -= (TildeSum1)*(TildeSum2)*C0(s);
													BathSum1 -= (TildeSum1)*(TildeSum2)*C1(s);
													BathSum2 -= (TildeSum1)*(TildeSum2)*C2(s);											
													wSum -= Coths(s)*(TildeSum1+TildeSum2)*(TildeSum1+TildeSum2)/2.0;						
												}
											}
											if (TR)
												es *= -1.0;
											f0(It[0],It[1],It[2],It[3],It[4],It[5]) = exp(wSum + BathSum0 + mimi*t0*es);
											f1(It[0],It[1],It[2],It[3],It[4],It[5]) = exp(wSum + BathSum1 + mimi*t1*es);
											f2(It[0],It[1],It[2],It[3],It[4],It[5]) = exp(wSum + BathSum2 + mimi*t2*es);
										}
									}															
								}								
							}					
						}
					} 
				}	
				else 
				{
					cout << "ERROR... WRONG NUM BOSON INDICES " << endl; 
				}
				"""
		weave.inline(code,
					['t0','t1','t2','f0','f1','f2','CI','C0','C1','C2','EDSTensor','MTil','ShapeOfSummed','Adia','TR',
					'Coths', 'smax','Unique','LoopDim','RedIndicies','RedTypes','RedShifts','SIndices','UVCut'],
					headers = ["<complex>","<iostream>"],
					global_dict = dict(), 					
					type_converters = scipy.weave.converters.blitz, 
					compiler = Params.Compiler,
					verbose = 1)
		self.CurrentIntegral += (f0*W3[0]+f1*W3[1]+f2*W3[2])*((b-a)/2.0)					
		self.OldTime = T
		return 


	# this is the version uses a continuous spectral density. 
	# with only a single coupling constant. 
	def UpdateTermContBCF(self,T):
		if (T == self.OldTime): 
			return		
		T3 = numpy.array([-sqrt(3.0/5.0),0.0,sqrt(3.0/5.0)])
		W3 = numpy.array([5.0/9.0,8.0/9.0,5.0/9.0])		
		a = self.OldTime
		b = T
		ts = T3*((b-a)/2.0) + ((a+b)/2.0)
		t0 = float(ts[0])
		t1 = float(ts[1])
		t2 = float(ts[2])
		C0 = self.BosInf.CAt(ts[0])
		C1 = self.BosInf.CAt(ts[1])
		C2 = self.BosInf.CAt(ts[2])
		OhmicEqCF = self.BosInf.OhmicEqCF
		CI = self.CurrentIntegral
		f0 = self.IatT0
		f1 = self.IatT1
		f2 = self.IatT2
		f0*=0.0
		f1*=0.0
		f2*=0.0		
		EDSTensor = self.EDSTensor
		MTil = self.BosInf.MTilde
		Freqs = self.BosInf.Freqs
		Coths = self.BosInf.Coths
		smax = self.BosInf.NBos
		Unique = self.UniqueBCIndices
		# 4-19-2012 It can happen that the unique indices2 aren't contiguous
		# Check to make sure that isn't broken here. 
		LoopDim = len(self.UniqueBCIndices)
		try : 
			RedIndicies = [self.UniqueBCIndices.index(self.BCIndices[i]) for i in range(8)]
			RedTypes = map(self.HPTypeDummy, RedIndicies)
			RedShifts = [ Params.nocc if X == 1 else 0 for X in RedTypes]
			SIndices = [ self.UniqueBCIndices.index(X[1]) for X in self.EDeltas ] 													
			TypesOfSummed = map(self.HPTypeDummy,self.UniqueBCIndices)
			ShapeOfSummed = [ Params.nocc if X==0 else Params.nvirt for X in TypesOfSummed]
		except Exception as Ex: 
			self.Print()
			raise		
		Adia = Params.Adiabatic 
		UVCut = Params.UVCutoff
		TR = int(self.TimeReversed)
		code =	"""
				#line 2401 "Wick.py"
				using namespace std;
				int ri[8]; 
				int rs[8];
				int si[4]; 
				int ci[6];
				for (int k = 0 ; k<8; ++k)
				{
					ri[k] = (int)RedIndicies[k];
					rs[k] = (int)RedShifts[k]; // Picks out Occupied or V
				}
				for (int k = 0 ; k<4; ++k)
				{
					si[k] = (int)SIndices[k];
				}
				for (int k = 0 ; k<(int)LoopDim; ++k)
				{
					ci[k] = (int)ShapeOfSummed[k]; 					
				}	
				complex<double> imi(0.0,1.0); 
				complex<double> mimi(0.0,-1.0); 				
				complex<double> BathSum0,BathSum1,BathSum2;				
				double TildeSum1 = 0.0; 
				double TildeSum2 = 0.0;
				double es;  				
				double wSum = 0.0; // Sum of equilibrium piece. 
				int It[6];
                int s=0;
				if (LoopDim == 4)
				{
					for (It[0] = 0 ; It[0] < ci[0] ; It[0]=It[0]+1)
					{
						for (It[1] = 0 ; It[1] < ci[1] ; It[1]=It[1]+1)
						{
							for (It[2] = 0 ; It[2] < ci[2] ; It[2]=It[2]+1)
							{
								for (It[3] = 0 ; It[3] < ci[3] ; It[3]=It[3]+1)
								{
									es = EDSTensor(0,It[si[0]]) + EDSTensor(1,It[si[1]]) + EDSTensor(2,It[si[2]]) + EDSTensor(3,It[si[3]]);
									if (abs(es) > UVCut && Adia == 0)
									{
										if (TR) 
											CI(It[0],It[1],It[2],It[3]) = conj((1.0-( cos(t1*es) - imi*sin(t1*es)))/(imi*es));
										else 
											CI(It[0],It[1],It[2],It[3]) = (1.0-( cos(t1*es) - imi*sin(t1*es)))/(imi*es);
										continue; 
									}
									BathSum0 = 0.0;
									BathSum1 = 0.0;
									BathSum2 = 0.0;
									wSum = 0.0;
									if (Adia == 0)
									{
            
                                        TildeSum1 = (MTil(s,It[ri[0]]+rs[0])+MTil(s,It[ri[1]]+rs[1])-MTil(s,It[ri[2]]+rs[2])-MTil(s,It[ri[3]]+rs[3]));
                                        TildeSum2 = (MTil(s,It[ri[4]]+rs[4])+MTil(s,It[ri[5]]+rs[5])-MTil(s,It[ri[6]]+rs[6])-MTil(s,It[ri[7]]+rs[7]));        
                                        BathSum0 -= (TildeSum1)*(TildeSum2)*C0(s);
                                        BathSum1 -= (TildeSum1)*(TildeSum2)*C1(s);
                                        BathSum2 -= (TildeSum1)*(TildeSum2)*C2(s);
                                        wSum -= (TildeSum1+TildeSum2)*(TildeSum1+TildeSum2)*OhmicEqCF;
									}
									if (TR)
										es *= -1.0;									
									f0(It[0],It[1],It[2],It[3]) = exp(wSum + BathSum0 + mimi*t0*es);
									f1(It[0],It[1],It[2],It[3]) = exp(wSum + BathSum1 + mimi*t1*es);
									f2(It[0],It[1],It[2],It[3]) = exp(wSum + BathSum2 + mimi*t2*es);
								}								
							}					
						}
					}
				}
				else if (LoopDim == 5)
				{
					for (It[0] = 0 ; It[0] < ci[0] ; It[0]=It[0]+1)
					{
						for (It[1] = 0 ; It[1] < ci[1] ; It[1]=It[1]+1)
						{
							for (It[2] = 0 ; It[2] < ci[2] ; It[2]=It[2]+1)
							{
								for (It[3] = 0 ; It[3] < ci[3] ; It[3]=It[3]+1)
								{
									for (It[4] = 0 ; It[4] < ci[4] ; It[4]=It[4]+1)
									{
										es = EDSTensor(0,It[si[0]]) + EDSTensor(1,It[si[1]]) + EDSTensor(2,It[si[2]]) + EDSTensor(3,It[si[3]]);
										if (abs(es) > UVCut && Adia == 0)
										{
											if (TR) 
												CI(It[0],It[1],It[2],It[3],It[4]) = conj((1.0-( cos(t1*es) - imi*sin(t1*es)))/(imi*es));
											else 
												CI(It[0],It[1],It[2],It[3],It[4]) = (1.0-( cos(t1*es) - imi*sin(t1*es)))/(imi*es);
											continue; 
										}
										BathSum0 = 0.0;
										BathSum1 = 0.0;
										BathSum2 = 0.0;
										wSum = 0.0;
										if (Adia == 0)
										{
            
                                            TildeSum1 = (MTil(s,It[ri[0]]+rs[0])+MTil(s,It[ri[1]]+rs[1])-MTil(s,It[ri[2]]+rs[2])-MTil(s,It[ri[3]]+rs[3]));
                                            TildeSum2 = (MTil(s,It[ri[4]]+rs[4])+MTil(s,It[ri[5]]+rs[5])-MTil(s,It[ri[6]]+rs[6])-MTil(s,It[ri[7]]+rs[7]));
                                            BathSum0 -= (TildeSum1)*(TildeSum2)*C0(s);
                                            BathSum1 -= (TildeSum1)*(TildeSum2)*C1(s);
                                            BathSum2 -= (TildeSum1)*(TildeSum2)*C2(s);											
                                            wSum -= (TildeSum1+TildeSum2)*(TildeSum1+TildeSum2)*OhmicEqCF;
										}
										if (TR)
											es *= -1.0;
										f0(It[0],It[1],It[2],It[3],It[4]) = exp(wSum + BathSum0 + mimi*t0*es);
										f1(It[0],It[1],It[2],It[3],It[4]) = exp(wSum + BathSum1 + mimi*t1*es);
										f2(It[0],It[1],It[2],It[3],It[4]) = exp(wSum + BathSum2 + mimi*t2*es);
									}															
								}								
							}					
						}
					}
				}
				else if (LoopDim == 6) 
				{
					for (It[0] = 0 ; It[0] < ci[0] ; It[0]=It[0]+1)
					{
						for (It[1] = 0 ; It[1] < ci[1] ; It[1]=It[1]+1)
						{
							for (It[2] = 0 ; It[2] < ci[2] ; It[2]=It[2]+1)
							{
								for (It[3] = 0 ; It[3] < ci[3] ; It[3]=It[3]+1)
								{
									for (It[4] = 0 ; It[4] < ci[4] ; It[4]=It[4]+1)
									{
										for (It[5] = 0 ; It[5] < ci[5] ; It[5]=It[5]+1)
										{										
											es = EDSTensor(0,It[si[0]]) + EDSTensor(1,It[si[1]]) + EDSTensor(2,It[si[2]]) + EDSTensor(3,It[si[3]]);
											if (abs(es) > UVCut && Adia == 0)
											{
												if (TR) 
													CI(It[0],It[1],It[2],It[3],It[4],It[5]) = conj((1.0-( cos(t1*es) - imi*sin(t1*es)))/(imi*es));
												else 
													CI(It[0],It[1],It[2],It[3],It[4],It[5]) = (1.0-( cos(t1*es) - imi*sin(t1*es)))/(imi*es);
												continue; 
											}
											BathSum0 = 0.0;
											BathSum1 = 0.0;
											BathSum2 = 0.0;
											wSum = 0.0;
											if (Adia == 0)
											{
                                                TildeSum1 = (MTil(s,It[ri[0]]+rs[0])+MTil(s,It[ri[1]]+rs[1])-MTil(s,It[ri[2]]+rs[2])-MTil(s,It[ri[3]]+rs[3]));
                                                TildeSum2 = (MTil(s,It[ri[4]]+rs[4])+MTil(s,It[ri[5]]+rs[5])-MTil(s,It[ri[6]]+rs[6])-MTil(s,It[ri[7]]+rs[7]));
                                                BathSum0 -= (TildeSum1)*(TildeSum2)*C0(s);
                                                BathSum1 -= (TildeSum1)*(TildeSum2)*C1(s);
                                                BathSum2 -= (TildeSum1)*(TildeSum2)*C2(s);											
                                                wSum -= (TildeSum1+TildeSum2)*(TildeSum1+TildeSum2)*OhmicEqCF;
                                            }
											if (TR)
												es *= -1.0;
											f0(It[0],It[1],It[2],It[3],It[4],It[5]) = exp(wSum + BathSum0 + mimi*t0*es);
											f1(It[0],It[1],It[2],It[3],It[4],It[5]) = exp(wSum + BathSum1 + mimi*t1*es);
											f2(It[0],It[1],It[2],It[3],It[4],It[5]) = exp(wSum + BathSum2 + mimi*t2*es);
										}
									}															
								}								
							}					
						}
					}  
				}	
				else 
				{
					cout << "ERROR... WRONG NUM BOSON INDICES " << endl; 
				}
				"""
		weave.inline(code,
					['t0','t1','t2','f0','f1','f2','CI','C0','C1','C2','EDSTensor','MTil','ShapeOfSummed','Adia','TR',
					'Coths', 'smax','Unique','LoopDim','RedIndicies','RedTypes','RedShifts','SIndices','UVCut','OhmicEqCF'],
					headers = ["<complex>","<iostream>"],
					global_dict = dict(), 					
					type_converters = scipy.weave.converters.blitz, 
					compiler = Params.Compiler,
					verbose = 1)
		self.CurrentIntegral += (f0*W3[0]+f1*W3[1]+f2*W3[2])*((b-a)/2.0)					
		self.OldTime = T
		return 
				
	# this is the version which is actually called if adiabatic == 0 
	def CompareWithExact(self):
		deltat = 0.0
		CI = self.CurrentIntegral
		OI = self.OldIntegrand
		EDSTensor = self.EDSTensor
		MTil = self.BosInf.MTilde
		Freqs = self.BosInf.Freqs
		Coths = self.BosInf.Coths
		Css = self.BosInf.Css
		smax = self.BosInf.NBos
		Unique = self.UniqueBCIndices		
		# 4-19-2012 It can happen that the unique indices2 aren't contiguous
		# Check to make sure that isn't broken here. 
		LoopDim = len(self.UniqueBCIndices)
		try : 
			RedIndicies = [self.UniqueBCIndices.index(self.BCIndices[i]) for i in range(8)]
			RedTypes = map(self.HPTypeDummy, RedIndicies)
			RedShifts = [ Params.nocc if X == 1 else 0 for X in RedTypes]
			SIndices = [ self.UniqueBCIndices.index(X[1]) for X in self.EDeltas ] 													
			TypesOfSummed = map(self.HPTypeDummy,self.UniqueBCIndices)
			ShapeOfSummed = [ Params.nocc if X==0 else Params.nvirt for X in TypesOfSummed]
		except Exception as Ex: 
			self.Print()
			raise		
		Adia = Params.Adiabatic 
		Eta = Params.MarkovEta
		UVCut = Params.UVCutoff
		code =	"""
				#line 2612 "Wick.py"
				using namespace std;
				int ri[8]; 
				int rs[8];
				int si[4]; 
				int ci[6];
				for (int k = 0 ; k<8; ++k)
				{
					ri[k] = (int)RedIndicies[k];
					rs[k] = (int)RedShifts[k]; // Picks out Occupied or V
				}
				for (int k = 0 ; k<4; ++k)
				{
					si[k] = (int)SIndices[k];
				}
				for (int k = 0 ; k<(int)LoopDim; ++k)
				{
					ci[k] = (int)ShapeOfSummed[k]; 					
				}	
				complex<double> NewBCF,es; 
				complex<double> imi(0.0,1.0); 
				complex<double> mimi(0.0,-1.0); 				
				complex<double> BathSum;				
				double TildeSum1 = 0.0; 
				double TildeSum2 = 0.0; 				
				double wSum = 0.0; // Sum of equilibrium piece. 
				int It[6];
				if (LoopDim == 4)
				{
					for (It[0] = 0 ; It[0] < ci[0] ; It[0]=It[0]+1)
					{
						for (It[1] = 0 ; It[1] < ci[1] ; It[1]=It[1]+1)
						{
							for (It[2] = 0 ; It[2] < ci[2] ; It[2]=It[2]+1)
							{
								for (It[3] = 0 ; It[3] < ci[3] ; It[3]=It[3]+1)
								{
									es = complex<double>(0.0,0.0);
									for (int j = 0 ; j < 4; ++j) 
										es += EDSTensor(j,It[si[j]]);
									cout << CI(It[0],It[1],It[2],It[3]) << " : " << 1.0/(imi*es + Eta) << endl; 
								}								
							}					
						}
					}
				}
				else if (LoopDim == 5)
				{
					for (It[0] = 0 ; It[0] < ci[0] ; It[0]=It[0]+1)
					{
						for (It[1] = 0 ; It[1] < ci[1] ; It[1]=It[1]+1)
						{
							for (It[2] = 0 ; It[2] < ci[2] ; It[2]=It[2]+1)
							{
								for (It[3] = 0 ; It[3] < ci[3] ; It[3]=It[3]+1)
								{
									for (It[4] = 0 ; It[4] < ci[4] ; It[4]=It[4]+1)
									{
										es = complex<double>(0.0,0.0);
										for (int j = 0 ; j < 4; ++j) 
											es += EDSTensor(j,It[si[j]]);
										cout << CI(It[0],It[1],It[2],It[3],It[4]) << " : " << 1.0/(imi*es + Eta) << endl; 
									}															
								}								
							}					
						}
					}
				}
				else if (LoopDim == 6) 
				{
					for (It[0] = 0 ; It[0] < ci[0] ; It[0]=It[0]+1)
					{
						for (It[1] = 0 ; It[1] < ci[1] ; It[1]=It[1]+1)
						{
							for (It[2] = 0 ; It[2] < ci[2] ; It[2]=It[2]+1)
							{
								for (It[3] = 0 ; It[3] < ci[3] ; It[3]=It[3]+1)
								{
									for (It[4] = 0 ; It[4] < ci[4] ; It[4]=It[4]+1)
									{
										for (It[5] = 0 ; It[5] < ci[5] ; It[5]=It[5]+1)
										{
											es = complex<double>(0.0,0.0);
											for (int j = 0 ; j < 4; ++j) 
												es += EDSTensor(j,It[si[j]]);
											cout << CI(It[0],It[1],It[2],It[3],It[4],It[5]) << " : " << 1.0/(imi*es + Eta) << endl; 
										}
									}															
								}								
							}					
						}
					}  
				}	
				else 
				{
					cout << "ERROR... WRONG NUM BOSON INDICES " << endl; 
				}
				"""		
		weave.inline(code,
					['CI','OI','EDSTensor','MTil','Freqs','ShapeOfSummed','Adia','Eta', 'UVCut',
					'Css','Coths', 'smax','Unique','LoopDim','RedIndicies','RedTypes','RedShifts','SIndices'],
					headers = ["<complex>","<iostream>"],
					global_dict = dict(), 					
					type_converters = scipy.weave.converters.blitz, 
					compiler = Params.Compiler,
					verbose = 1)
		return 		

	# This is a special use routine, Specifically for the homogeneous terms of the polarization propagator. 
	# if adiabatic == 2 then This routine isn't called. 
	# if adiabatic == 1 then exp i delta s & t are integrated numerically. 
	# if adiabatic == 0 then the boson correlation function is also integrated. 
	# more documentation is available in the Params object 
	def EvaluateMyselfTrivially(self,OldState,Tt):
		SummedIndices = list(set(self.AllIndices())) # These are then sorted. 
		TypesOfSummed = map(self.HPTypeDummy,SummedIndices)		
		ShapeOfSummed = range(6)
		for i in rlen(ShapeOfSummed): 
			if i in SummedIndices: 
				if TypesOfSummed[SummedIndices.index(i)] == 0 : 
					ShapeOfSummed[i] = Params.nocc
				else :
					ShapeOfSummed[i] = Params.nvirt				
			else :
				ShapeOfSummed[i] = 1
		Vtn = None 
		Vsn = None 
		VtIndices = None
		VsIndices = None 
		RhoIndices = None
		VacIndices = None 
		r1 = None 
		VacShape = None	
		for T in self.Tensors: 
			if (T.IsVac()):  
				VacIndices = T.indices
				VacShape = OldState[T.NonVacName()].shape
			elif T.Time == "t" : # This is V at time t. 
				Vtn=T.Name()
				VtIndices=T.indices
			elif T.Time == "s": 
				Vsn=T.Name()
				VsIndices=T.indices
			elif (T.name == "r1"): 
				r1=OldState[T.NonVacName()]
				RhoIndices = T.indices
		if (RhoIndices == None or VacShape == None or r1 == None): 
			raise Exception("Error, didn't find rho") 	
		CIndices = self.UniqueBCIndices
		NumCIndices = len(CIndices)
		RankOfSummation = len(CIndices)
		Ext = numpy.zeros(shape=VacShape, dtype = complex)
		Vt = Integrals[Vtn]
		Vs = Integrals[Vsn]
		Ci = self.CurrentIntegral
		code =	"""
				#line 2772 "Wick.py"
				using namespace std;
				int vti[4]; 
				int vsi[4]; 
				int ri[2];
				int ei[2];				
				int ci[6];
				int SOS[6];	
				int nci = (int)NumCIndices; 			
				for (int k = 0 ; k<4; ++k)
				{
					vti[k] = (int)VtIndices[k];
					vsi[k] = (int)VsIndices[k];
				}
				for (int k=0 ; k<6 ; ++k)
					SOS[k] = (int)ShapeOfSummed[k];
				for (int k = 0 ; k<nci; ++k)
					ci[k] = (int)CIndices[k];
				for (int k = 0 ; k<2; ++k)
				{
					ri[k] = (int)RhoIndices[k];
					ei[k] = (int)VacIndices[k];					
				}
				complex<double> r,ext,c;
				double vs,vt; 
				complex<double> imi(0.0,1.0); 				
				int It[6];
				for (It[0] = 0 ; It[0] < SOS[0] ; It[0]=It[0]+1)
				{
					for (It[1] = 0 ; It[1] < SOS[1] ; It[1]=It[1]+1)
					{
						for (It[2] = 0 ; It[2] < SOS[2] ; It[2]=It[2]+1)
						{
							for (It[3] = 0 ; It[3] < SOS[3] ; It[3]=It[3]+1)
							{
								for (It[4] = 0 ; It[4] < SOS[4] ; It[4]=It[4]+1)
								{
									for (It[5] = 0 ; It[5] < SOS[5] ; It[5]=It[5]+1)
									{
										r = r1(It[ri[0]],It[ri[1]]);
										if (r == 0.0)
											continue; 
										vt = Vt(It[vti[0]],It[vti[1]],It[vti[2]],It[vti[3]]);
										if (vt == 0.0)
											continue;											
										vs = Vs(It[vsi[0]],It[vsi[1]],It[vsi[2]],It[vsi[3]]);
										if (vs == 0.0)
											continue;
				
										if (nci==3)
											c = Ci(It[ci[0]],It[ci[1]],It[ci[2]]);
										else if (nci==4)
											c = Ci(It[ci[0]],It[ci[1]],It[ci[2]],It[ci[3]]);
										else if (nci==5)
											c = Ci(It[ci[0]],It[ci[1]],It[ci[2]],It[ci[3]],It[ci[4]]);
										else if (nci==6)
											c = Ci(It[ci[0]],It[ci[1]],It[ci[2]],It[ci[3]],It[ci[4]],It[ci[5]]);
										else 
											cout << "Error, Wrong number of boson indices" << endl;

										// Ext(It[ei[0]],It[ei[1]]) += vt*vs*r*exp(imi*Tt*et)*c;
										// Since es = - et
										// 4-20-2012	Ext(It[ei[0]],It[ei[1]]) += vt*vs*r*(cos(Tt*es) - imi*sin(Tt*es))*c;
										//  The T exp is now included in c. 
										
										Ext(It[ei[0]],It[ei[1]]) += vt*vs*r*c;
											
										// In the Markov approximation c = I/\Delta_s with an imaginary part given by \eta. 
										// Differing from the analytical integral which has another term, -iExp(it\Delta_s)/\Delta_s
										// Let's check to see how accurate the two are...  (\tilde{M} = 0)											
										//	cout << "Markov integral. " <<  c << " Analytical Integral " << imi/(es) << endl; 
									}								
								}															
							}								
						}					
					}
				}
				"""
		weave.inline(code,
					['Ext','Vt','Vs','r1','VtIndices', 'Ci', 'CIndices', 
					'VsIndices','RhoIndices','ShapeOfSummed','VacIndices','NumCIndices'],
					global_dict = dict(), 					
					headers = ["<complex>","<iostream>"],
					type_converters = scipy.weave.converters.blitz, 
					compiler = Params.Compiler,
					verbose = 1)
		Ext *= self.PreFactor
		return Ext	

	# This is a special use routine, Specifically for the homogeneous terms of the polarization propagator. 
	# if adiabatic == 2 then This routine isn't called. 
	# if adiabatic == 1 then exp i delta s & t are integrated numerically. 
	# if adiabatic == 0 then the boson correlation function is also integrated. 
	# more documentation is available in the Params object 
	def EvaluateMatrixForm(self,OldState):
		SummedIndices = list(set(self.AllIndices())) # These are then sorted. 
		TypesOfSummed = map(self.HPTypeDummy,SummedIndices)		
		ShapeOfSummed = range(6)
		for i in rlen(ShapeOfSummed): 
			if i in SummedIndices: 
				if TypesOfSummed[SummedIndices.index(i)] == 0 : 
					ShapeOfSummed[i] = Params.nocc
				else :
					ShapeOfSummed[i] = Params.nvirt				
			else :
				ShapeOfSummed[i] = 1
		Vtn = None 
		Vsn = None 
		VtIndices = None
		VsIndices = None 
		RhoIndices = None
		VacIndices = None 
		r1 = None 
		VacShape = None	
		for T in self.Tensors: 
			if (T.IsVac()):  
				VacIndices = T.indices
				VacShape = OldState[T.NonVacName()].shape
			elif T.Time == "t" : # This is V at time t. 
				Vtn=T.Name()
				VtIndices=T.indices
			elif T.Time == "s": 
				Vsn=T.Name()
				VsIndices=T.indices
			elif (T.name == "r1"): 
				r1=OldState[T.NonVacName()]
				RhoIndices = T.indices
		if (RhoIndices == None or VacShape == None or r1 == None): 
			raise Exception("Error, didn't find rho") 	
		CIndices = self.UniqueBCIndices
		NumCIndices = len(CIndices)
		RankOfSummation = len(CIndices)
		Ext = numpy.zeros(shape=VacShape, dtype = complex)
		Vt = Integrals[Vtn]
		Vs = Integrals[Vsn]
		Ci = self.CurrentIntegral
		Ext = numpy.zeros(shape=(VacShape[0],VacShape[1],VacShape[0],VacShape[1]) , dtype = complex)	
		# TimeReversal is taken care of in the integrator. 
		code =	"""
				#line 2911 "Wick.py"
				using namespace std;
				int vti[4]; 
				int vsi[4]; 
				int ri[2];
				int ei[2];				
				int ci[6];
				int SOS[6];	
				int nci = (int)NumCIndices; 			
				for (int k = 0 ; k<4; ++k)
				{
					vti[k] = (int)VtIndices[k];
					vsi[k] = (int)VsIndices[k];
				}
				for (int k=0 ; k<6 ; ++k)
					SOS[k] = (int)ShapeOfSummed[k];
				for (int k = 0 ; k<nci; ++k)
					ci[k] = (int)CIndices[k];
				for (int k = 0 ; k<2; ++k)
				{
					ri[k] = (int)RhoIndices[k];
					ei[k] = (int)VacIndices[k];					
				}
				complex<double> r,ext,c;
				double vs,vt; 
				complex<double> imi(0.0,1.0); 				
				int It[6];
				for (It[0] = 0 ; It[0] < SOS[0] ; It[0]=It[0]+1)
				{
					for (It[1] = 0 ; It[1] < SOS[1] ; It[1]=It[1]+1)
					{
						for (It[2] = 0 ; It[2] < SOS[2] ; It[2]=It[2]+1)
						{
							for (It[3] = 0 ; It[3] < SOS[3] ; It[3]=It[3]+1)
							{
								for (It[4] = 0 ; It[4] < SOS[4] ; It[4]=It[4]+1)
								{
									for (It[5] = 0 ; It[5] < SOS[5] ; It[5]=It[5]+1)
									{
										vt = Vt(It[vti[0]],It[vti[1]],It[vti[2]],It[vti[3]]);
										if (vt == 0.0)
											continue;											
										vs = Vs(It[vsi[0]],It[vsi[1]],It[vsi[2]],It[vsi[3]]);
										if (vs == 0.0)
											continue;
				
										if (nci==3)
											c = Ci(It[ci[0]],It[ci[1]],It[ci[2]]);
										else if (nci==4)
											c = Ci(It[ci[0]],It[ci[1]],It[ci[2]],It[ci[3]]);
										else if (nci==5)
											c = Ci(It[ci[0]],It[ci[1]],It[ci[2]],It[ci[3]],It[ci[4]]);
										else if (nci==6)
											c = Ci(It[ci[0]],It[ci[1]],It[ci[2]],It[ci[3]],It[ci[4]],It[ci[5]]);
										else 
											cout << "Error, Wrong number of boson indices" << endl;

										// Ext(It[ei[0]],It[ei[1]]) += vt*vs*r*exp(imi*Tt*et)*c;
										// Since es = - et
										// 4-20-2012	Ext(It[ei[0]],It[ei[1]]) += vt*vs*r*(cos(Tt*es) - imi*sin(Tt*es))*c;
										//  The T exp is now included in c. 	

										Ext(It[ei[0]],It[ei[1]],It[ri[0]],It[ri[1]]) += vt*vs*c;
										
										// In the Markov approximation c = I/\Delta_s with an imaginary part given by \eta. 
										// Differing from the analytical integral which has another term, -iExp(it\Delta_s)/\Delta_s
										// Let's check to see how accurate the two are...  (\tilde{M} = 0)											
										//	cout << "Markov integral. " <<  c << " Analytical Integral " << imi/(es) << endl; 
									}								
								}															
							}								
						}					
					}
				}
				"""
		weave.inline(code,
					['Ext','Vt','Vs','r1','VtIndices', 'Ci', 'CIndices',
					'VsIndices','RhoIndices','ShapeOfSummed','VacIndices','NumCIndices'],
					global_dict = dict(), 					
					headers = ["<complex>","<iostream>"],
					type_converters = scipy.weave.converters.blitz, 
					compiler = Params.Compiler,
					verbose = 1)
		Ext *= self.PreFactor
		return Ext	

# This is a special use routine, Specifically for the homogeneous terms of the polarization propagator. 	
	def EvaluateMyselfAdiabatically(self,OldState,Tt):
		SummedIndices = list(set(self.AllIndices())) # These are then sorted. 
		TypesOfSummed = map(self.HPTypeDummy,SummedIndices)		
		ShapeOfSummed = range(6)
		for i in rlen(ShapeOfSummed): 
			if i in SummedIndices: 
				if TypesOfSummed[SummedIndices.index(i)] == 0 : 
					ShapeOfSummed[i] = Params.nocc
				else :
					ShapeOfSummed[i] = Params.nvirt				
			else :
				ShapeOfSummed[i] = 1
		Vtn = None 
		Vsn = None 
		VtIndices = None
		VsIndices = None 
		RhoIndices = None
		EDTIndices = [X[1] for X in self.EDeltat]
		EDSIndices = [X[1] for X in self.EDeltas]		
		RhoIndices = None
		VacIndices = None 
		r1 = None 
		VacShape = None	
		try:
			for T in self.Tensors: 
				#print T.Time
				if (T.IsVac()):  
					VacIndices = T.indices
					VacShape = OldState[T.NonVacName()].shape
				elif T.Time == "t" : # This is V at time t. 
					Vtn=T.Name()
					VtIndices=T.indices	
				elif T.Time == "s": 
					Vsn=T.Name()
					VsIndices=T.indices
				elif (T.name == "r1"):
					r1=OldState[T.NonVacName()]				
					RhoIndices = T.indices
				elif T.Time == "i":
					print "We've got I's in this shit"
			if (RhoIndices == None or VacShape == None or r1 == None): 
				raise Exception("Error, didn't find rho") 	
		except KeyError:
			Term.Print

		Ext = numpy.zeros(shape=VacShape, dtype = complex)
		Vt = Integrals[Vtn]
		Vs = Integrals[Vsn]
		EDTTensor = self.EDTTensor
		EDSTensor = self.EDSTensor
		TR = int(self.TimeReversed)
		code =	"""
				#line 3050 "Wick.py"
				using namespace std;
				int vti[4]; 
				int vsi[4]; 
				int eti[4];
				int esi[4];
				int SOS[6];				
				int ri[2];
				int ei[2];				
				for (int k = 0 ; k<4; ++k)
				{
					vti[k] = (int)VtIndices[k];
					vsi[k] = (int)VsIndices[k];
					esi[k] = (int)EDSIndices[k];
					eti[k] = (int)EDTIndices[k];					
				}
				for (int k = 0 ; k<6; ++k)
					SOS[k] = (int)ShapeOfSummed[k];
				for (int k = 0 ; k<2; ++k)
				{
					ri[k] = (int)RhoIndices[k];
					ei[k] = (int)VacIndices[k];	
				}
				complex<double> r,ext;
				double vs,vt;
				double es; 
				complex<double> imi(0.0,1.0);
				int It[6];																
				for (It[0] = 0 ; It[0] < SOS[0] ; It[0]=It[0]+1)
				{
					for (It[1] = 0 ; It[1] < SOS[1] ; It[1]=It[1]+1)
					{
						for (It[2] = 0 ; It[2] < SOS[2] ; It[2]=It[2]+1)
						{
							for (It[3] = 0 ; It[3] < SOS[3] ; It[3]=It[3]+1)
							{
								for (It[4] = 0 ; It[4] < SOS[4] ; It[4]=It[4]+1)
								{
									for (It[5] = 0 ; It[5] < SOS[5] ; It[5]=It[5]+1)
									{																					
										r = r1(It[ri[0]],It[ri[1]]);										
										if ( r == 0.0)
											continue; 										
										vt = Vt(It[vti[0]],It[vti[1]],It[vti[2]],It[vti[3]]);
										if ( vt == 0.0 )
											continue; 
										vs = Vs(It[vsi[0]],It[vsi[1]],It[vsi[2]],It[vsi[3]]);
										if ( vs == 0.0 )
											continue; 
										es = 0.0;
										for (int j = 0 ; j < 4; ++j) 
											es += EDSTensor(j,It[esi[j]]); 																					
										// The sign of this integration is in question. 
										// The older code had: (1.0-exp(-1.0*imi*Tt*et))/(-1.0*imi*et);
										// The present code has. (1.0-exp(-imi*Tt*es))/(imi*es) based on mathmatica (attempts at integrating the correlation funciton.)
										// And since es = -et these are different by -1. 
										//	Ext(It[ei[0]],It[ei[1]]) += vt*vs*r*(1.0-exp(-imi*Tt*es))/(imi*es);
										if (es == 0.0)
										{
											Ext(It[ei[0]],It[ei[1]]) += vt*vs*r*Tt;
											continue; 
										}
										if (TR)
											Ext(It[ei[0]],It[ei[1]]) += vt*vs*r*conj((1.0-( cos(Tt*es) - imi*sin(Tt*es)))/(imi*es));
										else 
											Ext(It[ei[0]],It[ei[1]]) += vt*vs*r*(1.0-( cos(Tt*es) - imi*sin(Tt*es)))/(imi*es);
									}								
								}															
							}								
						}					
					}
				}
				"""
		weave.inline(code,
					['Ext','r1','Vt','Vs','VtIndices','EDTIndices','EDSIndices', 
					'VsIndices','RhoIndices','ShapeOfSummed', 'VacIndices','TR',
					'EDTTensor','EDSTensor','Tt'],
					global_dict = dict(), 
					headers = ["<complex>","<iostream>"],
					type_converters = scipy.weave.converters.blitz, 
					compiler = Params.Compiler,
					verbose = 1)	
		Ext *= self.PreFactor
		if (cmath.isnan(cmath.sqrt(numpy.sum(Ext*Ext)))): 
			print "Post contraction Is nan... "
			self.Print()
			raise Exception("Nan.")
		return Ext	
	
# -----------------------------------------------------------
# A list of second quantized operator strings. 
# where the list implies summation over all the strings. 
#

class ManyOpStrings(list):
	# Initalize with two lists of "OperatorString"s. 
	# the new list will be the tensor product of the two. 
	# hopefully they were normal ordered. 
	def __init__(self):
		self.InhomogeneousTerms = False 
		return 	
	def ToFermi(self): 
		tore = ManyOpStrings()
		for T in self: 
			tore.Add(T.ToFermi())
		return tore
	def clone(self): 
		return copy.deepcopy(self)
	# this can be useful to 
	# denote other times, etc. 
	def	GiveTime(self,sufx): 
		for Term in self: 
			Term.GiveTime(sufx)
		return
	# Debug... Remove later. 
	def CheckOpConsistency(self): 
		for Term in self: 
			for T in Term.Tensors: 
				W = set(tuple(map(lambda X: X[1], T.ops)))
				V = set(tuple(T.indices))
				if W != V: 
					print "Inconsistent Indices!!!"
					Term.Print()
					print T.Name()
					print T.ops
					print T.indices
					raise Exception("FUCK")
		return 
	def Subtract(self, OpList2): 
		n0 = len(self)
		self.extend(OpList2)
		for nn in range(n0,len(self)): 
			self[nn].PreFactor *= -1.0
		return 
	def MyConjugate(self):
		tore = self.clone()
		for T in tore: 
			T.MyConjugate()
		return tore
	def Add(self, OpList2): 
		self.extend(OpList2)
		return
	def MultiplyScalar(self,AScalar):
		for Term in self: 
			Term.MultiplyScalar(AScalar)
		return	

	#Right Acting Multiplication. 
	def Times(self, OpList2): 
		tmp = []
		dim1 = range(len(self)) 
		dim2 = range(len(OpList2))
		if len(self)*len(OpList2) > 1000: 
			print "Multipying ", len(self), " terms with ", len(OpList2), " terms"
		for one in dim1:
			for two in dim2:
				tmp2 = copy.deepcopy(self[one])
				tmp.append(tmp2.TensorProduct(OpList2[two]))
		del self[:]
		self.extend(tmp)
		return
	#Left Acting Multiplication. 
	def LeftTimes(self, OpList2): 
		tmp = []
		dim1 = range(len(self)) 
		dim2 = range(len(OpList2))
		if len(self)*len(OpList2) > 1000: 
			print "Multipying ", len(self), " terms with ", len(OpList2), " terms"
		for one in dim1:
			for two in dim2:
				tmp2 = copy.deepcopy(OpList2[two])
				tmp.append(tmp2.TensorProduct(self[one]))
		del self[:]
		self.extend(tmp)
		return	
		
	def ReDummy(self):
		for T in self: 
			T.ReDummy()
		return 
	# Assign boson correlation indices and time exponential	
	def AssignBCFandTE(self): 
		for T in self: 
			T.AssignBCFandTE()
		return 
	def AllReferencedTensors(self):
		tore = []
		for T in self: 
			tore.extend(T.AllReferencedTensors())
		return list(set(tuple(tore)))

#	Make Dummy copies of all my constituient tensors 
#	and update all the references. 
#	def DuplicateTensors(self,tag = 1):
#		for Term in self: 
#			Term.DuplicateTensors(tag)
#		return 
	def MakeTensorsWithin(self,AVectorOfTensors):
		for Term in self: 
			Term.MakeTensorsWithin(AVectorOfTensors)
		return 		
#	Routines which chop down the number of terms based on various reasons
#	we would like to ignore them. 
	def ConnectedPart(self): 
		NewTensors = ManyOpStrings()
		for Term in self: 
			if (not Term.IsSimple()): 
				if (Term.IsConnected()): 
					NewTensors.append(Term.clone())
		del self[:]
		self.Add(NewTensors)
		return
#	Select terms which are a net Deexcitation. 
	def QPAnnPart(self): 
		NewTensors = ManyOpStrings()
		for Term in self: 
			if (Term.QPRank() < 0):
				NewTensors.append(Term.clone())
		del self[:]
		self.Add(NewTensors)
		return
#	Select terms which are a net excitation
	def QPCreatePart(self): 
		NewTensors = ManyOpStrings()
		for Term in self: 
			if (Term.QPRank() > 0):
				NewTensors.append(Term.clone())
		del self[:]
		self.Add(NewTensors)
		return
#   Select Terms of a given QuasiParticle Rank. 
	def QPRankPart(self, ARank, Exclusive): 
		NewTensors = ManyOpStrings()
		for Term in self: 
			if (Exclusive): 
				if (Term.QPRank() != ARank):
					NewTensors.append(Term.clone())			
			else: 
				if (Term.QPRank() == ARank):
					NewTensors.append(Term.clone())
		del self[:]
		self.Add(NewTensors)
		return
	def NormalPart(self,Vac=0): 
		NewTensors = ManyOpStrings()
		for Term in self: 
			if (Term.IsNormal(Vac)):
				NewTensors.append(Term.clone())
		del self[:]
		self.Add(NewTensors)
		return
	#Fully uncontracted part. 
	def UnContractedPart(self): 
		NewTensors = ManyOpStrings()
		for Term in self: 
			if (len(Term.Deltas) == 0):
				NewTensors.append(Term.clone())
		del self[:]
		self.Add(NewTensors)
		return
	# No HF-like pieces which contract V to itself. 
	def NoSelfContractPart(self): 
		NewTensors = ManyOpStrings()
		for Term in self: 
			if Term.IsNotSelfContracted():
				NewTensors.append(Term.clone())
		del self[:]
		self.Add(NewTensors)
		return
	def FullyContractedPart(self): 
		NewTensors = ManyOpStrings()
		for Term in self: 
			if (len(Term.Deltas)*2 == len(Term.Ops)):
				NewTensors.append(Term.clone())
		del self[:]
		self.Add(NewTensors)
		return
	# This is useful shorthand since 
	# only odd particle number parts should contribute to some theories. 
	def OddParticlePart(self): 	
		NewTensors = ManyOpStrings()
		for Term in self: 
			if ((Term.NumberOfEachType()[1]+Term.NumberOfEachType()[3])%2!=0 ):
				NewTensors.append(Term.clone())
		del self[:]
		self.Add(NewTensors)
		return
	# This Removes any ph->hp type couplings.
	def ApplyTDA(self): 
		NewTensors = ManyOpStrings()
		for Term in self:
			ART = [T.Name() for T in Term.Tensors]
			for T in Term.Tensors: 
				if T.IsVac(): 
					if (T.NonVacName() in ART): 
						NewTensors.append(Term.clone())
		del self[:]
		self.Add(NewTensors)
		return

	# This is "specific" routine for perturbation theory which only 
	# allows beyond 1-particle maps. 
	# This assumes that the indices have already been made redundant by AssignBCF. 
	def QPart(self): 
		NewTensors = ManyOpStrings()
		for Term in self: 
# The rule is that 
# between the vac, Vt  ||  and  Vs rho, there needs to be more than a pair of hole-particle lines. 
				Vt = None 
				Vs = None 
				Vac = None 
				Rho = None 
				for T in Term.Tensors:
					if (T.RootName() == "v" and T.Time == "t"):
						Vt = T
					elif (T.IsVac()):
						Vac = T
					elif (T.RootName() == "v" and (T.Time == "s" or T.Time == "i")):
						Vs = T
					elif (T.RootName() == "r1"): 
						Rho = T
				LinesAcross = [0,0]
				for I in Vac.indices: 
					if I in Vs.indices: 
						LinesAcross[Term.HPTypeDummy(I)] += 1
					elif I in Rho.indices:
						LinesAcross[Term.HPTypeDummy(I)] += 1
				for I in Vt.indices: 
					if I in Vs.indices: 
						LinesAcross[Term.HPTypeDummy(I)] += 1
					elif I in Rho.indices:
						LinesAcross[Term.HPTypeDummy(I)] += 1
				if (LinesAcross[0] == 1 and LinesAcross[1] == 1): 
					#print "Disobeys Q-Space Projector: "
					#Term.Print()
					continue 
				else: 
					NewTensors.append(Term.clone())
		del self[:]
		self.Add(NewTensors)
		return	
	def AddConjugateTerms(self):
		print "Warning Add Conjugate Terms is a Hack."
		NewTensors = ManyOpStrings()
		for Term in self: 
			NewTensors.append(Term.ConjugateTerm())
		self.Add(NewTensors)
		return 
	def AddAntiHermitianCounterpart(self):
		NewTensors = ManyOpStrings()
		for Term in self: 
			NewTensors.append(Term.AntiHermitianCounterpart())
		self.Add(NewTensors)
		self.MultiplyScalar(0.5)
		return 
	# this is a hack to avoid calculating a certain class of terms in the polarization propagator. 
	def PermuteTimesOnV(self): 
		for Term in self: 
			Term.PermuteTimesOnV()
	# This is useful for debugging/approximating. 
	def TermsNotContainingTensor(self,Tnam): 
		NewTensors = ManyOpStrings()
		for Term in self: 
			if (not Term.ContainsTensor(Tnam)):
				NewTensors.append(Term.clone())
		del self[:]
		self.Add(NewTensors)			
		return 

	def TermsContainingTensor(self,Tnam): 
		NewTensors = ManyOpStrings()
		for Term in self: 
			if (Term.ContainsTensor(Tnam)):
				NewTensors.append(Term.clone())
		del self[:]
		self.Add(NewTensors)			
		return 

#   --------------------------------		
#   Collapses like terms by flipping symmetry, etc.  
#   --------------------------------		
	
	def CombineLikeTerms(self, Debug = False, Signed = True): 	
		self.ReDummy()
		for I in range(len(self)):
			for J in range(I+1,len(self)):
				if (self[I].PreFactor*self[J].PreFactor != 0.0): 
					self[I].CombineIfLike(self[J],Debug,Signed)
					if (Debug and self[J].PreFactor == 0.0): 
						print "Combined:"
						self[J].Print()
						print "Into:"
						self[I].Print()
		NewTensors = ManyOpStrings()
		for Term in self: 
			if (Term.PreFactor != 0.0): 
				NewTensors.append(Term.clone())
		del self[:]
		self.Add(NewTensors)			
		return 	

	# Perform Wick's theorem on every string in this list. 
	# Resulting in a MUCH larger list of normal-ordered strings. 
	# Can be much faster if you know what the dangling lines should be. (TypesClosingWith)
	# ----------
	# This version combines things totally ignoring signs and factors then rederives them. 
	# Based on a diagrammatic arguement.
	# ----------	
	# Only works for the Fermi Vaccum. 
	def NormalOrderDiagBigMemory(self, TypesClosingWith = [], Debug = False): 
		Sz = len(self)
		PrintSz = int(.1*Sz)
		if (len(self) > 100): 
			print "Normal Ordering ", len(self), " terms... "
		if Debug: 
			print "WickDiag: After Initial Reduction...: "
			for T in WorkingStrings: 
				T.Print()
			print "WickDiag ------------------"
		NormalStrings = ManyOpStrings()
		for Z in range(Sz):
			Wicked = True 
			if (Sz > 400 and Z%PrintSz==0): 
				print "Finished", Z ," out of ", Sz, ", Terms. There are ", len(NormalStrings), " normal terms now."
			InputChunk = ManyOpStrings() 
			InputChunk.Add([self[Z]])
			InputChunk[0].OrderHP()
			WorkingChunk = ManyOpStrings() # Partially Ordered Strings. 
			OutputChunk = ManyOpStrings() 
			while (Wicked):
				Wicked = False
				for Y in InputChunk:  
					X = Y.AntiCommuteFermi(TypesClosingWith)
					if X[0]: #Anticommutator was applied
						WorkingChunk.extend(X[1])
						Wicked = True
					else: #No anticommutator was applied, so term is normal ordered. 
						if (Y.ClosesWith(TypesClosingWith)): 
							OutputChunk.append(copy.deepcopy(Y))
				InputChunk, WorkingChunk = WorkingChunk, InputChunk
				del WorkingChunk[:]
			OutputChunk.CombineLikeTerms(Debug,False)
			NormalStrings.extend(OutputChunk)
		NormalStrings.CombineLikeTerms(Debug,False)
		# Now NormalStrings needs to be Resigned and ReFactored.	
		for T in NormalStrings: 
			T.DiagrammaticSignAndFactor()
			if (Debug): 
				print "After DiagSignAndFactor:"
				T.Print()		
		del self[:]
		self.extend(NormalStrings)


	# Perform Wick's theorem on every string in this list. 
	# Resulting in a MUCH larger list of normal-ordered strings. 
	# Can be much faster if you know what the dangling lines should be. (TypesClosingWith)
	# ----------
	# This version combines things totally ignoring signs and factors then rederives them. 
	# Based on a diagrammatic arguement.
	# ----------	
	# 1 = Genuine Vacuum
	# 0 = Fermi Vacuum. 
	def NormalOrderDiag(self, TypesClosingWith = [], Debug = False): 
		if (len(self) > 100): 
			print "Normal Ordering ", len(self), " terms... "
		Wicked = True
		WorkingStrings = copy.deepcopy(self)
		for T in WorkingStrings:	
			T.OrderHP() 
		WorkingStrings.CombineLikeTerms(Debug,False)
		if Debug: 
			print "WickDiag: After Initial Reduction...: "
			for T in WorkingStrings: 
				T.Print()
			print "WickDiag ------------------"

		while (Wicked):
			Wicked = False
			WKS2 = ManyOpStrings()
			for Z in WorkingStrings:
				if Debug: 
					print "-----------"
					print "Wicking Term: ", Z.Print()
				X = Z.AntiCommuteFermi(TypesClosingWith)
				if X[0]: #Anticommutator was applied
					if Debug: 				
						print "Applied Anticommutator, and got ", X[1][0].Print()
					WKS2.extend(X[1])
					Wicked = True
				else: 
					if Debug:
						print "Didn't apply any anticommutator..." 
					if (Z.ClosesWith(TypesClosingWith)): 
						WKS2.append(copy.deepcopy(Z))
						if (Debug): 
							print "Appended term to working list."
					elif (Debug): 
						print "Term didn't close with vac..." 
						Z.Print()
			if Debug:
				print "Result of a wick round:"
				for T in WKS2: 
					T.Print()
			# OrderHP has already been called on every term. 
			WKS2.CombineLikeTerms(Debug,False)
			if Debug:
				print "result of combination" 
				for T in WKS2:	
					T.Print()			
			del WorkingStrings[:]
			WorkingStrings = WKS2 #copy.deepcopy(WKS2)		
			
		# Now Workingstrings needs to be Resigned and ReFactored.	
		for T in WorkingStrings: 
			T.DiagrammaticSignAndFactor()
			if (Debug): 
				print "After DiagSignAndFactor:"
				T.Print()
		
		del self[:]
		self.extend(WorkingStrings)

	# Perform Wick's theorem on every string in this list. 
	# Resulting in a MUCH larger list of normal-ordered strings. 
	# Can be much faster if you know what the dangling lines should be. (TypesClosingWith)
	# 1 = Genuine Vacuum
	# 0 = Fermi Vacuum. 
	def NormalOrder(self,Vac = 0, TypesClosingWith = [], Debug = False): 
		if (len(self) > 100): 
			print "Normal Ordering ", len(self), " terms... "
		Wicked = True
		WorkingStrings = copy.deepcopy(self)
		while (Wicked):
			Wicked = False
			WKS2 = ManyOpStrings()
			for Z in WorkingStrings:
				if Debug: 
					print "-----------"
					print "Wicking Term: ", Z.Print()
				if (Vac == 0): 
					X = Z.AntiCommuteFermi(TypesClosingWith)
				elif (Vac == 1):
					X = Z.AntiCommuteGenuine(TypesClosingWith)
				if X[0]: #Anticommutator was applied
					if Debug: 				
						print "Applied Anticommutator, and got ", X[1][0].Print()
					WKS2.extend(X[1])
					Wicked = True
				else: 
					if Debug:
						print "Didn't apply any anticommutator..." 
					# in this case the term should already be Normal ordered
					# and must close with any specified vaccuum.
					if (Z.ClosesWith(TypesClosingWith)): 
						WKS2.append(copy.deepcopy(Z))
						if (Debug): 
							print "Appended term to working list."
					elif (Debug): 
						print "Term didn't close with vac..." 
						Z.Print()
			if Debug:
				print "Result of a wick round:"
			for T in WKS2:	
				if Debug:
					T.Print()
				T.OrderHP()  #Remove Redundancies. 
			WKS2.CombineLikeTerms(Debug,True)
			if Debug:
				print "result of combination" 
				for T in WKS2:	
					T.Print()			
			del WorkingStrings[:]
			WorkingStrings = copy.deepcopy(WKS2)		
		del self[:]
		self.extend(WorkingStrings)
		return
		
	def ClassesIContain(self):
		cls = lambda Z: tuple(Z.NumberOfEachType())
		tmp = tuple(map(cls, self))
		tmp2 = set(tmp)
		tmp3 = list(tmp2) 
		return tmp3
	# am I just a wrapper for a tensor? Or am I a list with sums? 
	def IsSimple(self): 
		for term in self: 
			if (not term.IsSimple()):
				return False
		return True
	def MakeBCFs(self,BosonInformation):
		if (Params.Adiabatic == 2): 
			return
		print "Making ", len(self), " Boson correlation integrals."	
		for Term in self: 
			Term.MakeBCF(BosonInformation)
			if (Params.Adiabatic == 4): 
				Term.MakeMatrixMarkov()
		return 
	def UpdateBCF(self,TimeInformation):
		for Term in self: 
			if (Params.ContBath): 
				Term.UpdateTermContBCF(TimeInformation)						
			else: 
				Term.UpdateTermBCF(TimeInformation)			
		return 

	# self should be a simple tensor
	# Other should be contracted tensors 
	# and should include "Vac" operators.
	# NewState and OldState should be StateVectors appropriate for the algebra
	def EvaluateContributions(self,Other,NewState,OldState, FieldInformation=None, TimeInformation=None, MultiplyBy = 1.0):
		if (not self.IsSimple()):
			print "EvaluateContributions() can only add, not solve linear problems :( " 
		if (TimeInformation != None and Params.Adiabatic != 2 and Params.Adiabatic != 3 and Params.Adiabatic != 4 and not Params.Undressed):
			Other.UpdateBCF(TimeInformation)
		for Term1 in Other:
			for Dest in self:
				# Dest must have the correct vaccuum contribution. 
				if (Term1.ContainsVacOf(Dest)): 
					Dest.Contract(Term1,NewState,OldState, TimeInformation , MultiplyBy, FieldInformation)
		return

	def Print(self, LatexFormat = False): 
		print "I am a list of ", len(self), " operator strings. "
		print "Containing terms of types: ", self.ClassesIContain()
		if (len(self)<10): 
			for ZZZ in self: 
				ZZZ.Print(LatexFormat)
		return 

#   --------------------------------		
#	Some tensor-arithmetical operations 
#   Need to be aware of the algebraic shape they represent. 
#   In which case those operations go here. 
#   --------------------------------		
	def Hermitize(self,VectorOfTensors): 
		if (not self.IsSimple()): 
			print "Can't Hermitize... " 
		# finds the conjugate pairs of elements in my tensors and makes
		# sure they are complex conjugates of one another. 
		ToPair = range(len(self))
		PairedUp = []
		for I in ToPair: 
			for J in ToPair[I:] : 
				if self[I].IsConjugateTo(self[J]): 
					PairedUp.append([I,J])
		for Pair in PairedUp: 
			self[Pair[0]].HermitizeWith(self[Pair[1]])
		return 
	def AntiSymmetrize(self,VectorOfTensors): 
		for Term in self: 
			Term.Antisymmetrize(VectorOfTensors)
		return 		


#-----------------------------------------
# Now shortcuts to making density matrices, amplitudes 
# and Hamiltonian bits with the above classes. 
# ----------------------------------------

def StringOfType(typ,rng):
	LL = lambda NN : [typ,NN]
	return map(LL,rng)
def UpDownString(num): 
	rng1 = range(num)
	rng2 = range(num,2*num)
	tore = StringOfType(5,rng1)
	tore.extend(StringOfType(6,rng2))
	return tore
# note representations of the Hamiltonian 
# Purely in terms of the 2 body density. (PRL 57, 4219)
# The characters matter. They are used to interface with the tensor dictionary.
h1_Gen = OperatorString(1.0,[],[[5,0],[6,1]],"h")
h2_Gen = OperatorString(0.25,[],[[5,0],[5,1],[6,2],[6,3]],"v")
H_Gen = ManyOpStrings()
H_Gen.Add([h1_Gen,h2_Gen])
h1_Fermi = h1_Gen.ToFermi() # generates a ManyOpStrings with all h,p combinations. 
h2_Fermi = h2_Gen.ToFermi()
H_Fermi = h1_Fermi.clone()
H_Fermi.Add(h2_Fermi)
Hn_Fermi = H_Fermi.clone()
Hn_Fermi.NormalPart(0) 
V_Fermi = h2_Gen.ToFermi()
Vn_Fermi = V_Fermi.clone()
Vn_Fermi.NormalPart(0) 

mux_Gen = OperatorString(1.0,[],[[5,0],[6,1]],"mux")
muy_Gen = OperatorString(1.0,[],[[5,0],[6,1]],"muy")
muz_Gen = OperatorString(1.0,[],[[5,0],[6,1]],"muz")
mux_Fermi = mux_Gen.ToFermi()
muy_Fermi = muy_Gen.ToFermi() 
muz_Fermi = muz_Gen.ToFermi()
mu_Fermi = ManyOpStrings()
mu_Fermi.Add(mux_Fermi)
mu_Fermi.Add(muy_Fermi)
mu_Fermi.Add(muz_Fermi)

BathGen = OperatorString(1.0,[],[[5,0],[6,1]],"B")
BathFermi = BathGen.ToFermi()
BathCoupling = ManyOpStrings()
BathCoupling.Add(BathFermi)

# Choices of DM normalization 
def RhosUpTo(rank,nrmztn = 2,NamePrefix=""): 
	if (nrmztn == 1) : 
	# rho_n = 1/n!<\psi a+_1a+_2..a+_n a_1a_2..a_n\psi>
	# this is the tr(Rho) = 1 Normalization. 
	# Trace Conditions:
		tore = []
		for r in range(1,rank): 
			tmp = OperatorString(1.0/math.factorial(r),[],UpDownString(r),NamePrefix+"r"+str(r))
			tore.append(tmp)
		return ManyOpStrings(tore)
	elif (nrmztn == 2) : 
	#tr rho(1) = N
	#tr rho(2) = N(N-1)/2
	#tr rho(M) = Binomial(N,M) = N!/(N-M)!
		tore = ManyOpStrings()
		for r in range(1,rank+1): 
			tmp = OperatorString(1.0,[],UpDownString(r),NamePrefix+"r"+str(r))
			tore.Add([tmp])
		return tore
# Some other standard Operators: 
	
def PureExcitations(Rank): 
	tore = ManyOpStrings()
	tore.Add( map(PureExcitation, range(1,Rank+1)) )
	return tore
	
def PureExcitation(Rank): 
	TMP = map(lambda X:[1,X],range(Rank))
	TMP2 = map(lambda Z:[2,Z],range(Rank,2*Rank))
	TMP.extend(TMP2)
	return OperatorString(pow(1.0/(math.factorial(Rank)),2.0),[],TMP,"T"+str(Rank)+"_hp")
			
