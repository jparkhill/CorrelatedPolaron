# scipy crap 
import matplotlib
# Makes figures without popup
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy, scipy
from scipy import optimize, special
from numpy import array
from numpy import linalg
# standard crap
import os, sys, time, math, re, random
from time import gmtime, strftime
from types import * 
from itertools import izip
from heapq import nlargest
from multiprocessing import Process, Queue, Pipe
from math import pow, exp, cos, sin, log, pi, sqrt
from numgrad import * 
output = sys.stdout

# the only important outputs of this object are: 
# HAdiaVert - The diagonalized hamiltonian matrix (liouville)
# MakeRedfield - The redfield operator (lioville)
# MuMatrix(i) which is the dipole matrix in the basis that diagonalizes the H. 

class VibronicData: 
	def __init__(self,MolName,IfFiniteDifferences=False,IfAdiabatic = True,IfInteractive = False): 
		self.MolName = MolName	
		print "VibronicData, Init. ", MolName
		self.Temp = 293.15
		self.homo = 50*AuPerWavenumber	# The homogeneous broadening. 
		self.nexsts = 6
		self.EEn = [] # Vertical Electronic Energies
		self.VEn = [] # Adiabatic Vibrational Energies we have sdnc's for. 
		self.AllVEn = [] 
		self.VertOmegaij = [] # Adiabatic(Ver + Reorg) Transition energies. 
		self.TMom = [] # Dipole matrix. 	
		self.OTMom = [] #Transistion moments in orca.
		self.OEEn = [] #Transistion Energies in orca.
		self.Dev = [] # Dimensionless displacements.
		self.QEEn = [] # Electronic Energies in Qchem 
		self.QTMom = [] # TransMom in Qchem 
		# Note: the number of vibrations read from .asa.inp 
		# may be less than 3N, because of small displacements. 
		# these quantites are for finite difference displacements. 
		self.CartesianModes = [] # Nuclear modes 
		self.CartesianHessian = []
		self.FD_Dev = [] # dimension: [Nuclear Mode][state i][state j]
		self.H_diab = []
		self.ExcitonCoef=[] # These Diagonalize H + H_Reorg. 
		self.ExcitonEnergies=[] # Corresponding Eigenvalues. 
		# Inputfiles and whatnot ----------------------------------
		self.InputVibronic = MolName+str(".asa.inp")
		self.InputQchem = MolName+str(".qchem.out")
		self.InputHessian = MolName+str(".hess")
		self.MyGeometry = AGeometry()
		self.fd = None
		
		self.ReadOrcaParams() # reads the dimensionless displacement
		self.ReadQchemParams()
		self.ReadHessian() # reads cartesian modes and frequencies from an orca.hess file. 
		self.IfFiniteDifferences = IfFiniteDifferences
		self.IfAdiabatic = IfAdiabatic
		
		# this is a hack which Silences off-diagonal reorganization energies. 
		# in MakeVertical. 
		self.OnlyDiagReorg = True
		# False silences those elements in the reorganization Hamiltonian as well. 
		self.OffDiagonalRelaxation = False
		
		if (IfFiniteDifferences): 
			print "Collecting Finite Difference data."
			inpf = InputFiles(MolName)
			self.fd = FiniteDifferences(inpf,self)
			if (not IfInteractive): 
				# note we can collect just adiabatic energies from this routine too... 	
				self.fd.Collect(IfAdiabatic)
				print "Calculating Dimensionless displacements"
				self.FD_Dev = self.DimensionlessDisplacements(self.fd.Derivatives)
				if (not IfAdiabatic): 
					if ( not self.OffDiagonalRelaxation): 
						for i in range(len(self.AllVEn)): 
							ZeroNonDiagonal(self.FD_Dev[i])
			#		print "Printing Dimensionless Displacements: "
			#		for i in range(len(self.AllVEn)):
			#			print self.AllVEn[i]
			#			print self.FD_Dev[i]
					self.H_diab = self.fd.Zero()
					print "H_diab:"
					print self.H_diab

		if (not IfInteractive): 			
			if (not IfFiniteDifferences): 
				self.CheckConsis()
				self.MakeVertical()
			else: 
				self.MakeExciton()

		print " - - - - - - - VIBRONIC DATA COLLECTION COMPLETE - - - - - - - "
		return
		
	def Center(self):
		"""Average energy weighted by strength"""
		tore = 0.0
		strens = 0.0
		for ei in range(1,self.nexsts+1): 
			tore += self.VertOmegaij[0][ei]*Norm(self.TMom[0][ei])
			strens += Norm(self.TMom[0][ei])
		return tore/strens	
	def SystemTimescales(self):
		"""List of system timescales (ps)"""
		print "System Timescales (ps) "
		for i in range(self.nexsts+1) : 
			for j in range(1,self.nexsts+1) : 
				if i<j : 
					print "i,j:",(i,j)," T: ",PsPerAu*(2.0*pi/(abs(self.VertOmegaij[0][i]-self.VertOmegaij[0][j])))," ps "," f ", Norm(self.TMom[i][j]), " Angle: ", VectorCosine(self.TMom[0][i],self.TMom[0][j]) 
	def Gap(self):
		"""Returns gap between two brightest states"""
		inds = [ ei for ei in range(self.nexsts)] 
		ost = [Norm(self.TMom[0][ei+1]) for ei in range(self.nexsts)]
		bst = nlargest(2,inds,key=lambda i : ost[i])
		return abs((self.VertOmegaij[0][bst[0]+1])-(self.VertOmegaij[0][bst[1]+1]))
	def HAdiaVert(self):
		tore = [[0.0 for ei in range(self.nexsts+1)]  for ej in range(self.nexsts+1)]
		for ei in range(self.nexsts+1): 
			tore[ei][ei] = self.VertOmegaij[0][ei]
		return tore		
	def CheckConsis(self):
		print BreakLine
		print "Checking Consistency QCHEM vs ORCA" 
		for e in range(len(self.QEEn)):
			print "Energies: ",self.QEEn[e], " : ", self.EEn[e]
			print "Moms: ", self.QTMom[e]," : ", self.OTMom[e]
		print BreakLine
		# using the energies of qchem not orca. 
		UseOrca = True 
		if (not UseOrca): 
			self.EEn = self.QEEn
		return 
	def ReadQchemParams(self):
		InputFile = open(self.InputQchem,'r')
		Ne=0
		Nv=0
		Tenen=[]
		Tenem=[]
		Tene0=[]
		UseOldMultipole = True
		if (UseOldMultipole) :
			print "-----------------------------------------------"
			print "Using the older of qchem's two multipole codes."
			print "-----------------------------------------------"			
		print "Opened File ",self.InputQchem
		s = InputFile.readline()
		while s: 
			if (s.count('Excited state') > 0 ):
				self.QEEn.append(float(s.split()[-1])/EvPerAu)
			elif (s.count('Trans. Mom.') > 0):
				self.QTMom.append([float(s.split()[2]),float(s.split()[4]),float(s.split()[6])])
			elif (s.count('Electron Dipole Moments of Ground State') > 0):
				InputFile.readline()
				InputFile.readline()
				InputFile.readline()
				g = InputFile.readline()
				while (g.count('.')>0):
					Tenen.append(map(float,g.split()[1:4]))
					s = g = InputFile.readline()
			elif (s.count('Dipole Moments of Excited State') > 0 or s.count('Dipole Moments of Singlet Excited') > 0):
				InputFile.readline()
				InputFile.readline()
				InputFile.readline()
				g = InputFile.readline()
				while (g.count('.')>0):
					try: 
						Tenen.append(map(float,g.split()[1:4]))
					except Exception as exc : 
						print "Failed to read line: ", g 
						raise exc
					s = g = InputFile.readline()
			elif (s.count('Cartesian Multipole Moments') > 0 and UseOldMultipole):
				InputFile.readline()
				InputFile.readline()
				InputFile.readline()
				InputFile.readline()
				# the older moment code is in debye. 
				g = InputFile.readline()
				mom = [float(g.split()[1])*AuPerDebye,float(g.split()[3])*AuPerDebye,float(g.split()[5])*AuPerDebye]
				Tenen[0] = mom
			elif (s.count('TDA Excited-State Multipoles, State') > 0 and UseOldMultipole):
				n = s.split()[-1]
				InputFile.readline()
				InputFile.readline()
				InputFile.readline()
				InputFile.readline()
				# the older moment code is in debye. 
				g = InputFile.readline()
				mom = [float(g.split()[1])*AuPerDebye,float(g.split()[3])*AuPerDebye,float(g.split()[5])*AuPerDebye]
				Tenen[int(n)] = mom
			elif (s.count('Transition Moments Between Ground and ') > 0 or s.count('Moments Between Ground and Singlet') > 0):
				InputFile.readline()
				InputFile.readline()
				InputFile.readline()
				g = InputFile.readline()
				while (g.count('.')>0):
					Tene0.append(map(float,g.split()[2:-1]))
					s = g = InputFile.readline()
			elif (s.count('Transition Moments Between Excited States') > 0 or s.count('Transition Moments Between Singlet') > 0):
				InputFile.readline()
				InputFile.readline()
				InputFile.readline()
				g = InputFile.readline()
				while (g.count('.')>0):
					Tenem.append(map(float,g.split()[:-1]))
					s = g = InputFile.readline()
			s = InputFile.readline()
		InputFile.close()
		# build the dipole matrix. 
		Ne = max(self.nexsts,len(self.QEEn))
		print "Ne: " , Ne
		#print "Tenen ", Tenen
		self.TMom = [ [ [0.0 for k in range(3) ] for j in range(Ne+1)]  for i in range(Ne+1)]
		for n in range(Ne+1):
			self.TMom[n][n] = Tenen[n]
			if (n < Ne):
				self.TMom[0][n+1] = Tene0[n]
				self.TMom[n+1][0] = Tene0[n]			
		for l in Tenem :
			n = int(l.pop(0))
			m = int(l.pop(0))
			# if these are fucked up it can be really bad for a simulation. 
			# for some reason the MDAB moments between states are assinine in some cases. 
			if (Norm(l) > 1.0): 
				Scl = 0.1/Norm(l)
				for x in l: 
					x *= Scl
			self.TMom[n][m] = l 
			self.TMom[m][n] = l 			
		print "QC Read Complete."
		# if any transition moment has norm greater than 3,  silence that state. 				
		for i in range(len(self.TMom)): 
			if (Norm(self.TMom[0][i]) > 3.0): 
				print "Erroneous oscillator strength (State) ", i, ", Silencing it."
				for m in range(len(self.TMom)): 
					self.TMom[i][m] = [0.0,0.0,0.0]
					self.TMom[m][i] = [0.0,0.0,0.0]					
		print "Dipole Strength Matrix: "
		for x in self.TMom: 
			print map(Norm,x)
		#print "Ne:", len(self.QEEn), " ", self.QEEn
		#print "TMom:", self.TMom
		print "Coherence Moments: "
		print "1,2:", self.TMom[1][2]
		print "3,4:", self.TMom[3][4]		
		return 
	def ReadOrcaParams(self):
		InputFile = open(self.InputVibronic,'r')
		Ne=0
		Nv=0
		print "Opened File ",self.InputVibronic
		s = InputFile.readline()
		while s: 
			if (s.count('el_states') > 0 ):
				Ne = int(InputFile.readline())
				for i in range(Ne):
					l = (InputFile.readline()).split()
					if (len(l) < 1): 
						print "Readline:Fail"
						break					
					self.EEn.append(float(l[1])*4.5563*math.pow(10.0,-6.0)) 
					self.OEEn.append(float(l[1])*4.5563*math.pow(10.0,-6.0)) 					
					self.OTMom.append(map(float,l[4:]))
			elif (s.count('vib_freq_gs') > 0 ):
				Nv = int(InputFile.readline())
				for i in range(Nv):
					l = (InputFile.readline()).split()
					if (len(l) < 1): 
						print "Readline:Fail"
						break
					self.VEn.append(float(l[1])*4.5563*math.pow(10.0,-6.0))
			# This is the tedious bit. 
			elif (s.count('sdnc') > 0 ):
				InputFile.readline()
				InputFile.readline()
				for i in range(Nv):
					l = (InputFile.readline()).split()
					if (len(l) < 1): 
						print "Readline:Fail"
						break
					l.pop(0)
					self.Dev.append(map(float,l))
			s = InputFile.readline()
		InputFile.close()	
		print "Read Complete."
		print "Nele:", self.nexsts, " ", self.EEn
		print "Vib:", len(self.VEn) #," ", self.VEn
		#print "Dev:", self.Dev
		# print the large dEV.
		print ""
		return
#	obtain the harmonic modes from orca. 
#   units of bohr. 
	def ReadHessian(self):
		try : 
			InputFile = open(self.InputHessian,'r')
		except: 
			print "hess File Open failed. " 
		TempModes = []# nuc x mode. to be transposed. 
		Atoms = []
		coords = []
		Masses=[]
		print "Opened File ",self.InputHessian
		Lines = InputFile.readlines()
		it = iter(Lines)
		while True: 
			try: 
				s = it.next()
				if (s.count('$hessian') > 0 ):
					Nv = int((it.next()).split()[0])
					for i in range(Nv):
						XX = map(int,(it.next()).split()) 
						TempModes = []
						for j in range(Nv):
							tmp=(it.next()).split()
							k=int(tmp.pop(0))
							if (k != j): 
								print "Read suspicious... ",j,k,type(j),type(k)
							if (len(tmp) != 6):
								print "Wrong length of hessian.", len(tmp)
							TempModes.append(map(float,tmp))		
						TA = numpy.array(TempModes)
						self.CartesianHessian.extend(map(list,list(TA.transpose())))
						if (XX[-1]+1 == Nv):
							break
					print "Read Hessian "
				elif (s.count('normal_modes') > 0 ):
					Nv = int((it.next()).split()[0])
					for i in range(Nv):
						XX = map(int,(it.next()).split()) 
						TempModes = []
						for j in range(Nv):
							tmp=(it.next()).split()
							k=int(tmp.pop(0))
							if (k != j): 
								print "Read suspicious... ",j,k,type(j),type(k)
							if (len(tmp) != 6):
								print "Wrong length of hessian.", len(tmp)
							TempModes.append(map(float,tmp))		
						if (i != 0) :
							TA = numpy.array(TempModes)
							self.CartesianModes.extend(map(list,list(TA.transpose())))
						if (XX[-1]+1 == Nv):
							break
					print "Read, ",len(TempModes), "Modes"
					print "Some Norms: "
					for i in range(2): 
						print "i: ", i, " : ", Norm(self.CartesianModes[i])
				elif (s.count('$atoms') > 0 ):
					Na = int((it.next()))
					for i in range(Na): 
						tmp=(it.next()).split()
						Atoms.append(tmp.pop(0))
						Masses.append(float(tmp.pop(0)))
						coords.append(map(float,tmp))
				elif (s.count('ir_spectrum')>0) : 
					Nv = int((it.next()))
					for j in range(Nv): 
						tmp=(it.next()).split()
						if (float(tmp[0]) > 0.0): 
							self.AllVEn.append(float(tmp[0])*AuPerWavenumber)
			except StopIteration: 
				break 
		#transpose the modes. 
		TA2 = numpy.array(coords)
		TA2 /=BohrPerAngs
		coords = map(list,list(TA2))
		self.MyGeometry = AGeometry(Atoms,coords,Masses)
		InputFile.close()
		print "Read Complete."
		return
	def GetCartDisplaced(self,AtomNumber,Direction,Strength) : 
		Na = len(self.MyGeometry.MyAtoms)
		disp = numpy.array([[0.0 for j in range(3)] for i in range(Na)])
		disp[AtomNumber][Direction] += Strength
		return self.MyGeometry.DisplacedAsString(disp)

	#	Arguement is a Mass-Weighted Cartesian gradient (Eh/Bohr)
	def DimensionlessDisplacements(self,ACartesianGradient):
		# now dE/dQ = (dE/dXi) m_i^-1/2 U_il w_l^-1/2
		print "Projecting onto ",(len(self.AllVEn)), " Modes" #,self.AllVEn
		ts = list(ACartesianGradient[0].shape)
		ts.insert(0,len(self.AllVEn))
		tore = numpy.zeros(tuple(ts))		
		grad = numpy.array(ACartesianGradient)
		modes = numpy.array(self.CartesianModes)
		Na = len(self.MyGeometry.MyAtoms)
		for Mode in range(len(self.CartesianModes)): 
			for i in range(Na):
				for di in range(3):
					tore[Mode] += grad[i*3+di]*(modes[Mode][i*3+di])/(sqrt(self.MyGeometry.MyMasses[i])*sqrt(self.AllVEn[Mode]))
		# find most displaced mode for each state grad is a matrix or a vector. ????
		# since they come from finite differences, make sure they are reasonable. ? 
			if (tore[Mode].max() > 2): 
				print "Warning, very large Dimensionless displacement ",tore[Mode].max()," element: ", (tore[Mode].argmax()-tore[Mode].argmax()%tore[Mode].shape[0])/tore[Mode].shape[0],tore[Mode].argmax()%tore[Mode].shape[0] , "on Mode: ", Mode, " of Eigenvalue: ", self.AllVEn[Mode]
				# print tore[Mode]
				#self.fd.DebugMode()
		return tore	
		
	def VibrationCoordinates(self,ModeNumber,Atom=False) : 
		#returns a raw lists of atom positions along a mode. 
		Na = len(self.MyGeometry.MyAtoms)
		mind=-2.0
		maxd=2.0
		npts=13 
		ds = numpy.linspace(mind,maxd,npts)
		c0 = numpy.array([[0.0 for j in range(3)] for i in range(Na)])
		disp = numpy.array([[0.0 for j in range(3)] for i in range(Na)])
		cnew = numpy.array([[0.0 for j in range(3)] for i in range(Na)])
		c0 = numpy.array(self.MyGeometry.MyCoords)
		NOfThisAtom = Na 
		if (Atom != False): 
			NOfThisAtom = self.MyGeometry.MyAtoms.count(Atom)
		xs = numpy.array([0.0 for i in range(NOfThisAtom*npts)])
		ys = numpy.array([0.0 for i in range(NOfThisAtom*npts)])
		zs = numpy.array([0.0 for i in range(NOfThisAtom*npts)])
		di=0
		for d in ds: 
			for i in range(Na): 
				if (ModeNumber >= len(self.CartesianModes)): 
					print "mode out of range", ModeNumber
				for j in range(3): 			
					disp[i][j] = (self.CartesianModes[ModeNumber][i*3+j]/sqrt(self.MyGeometry.MyMasses[i]))
			disp *= d 
			cnew = numpy.add(c0, disp).copy()
			ii = 0 
			for i in range(Na):
				if (self.MyGeometry.MyAtoms[i] == Atom): 
					xs[NOfThisAtom*di+ii] = cnew[i][0]
					ys[NOfThisAtom*di+ii] = cnew[i][1]
					zs[NOfThisAtom*di+ii] = cnew[i][2]
					ii+=1
			di += 1
		return (xs,ys,zs)
		
	def GetDisplaced(self,ModeNumber,Strength) : 
		# note relation between Q an R, Q = sqrt(\mu \omega/\hbar)^(1/2) R 
		# but these are cartesian displacements. 
		Na = len(self.MyGeometry.MyAtoms)
		disp = numpy.array([[0.0 for j in range(3)] for i in range(Na)])
		for i in range(Na): 
			for j in range(3): 
				if (ModeNumber >= len(self.CartesianModes)): 
					print "mode out of range", ModeNumber
				elif (i*3+j > len(self.CartesianModes[ModeNumber])):
					print "Dex out of range", len(self.CartesianModes[ModeNumber]), " ", i*3+j 
				disp[i][j] = (self.CartesianModes[ModeNumber][i*3+j]/sqrt(self.MyGeometry.MyMasses[i]))
		return self.MyGeometry.DisplacedAsString(Strength*disp)
		
	def Mute(self,ToMute):
		dim = len(self.TMom)
		rdim = range(dim)
		for x in rdim:
			for y in rdim: 
				if (ToMute.count(x) + ToMute.count(y) > 0 ): 
					self.TMom[x][y] = [0.0 for z in range(3)]
	def Rotate(self,ARMatrix):
		for x in self.TMom:
			for y in x: 
				y = numpy.dot(ARMatrix,y)
	def MuMatrix(self,CarDir):
		"""Provides multipole for a given density"""
		if (len(self.TMom) < 1):
			print "can't Multipole not enough data"
			return
		if (2 < CarDir < 0):
			print "arguement length mismatch MuMatrix"
			return
		dim = range(self.nexsts+1)
		# print dim, len(self.TMom), len(self.TMom[0]), len(self.TMom[0][0])
		# Transform the dipole matrix into exciton basis: 
		if (self.IfFiniteDifferences): 
			tm = numpy.array(self.TMom)
			bas = numpy.array(self.ExcitonCoef)
			tore = numpy.zeros((self.nexsts+1,self.nexsts+1))
			for i in dim: 
				for j in dim: 
					for m in dim: 
						for n in dim: 
							tore[i][j] += self.TMom[m][n][CarDir]*bas[n][j]*bas[m][i]
			return tore.tolist()
		else : 
			return [[ self.TMom[x][y][CarDir] for y in dim] for x in dim]
	def Multipole(self,Dens):
		"""Provides multipole for a given density"""
		if (len(self.TMom) < 1):
			print "can't Multipole not enough data"
			return
		if (len(Dens[1]) != (self.nexsts+1)):
			print "arguement length mismatch Multipole"
			return
		dim = range(self.nexsts+1)
		ToRe = [0.0,0.0,0.0]
		for x in dim: 
			for y in dim : 
				ToRe = map(sum, izip(ToRe,[Dens[x][y]*Z for Z in self.TMom[x][y]]))
		return ToRe
	# -1 is the gs in Uw_IMDHO
	# 0 is the gs in VertOmegaij
	# this gives the diagonal bit of the relaxation operator. 
	def MakeCMNmmnn(self):
		dim = range(-1,self.nexsts)
#		C = lambda I,J: self.Uw_IMDHO(self.VertOmegaij[I+1][J+1],I,J)
		C = lambda I,J,w : self.Uw_IMDHO(w,I,J)
		return [[[[ C(m,n,(self.VertOmegaij[nn+1][mm+1])) for m in dim] for n in dim] for mm in dim] for nn in dim]
	def MakeGamma(self):
		dimp = range(self.nexsts+1)
		bas = numpy.zeros([self.nexsts+1,self.nexsts+1])
		if (self.IfFiniteDifferences):
			bas = self.ExcitonCoef
		else :
			bas = numpy.identity(self.nexsts+1)			
		CMN = self.MakeCMNmmnn() 
#		print "Cmn:", CMN
#		print "Inverse Lifetimes:"
#		for x in dimp:
#			for y in dimp: 
#				if (abs((CMN[x][y])) > 0.0): 
#					print "(i,j): ",(x,y)," ", (SecPerAu*(2*pi)/(CMN[x][y]))/(PsinSec), " self.VertOmegaij[I][J]: ",self.VertOmegaij[x][y]
		# self.ExcitonCoef is stateXExciton
		# and m is an exciton coefficient, M and N are state indices. 	
		MNsum = lambda m,n,mm,nn: sum([sum([(bas.transpose()[m][M])*bas[M][n]*(bas.transpose()[mm][N])*bas[N][nn]*CMN[M][N][mm][nn] for M in dimp]) for N in dimp])
		Gamma = [[[[ MNsum(m,n,mm,nn) for nn in dimp] for mm in dimp] for n in dimp] for m in dimp]
		print "Gamma[3,4,3,4]", Gamma[3][4][3][4]
		print "Gamma[3,3,4,4]", Gamma[3][3][4][4]
		#print "Made Gamma Operator: "
		if False: 
			for x in dimp:
				for y in dimp: 
					for z in dimp: 
						for q in dimp: 
							if abs(Gamma[x][y][z][q]) > pow(10.0,-10):
								print "Ind: ",[x,y,z,q], " Val ",abs(Gamma[x][y][z][q])
		return Gamma
	def GammaSum(self,Gamma,m,n,mm,nn):
		dim = range(self.nexsts+1)
		GS = Gamma[nn][n][m][mm]+Conj(Gamma[mm][m][n][nn])
		for k in dim: 
			GS -= (Delta(n,nn)*Gamma[m][k][k][mm]+Delta(m,mm)*Conj(Gamma[n][k][k][nn]))
		return GS
	def DissectRedfield(self,mm,nn,m,n): 
		print "Dissecting Redfield operator in excitonic basis, R:",mm,nn,m,n
				
	def MakeRedfield(self):	
		dim = range(self.nexsts+1)
		Gamma = self.MakeGamma()
		R = [[[[ self.GammaSum(Gamma,m,n,mm,nn) for nn in dim] for mm in dim] for n in dim] for m in dim]
		dimp = range(self.nexsts+1)
		print "Made Redfield Operator "
#		for x in dimp:
#			if (R[x][x][x][x] != 0.0) : 
#				print x," - ",x," : ", (SecPerAu*(2*pi)/(R[x][x][x][x]))/(PsinSec)," ps ", (SecPerAu*(2*pi)/(R[x][x][x][x]))/(NsinSec), " ns "
		print "Relaxation Lifetimes (ps) :"
		for y in dimp:
			for x in dimp:
				if (R[x][x][y][y] != 0.0) : 
					print x," - ",y," : ", (SecPerAu*(2*pi)/(R[x][x][y][y]))/(PsinSec)," ps ", (SecPerAu*(2*pi)/(R[x][x][y][y]))/(NsinSec), " ns "
		print "1-2 Coherence Lifetimes (ps) :", (SecPerAu/(R[1][2][1][2]))/(PsinSec), " Energy: ", (R[1][2][1][2])
		print "3-4 Coherence Lifetimes (ps) :", (SecPerAu/(R[3][4][3][4]))/(PsinSec), " Energy: ", (R[3][4][3][4])
		print "5-6 Coherence Lifetimes (ps) :", (SecPerAu/(R[5][6][5][6]))/(PsinSec), " Energy: ", (R[5][6][5][6])
		#print "50 wavenumber in en", 50.0*AuPerWavenumber ," in ps :", SecPerAu*(1/(50.0*AuPerWavenumber))/(PsinSec)
		#print "1.2ps in au. ", 1/((1.2*PsinSec)/(SecPerAu))
		
		print "VERTICAL TIMESCALES----------"
		moe = abs(self.QEEn[2]-self.QEEn[3])
		print "Ex. state: 3-4 Oscillation Energy: ", moe, " and in cm^-1: ", moe/AuPerWavenumber
		print "3-4 Oscillation Timescale naked: ", SecPerAu*(1/moe)/(PsinSec)
		print "3-4 Oscillation Timescale with relax: ", PsPerAu*(1/abs(abs(moe)-abs(R[3][4][3][4].real)))
		moe = abs(self.QEEn[0]-self.QEEn[1])
		print "2-1 Oscillation Energy: ", moe, " and in cm^-1: ", moe/AuPerWavenumber
		print "2-1 Oscillation Timescale naked: ", SecPerAu*(1/sqrt(pow(moe,2.0)))/(PsinSec)
		print "2-1 Oscillation Timescale with relax: ", PsPerAu*(1/abs(abs(moe)-abs(R[1][2][1][2].real)))
		print "ADIABATIC TIMESCALES----------"		
		moe = abs(self.VertOmegaij[3][4])
		print "Ex. state: 3-4 Oscillation Energy: ", moe, " and in cm^-1: ", moe/AuPerWavenumber
		print "3-4 Oscillation Timescale naked: ", SecPerAu*(1/sqrt(pow(moe,2.0)))/(PsinSec)
		print "3-4 Oscillation Timescale with relax: ", PsPerAu*(1/abs(abs(moe)-abs(R[3][4][3][4].real)))
		moe = abs(self.VertOmegaij[1][2])
		print "2-1 Oscillation Energy: ", moe, " and in cm^-1: ", moe/AuPerWavenumber
		print "2-1 Oscillation Timescale naked: ", SecPerAu*(1/sqrt(pow(moe,2.0)))/(PsinSec)
		print "2-1 Oscillation Timescale with relax: ", PsPerAu*(1/abs(abs(moe)-abs(R[1][2][1][2].real)))
		return R 
		
	#	This is only used if FiniteDifferences is True. 
	#	Else the routine below is called. 
	def MakeExciton(self):
		print "Making H+H_Reorg: "
		print "Diabatic Hamiltonian: ", self.H_diab
		w,v = numpy.linalg.eig(self.H_diab)
		print "Adiabatic Energies: ", w
		print "Adiabatic Energies (Sanity Check from qchem): ", self.QEEn
		H_reorg = numpy.zeros(self.H_diab.shape)
		if (self.OnlyDiagReorg == False) : 
			print "WARNING!!!!"
			print "Off Diagonal Reorganization energies for finite differences not yet implemented."
			print "WARNING!!!!"
		print "Reorganization Hamiltonian: "
		for i in range(len(self.AllVEn)): 
			tmp = (self.FD_Dev[i]*self.FD_Dev[i]).copy()
			ZeroNonDiagonal(tmp)
			H_reorg += (self.AllVEn[i]/2.0)*tmp
		print H_reorg
		InsertZeroRC(H_reorg)
		InsertZeroRC(self.H_diab)
		H_t = self.H_diab.copy()
		H_t += H_reorg
		print "H_t", H_t
		w,v = numpy.linalg.eig(H_t)
		SortEigsAscending(w,v)
		print v
		print "Exciton energies: ", w
		print "Shape of v", v.shape
		for i in range(v.shape[1]): 
			print "Exciton State ", i, v[:,i]
		self.ExcitonEnergies = w.copy()
		# self.ExcitonCoef is a (state)x(eigenvector) matrix
		self.ExcitonCoef = v.copy()
		dim = range(len(self.ExcitonEnergies))
		self.VertOmegaij = [[self.ExcitonEnergies[i] - self.ExcitonEnergies[j] for i in dim] for j in dim] 		
		return
		
	# The ground state is the first r,c of this matrix		
	# this is only called if FiniteDifferences = False
	# else the routine above is called. 
	def MakeVertical(self):
		dim = range(self.nexsts+1)
		EnInclGS = lambda i: self.EEn[i-1] if i > 0 else 0.0
		VertEn = [EnInclGS(i)-EnInclGS(0)-self.Reorg(i-1,i-1)  for i in dim]
		#self.VertOmegaij = [[ AdiaOmega(i,j)-self.Reorg(i-1,j-1) for i in dim] for j in dim] This made no fucking sense. 6/14/2011
		self.VertOmegaij = [[VertEn[i] - VertEn[j] for i in dim] for j in dim] 
		print BreakLine
		print "Made Vertical Energies: ", self.VertOmegaij
		print BreakLine
		return
				
# 
# Useful Shortcut functions:
# Ground State is present and has energy zero. 
#		

	def beta(self):
		return (1/Kb*self.Temp)
	#ground state is an arguement of -1 to these routines. 
	def Dijk(self,i,j,k):
		""" Return Dimensionless disp. between emoded i,j and v mode k """
		if (self.IfFiniteDifferences) : 
			if (i < 0) : 
				if (j < 0) : 
					return 0 
				else: 
					return (-self.FD_Dev[k][j+1][j+1])	
			elif (j < 0) : 
				return self.FD_Dev[k][i+1][i+1]
			else: 
				return (self.FD_Dev[k][i+1][i+1] - self.FD_Dev[k][j+1][j+1])
		else :
			if (i < 0) : 
				if (j < 0) : 
					return 0 
				else: 
					return (-self.Dev[k][j])	
			elif (j < 0) : 
				return self.Dev[k][i]
			else: 
				return (self.Dev[k][i] - self.Dev[k][j])
	def Sijk(self,i,j,k):
		""" Return Huang-Rhys Factor """
		return ((1.0/2.0)*pow(self.Dijk(i,j,k),2.0))
	def Reorg(self,ei,ej):
		""" Return Reorganization energy """
		return sum([self.Sijk(ei,ej,k)*self.VEn[k] for k in range(len(self.VEn))])		
	def Nbe(self,Omega):
		return 1/(exp(Omega*self.beta())-1)
	#
	# TODO: 
	# Add? Meier and tannor treatment with matsubara frequencies? jcp 111 3365
	# c.f. Mukamel 8.67b,c
	#

	# This formula is given in Mukamel 8.38a,b
	def Uw_IMDHO(self,Omega, ei, ej):	
		Homogeneous = self.homo
		RealPart = 0.0
		ImagPart = 0.0
		if (self.IfFiniteDifferences) : 
			for k in range(len(self.AllVEn)) :
				omegak = self.AllVEn[k]
				RealPart += self.Sijk(ei,ej,k)*pow(omegak,2)*coth(self.beta()*omegak/2)*(Lorentzian(Omega-omegak,Homogeneous)+Lorentzian(Omega+omegak,Homogeneous))
				ImagPart += self.Sijk(ei,ej,k)*pow(omegak,2)*(Lorentzian(Omega-omegak,Homogeneous)-Lorentzian(Omega+omegak,Homogeneous))
		else : 
			for k in range(len(self.VEn)) :
				omegak = self.VEn[k]
				RealPart += self.Sijk(ei,ej,k)*pow(omegak,2)*coth(self.beta()*omegak/2)*(Lorentzian(Omega-omegak,Homogeneous)+Lorentzian(Omega+omegak,Homogeneous))
				ImagPart += self.Sijk(ei,ej,k)*pow(omegak,2)*(Lorentzian(Omega-omegak,Homogeneous)-Lorentzian(Omega+omegak,Homogeneous))
		return complex(RealPart,ImagPart)

#		AbsOmega = abs(Omega)
#		for k in range(len(self.VEn)) :
#			omegak = self.VEn[k]
#			RealPart += self.Sijk(ei,ej,k)*pow(omegak,2)*coth(self.beta()*omegak/2)*(Lorentzian(AbsOmega-omegak,Homogeneous)+Lorentzian(AbsOmega+omegak,Homogeneous))
#			ImagPart += self.Sijk(ei,ej,k)*pow(omegak,2)*(Lorentzian(AbsOmega-omegak,Homogeneous)-Lorentzian(AbsOmega+omegak,Homogeneous))
#		if (Omega < 0): 
#			return exp((-1.0)*self.beta()*AbsOmega)*complex(RealPart,ImagPart)
#		else : 
#			return complex(RealPart,ImagPart)

	def PlotSpectralDensity(self,nam): 	
		xstep=.001/EvPerAu
		fs = numpy.arange(-1.0/EvPerAu,1.0/EvPerAu,xstep)
		# -1 is the gs for the routine below: 
		y1s = [ self.Uw_IMDHO(x,1,3).real for x in fs]
		y2s = [ self.Uw_IMDHO(x,2,3).real for x in fs]
		lines = plt.plot(fs,y1s,'k--',fs,y2s,'k:')
		l1,l2= lines 
		plt.setp(l1,linewidth=2, color='r')
		plt.setp(l2,linewidth=1, color='b')
		plt.legend(['Re (1,3) ','Re (2,3)'],'upper left', shadow=True)
		xmax = 0.4/EvPerAu
		plt.xlim(0.0,xmax)
		NMax = int((1.0+xmax)/xstep)
		plt.ylim(0.0,max(y1s[:NMax]))
		plt.xlabel('Frequency ($E_h$)')
		plt.ylabel('Bath Correlation Function ($E_h$)')
		plt.title(self.MolName+' $C_{mn}(\omega)$')
		plt.savefig("./figures/"+'Spectral Densities')
		plt.clf()
		plt.close()
		if (False):
			#fs = numpy.arange(0.01/EvPerAu,1.0/EvPerAu,.001/EvPerAu)
			fs = numpy.arange(-1.0/EvPerAu,1.0/EvPerAu,.001/EvPerAu)
			# -1 is the gs for the routine below: 
			y1s = [ self.Uw_IMDHO(x,-1,2).imag for x in fs]
			y2s = [ self.Uw_IMDHO(x,-1,3).imag for x in fs]
			y3s = [ self.Uw_IMDHO(x,2,3).imag for x in fs]
			plt.plot(fs,y1s,'k--',fs,y2s,'k:',fs,y3s,'k')
			plt.legend(['Im (0,3) ','Im (0,4)','Im (3,4)'])
			print "v23 ", self.VertOmegaij[3][4]
			#draws a vertical blue line
			#plt.axvline(x=0.002, linewidth=1, color='b')
			plt.annotate('$\omega_{34}$ ='+str(round(self.VertOmegaij[3][4],4)), xy=(self.VertOmegaij[3][4], 0), xytext=(self.VertOmegaij[3][4]-0.001, 0.001), arrowprops=dict(facecolor='red'))
			plt.xlim(-0.017,0.017)
			plt.xlabel('Frequency(au)')
			plt.ylabel('Strength')
			plt.title(nam+' Spectral Density')
			plt.savefig("./figures/"+' Spectral Densities')
			plt.clf()
			plt.close()
			# Zoom in on the 2-3 region
			fs = numpy.arange(self.VertOmegaij[3][4]-0.010/EvPerAu,self.VertOmegaij[3][4]+0.010/EvPerAu,.001/EvPerAu)
			y3s = [ self.Uw_IMDHO(x,2,3).imag for x in fs]
			plt.plot(fs,y3s,'k')
			plt.legend(['Im (3,4)'])
			for Veig in self.VEn:
				if fs[0] < Veig < fs[-1]: 
					plt.axvline(x=Veig, linewidth=1, color='b')
			plt.xlabel('Frequency(au)')
			plt.ylabel('Strength')
			plt.title(nam+' Zoomed Spectral Density')
			plt.savefig("./figures/"+'Zoom')
			plt.clf()
			plt.close()
		return
