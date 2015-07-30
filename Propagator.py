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
from LooseAdditions import * 
from TensorNumerics import * 
from Wick import * 
#from MatrixExponential import * 
from SpectralAnalysis import * 
from NumGrad import * 
	   
#
# This does the non-numerical work of propagation 
#

# Runge-Kutta
class Propagator: 
	def __init__(self,a_propagatable):
		print "Initalizing propagation."
		self.ShortTimePropagator = a_propagatable
		self.V0 = self.ShortTimePropagator.V0
		self.VNow = self.ShortTimePropagator.V0
		self.T0 = 0.0
		self.TNow = 0.0
		self.NSteps = int(Params.TMax/Params.TStep) + 1
		if (Params.DirectionSpecific): 
			self.Dipoles = numpy.zeros(shape=(self.NSteps,3),dtype = float)
		else: 
			self.Dipoles = numpy.zeros(shape=(self.NSteps),dtype = float)
		self.Norms = numpy.zeros(shape=(self.NSteps),dtype = float)		
		GoodMkdir("./Figures"+Params.MoleculeName)
		GoodMkdir("./Output"+Params.MoleculeName)		
		
	def Propagate(self): 
		Step=0
		while (self.TNow < Params.TMax): 
			# returns an estimate of the energy. 
			if(Params.RK45):
				self.RK45Step(self.VNow)
			else:
				self.RungeKuttaStep(self.VNow)
			self.Dipoles[Step] = self.ShortTimePropagator.DipoleMoment(self.VNow,self.TNow)
	#		self.ONSpectrum[Step] = self.ShortTimePropagator.ONSpectrum(self.VNow)
			print "T: ", self.TNow, " E: ", round(self.ShortTimePropagator.Energy(self.VNow),4), " Mu : ", self.Dipoles[Step], " |St| ", self.VNow.InnerProduct(self.VNow)
			if numpy.abs(self.Dipoles[Step]) > 4.0: 
				print "Error at step: ", Step 
			Step += 1
		import matplotlib.pyplot as plt
		import matplotlib.font_manager as fnt
		# Make plot styles visible. 
		PlotFont = {'fontname':'Helvetica','fontsize':18,'weight':'bold'}
		LegendFont = fnt.FontProperties(family='Helvetica',size='17',weight='bold')	
		Times = numpy.arange(len(self.Dipoles))*Params.TStep
		if(Params.DirectionSpecific): 
			lx,ly,lz = plt.plot(Times,self.Dipoles[:,0],'k',Times,self.Dipoles[:,0],'k--',Times,self.Dipoles[:,0],'k.')
			plt.setp(l1,linewidth=2, color='r')
			plt.setp(l1,linewidth=2, color='g')
			plt.setp(l1,linewidth=2, color='b')		
			plt.xlabel('Time (au)')
			plt.legend(['x','y','z'],loc=2)		
			plt.ylabel('Mu (au)')
			plt.savefig("./Figures"+Params.MoleculeName+"/"+'Dipole')
			plt.clf()
			Nrm = lambda X: (X[0]*X[0]+X[1]*X[1]+X[2]*X[2])
			DipoleStrength = numpy.vectorize(Nrm)
			SpectralAnalysis(DipoleStrength(self.Dipoles), Params.TStep, DesiredMaximum = 23.0/EvPerAu)

	def RungeKuttaStep(self,VNow): 
		V1 = self.ShortTimePropagator.Step(VNow)
		V2 = self.ShortTimePropagator.Step(VNow.LinearCombination(1.0,0.5*Params.TStep,V1))
		V3 = self.ShortTimePropagator.Step(VNow.LinearCombination(1.0,0.5*Params.TStep,V2))
		V4 = self.ShortTimePropagator.Step(VNow.LinearCombination(1.0,Params.TStep,V3))
		VNow.Add(V1,(1.0/6.0)*Params.TStep)
		VNow.Add(V2,(2.0/6.0)*Params.TStep)
		VNow.Add(V3,(2.0/6.0)*Params.TStep)
		VNow.Add(V4,(1.0/6.0)*Params.TStep)	
		self.TNow += Params.TStep		
		
