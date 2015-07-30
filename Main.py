# A toy-fermion propagator (algebra and numerics) 
# Begun by John Parkhill on Sept 29 2011
# uses external modules (no compilation)  
# scipy : (OSX-x64) http://stronginference.com/scipy-superpack/
# There should also be a script in the root directory to install that. 

# local modules written by Moi. 
from Propagatable_LCIS import *
from Propagatable_LCISD import * 
from Propagatable_AllPH import *
from Propagatable_Undressed import * 
from Propagatable_TCL2_Herm import * 
from Propagator import * 
from Propagator_TD import * # Propagator for Time-Dependent Hamiltonians. 
from TensorNumerics import * 
from Wick import * 
from BosonCorrelationFunction import * 
from LooseAdditions import * 
from SpectralAnalysis import * 

output = sys.stdout

usage = "usage: Just run." 
if len(sys.argv) < 0:
       print usage
	   
#
# Main Routine
#
class ToyManyBody: 
	""" Main MB Class """
	def __init__(self): 
		ToProp = None
		if (Params.AllDirections): 
			from multiprocessing import Process, Queue, Pipe
			print "Constructing Three Propagations..."	
			if False: #(Params.Correlated == False and Params.Undressed == True):	
				ToPropx = UndressedCIS(Polarization = "x") 
				ToPropy = UndressedCIS(Polarization = "y") 
				ToPropz = UndressedCIS(Polarization = "z") 				
			else: 
				ToPropx = Liouville_CISD(Polarization = "x") 
				ToPropy = Liouville_CISD(Polarization = "y") 
				ToPropz = Liouville_CISD(Polarization = "z") 
			xDynamics = Propagator_TD(ToPropx)  
			yDynamics = Propagator_TD(ToPropy)  
			zDynamics = Propagator_TD(ToPropz)  
			# Concurrent propogations for each direction
			PipeIn1,PipeOut1 = Pipe()
			PipeIn2,PipeOut2 = Pipe()
			PipeIn3,PipeOut3 = Pipe()
			print "Made output pipes..."					
			p1 = Process(target = xDynamics.Propagate, args = (PipeIn1,))
			p2 = Process(target = yDynamics.Propagate, args = (PipeIn2,))
			p3 = Process(target = zDynamics.Propagate, args = (PipeIn3,))
			print "Construction Completed Initializing propagation..."					
			print "Starting processes"		
			from time import sleep 
			p1.start()
			time.sleep(0.75)
			p2.start()
			time.sleep(0.75)
			p3.start()
			time.sleep(0.75)
			# Sometimes a process fails to start. This checks it. 
			Buh = PipeOut2.recv()
			Buh = PipeOut3.recv()
			Buh = PipeOut1.recv()
			print "Collecting Data..."
			Ty,MuY = PipeOut2.recv()
			Tz,MuZ = PipeOut3.recv()
			Tx,MuX = PipeOut1.recv()			
			p1.join()
			p2.join()
			p3.join()
			print "Data Collected, Generating proper isotropic spectrum."
			if (MuY.shape != MuX.shape != MuZ.shape):
				print "Hmm... Interpolate to the same times please."
				raise Exception("SameShpae")
			Mu = (1./3.)*(MuX+MuY+MuZ)
			SpectralAnalysis(Mu, Tx, DesiredMaximum = 26.0/EvPerAu, Title = "IsotropicSpectrum", Smoothing = True)
		else: 
			if (Params.Propagator == "phTDA2TCL"): 
				ToProp = Liouville_CISD() 
			elif (Params.Propagator == "AllPH"): 
				ToProp = AllPH()
			elif (Params.Propagator == "Whole2TCL"): 
				ToProp = TCL2_Herm()
			else: 
				raise Exception("Unknown ... ")
			TheDynamics = Propagator_TD(ToProp)  
			TheDynamics.Propagate()

ToyManyBody = ToyManyBody()
