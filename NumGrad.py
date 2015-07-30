# python routines to evaluate numerical derivatives along normal modes 
# using an electronic structure package. 
# JAP 2011

# scipy crap 
# Makes Figures without popup
# But works poorly on OSX. 
#matplotlib.use('Agg')
#from mpl_toolkits.mplot3d import Axes3D
import numpy, scipy
#import scipy.interpolate
#from scipy import special
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
output = sys.stdout

from TensorNumerics import * 

#
# Math utility functions 
#

EvPerAu = 27.2113
AuPerWavenumber = 4.5563*math.pow(10.0,-6.0)
WavenumberPerEv = 1/(EvPerAu*AuPerWavenumber)
SecPerAu = 2.418884326505*math.pow(10.0,-17.0)
NsinSec = pow(10.0,-9.0)
PsinSec = pow(10.0,-12.0)
FsinSec = pow(10.0,-15.0)
AuPerDebye = .393430307
PsPerAu = SecPerAu/PsinSec
FsPerAu = SecPerAu/FsinSec
RadPerDegree = 2.0*pi/360.0
AuInVoltPerMeter = 5.1421*pow(10.0,11.0)
BohrPerAngs = (1.0/0.529177249)
Kb = 8.61734315*pow(10.0,-5.0)/EvPerAu #Boltzmann Const. 
AvNumber = 6.022*pow(10.0,23)
AtomicMassInKg = 9.1093826*pow(10.0,-31)
AmuInAu = AtomicMassInKg/AvNumber
Imi = complex(0.0,1.0)

def coth(z):
	if (not (type(z) is FloatType )):
		print type(z)
		return 0.0
	elif ( z > 10.0 ) : 
		return 1.0
	elif ( z < 10.0 ) : 
		return -1.0
	else: 
		return ((exp(2.0*z)+1.0)/(exp(2.0*z)-1.0))
def Dot(l1,l2) : 
	if (len(l1) != len(l2)):
		print "Dim Mismatch, dot"
	else : 
		return sum(x*y for (x,y) in izip(l1,l2))
def Gaussian(x,Sig,x0 = 0.0): 
	return ((1/sqrt(2.0*pi*pow(Sig,2.0)))*exp((-1.0)*pow(x-x0,2.0)/(2*pow(Sig,2.0))))
def Lorentzian(x,Gam,x0 = 0.0): 
	return ((1/pi)*(0.5*Gam)/(pow(x-x0,2)+pow(Gam/2,2)))	
def Norm(arg):
	square = lambda X: X*X
	return sqrt(sum(map(square,arg)))
def Normalize(arg):
	Narg = Norm(arg)
	return [x/Narg for x in arg]
def RotationMatrix(a,b,theta,dim=2):
	tore = numpy.identity(dim)
	tore[a][a] = cos(theta)
	tore[a][b] = -sin(theta)
	tore[b][a] = sin(theta)
	tore[b][b] = cos(theta)
	return tore
def SortEigsAscending(val,vec): 
	changed = True
	tmpvl = 0.0
	tmpv = numpy.zeros(vec[:,0].shape)
	while changed: 
		changed = False 
		for i in range(len(val)): 
			for j in range(i,len(val)): 
				if val[i] > val[j] : 
					tmpv = vec[:,i].copy()
					vec[:,i] = vec[:,j]
					vec[:,j] = tmpv
					tmpvl = val[i]
					val[i] = val[j]
					val[j] = tmpvl
					changed = True
	return 
def VectorCosine(a,b): 
	if (Norm(a) == 0.0 or Norm(b) == 0.0):
		return 0.0
	return numpy.arccos(Dot(a,b)/(Norm(a)*Norm(b)))/RadPerDegree

def MakeSimplePlot(ys,xs=None,tit="NoName",xl="",yl="",xlimits=(0.0,0.0),ylimits=(0.0,0.0)):
	import matplotlib
	import matplotlib.pyplot as plt
	if (xs == None): 
		xs = numpy.arange(len(ys))
	lines = plt.plot(xs,ys,'k')
	l1 = lines 
	plt.setp(l1,linewidth=2, color='r')
	plt.xlabel(xl)
	plt.ylabel(yl)
	if (xlimits[0] != xlimits[1]) :
		plt.xlim(xlimits[0],xlimits[1])
	if (ylimits[0] != ylimits[1]) :
		plt.ylim(ylimits[0],ylimits[1])
	plt.title(tit)
	plt.savefig("./Figures"+Params.MoleculeName+"/"+tit)
	plt.clf()
	plt.close()
	return 
	
def _mkdir(newdir):
    """works the way a good mkdir should :)
        - already exists, silently complete
        - regular file in the way, raise an exception
        - parent directory(ies) does not exist, make them as well
    """
    if os.path.isdir(newdir):
        pass
    elif os.path.isfile(newdir):
        raise OSError("a file with the same name as the desired " \
                      "dir, '%s', already exists." % newdir)
    else:
        head, tail = os.path.split(newdir)
        if head and not os.path.isdir(head):
            _mkdir(head)
        #print "_mkdir %s" % repr(newdir)
        if tail:
            os.mkdir(newdir)

