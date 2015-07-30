
#Python Matrix Exponential Following ExpoKit Arnoldi Scheme
#JM
import scipy
from numpy import *
from scipy.linalg import norm, expm, expm2, expm3
from scipy.sparse import *

def expv(t, matVec, v, MatrixNormEst, m = 30, tol = 1.0e-7):
    """Find the action of a matrix exponential on a vector v
    for a time t, that is Exp(H t) * v

    Input Parameters:
    t --  Time to exponentiate for
    matVec -- A function which takes a vector v and returns a vector acted on by a matrix
    v -- The vector to be acted on
    MatrixNormEst -- Some estimate of the norm of the matrix being exponentiated
    tol -- The tolerance in the local error steps
    m -- the size of the Krylov Subspace which will be used to estimate the matrix
    
    Output: (w, FinalLocalError, hump)
    w -- Estimate of Exp(H t) * v
    FinalLocalError -- the accumlated local error estimates over all the steps
    hump -- The maximum norm attained for |Exp(H t) v|, which is one form of diagnostic
    """

    #Note: May want to generalize this stuff later
    v = mat(v) #Make it a Numpy Matrix
    n = max(shape(v)) #Get dimensions of v
    m = min(n, m) #Krylov Dimension, In Case n < m
    Beta = norm(v) #Norm of the Current vector of interest
    OriginalNormV = norm(v) #For error estimate at the end

    MaxRejects = 10
    btol = 1.0e-7
    gamma = 0.9
    delta = 1.2

    tLength = abs(t)
    tSign = sign(t)

    t_new = 0.0
    t_now = 0.0
    TotalLocalError = 0.0
    RoundOffErr = MatrixNormEst * finfo(float).eps
    
    #Two parameters for dealing with the "Happy Breakdown" Case
    #when our Arnoldi Process is near perfect for some dimension < m
    k1 = 2
    mb = m

    #Figure out the initial timestep data based on given norms
    InvM = 1.0/m
    fact = (((m+1) / exp(1.0))**(m+1)) * sqrt(2.0 * pi * (m+1.0))
    t_new = (1.0/MatrixNormEst) * ((fact*tol)/(4*Beta*MatrixNormEst))**InvM
    s = 10.0**(floor(log10(t_new))-1.0)
    t_new = ceil(t_new/s)*s
    nstep = 0

    w = v
    hump = Beta

    V = mat(zeros((n, m+1), dtype=complex))
    H = mat(zeros((m+2, m+2), dtype=complex))
    nsteps = 0 #Counter
    while (t_now < tLength):
        nsteps += 1
        t_step = min(tLength - t_now, t_new)

        V[:, 0] = (1.0/Beta)*w

        #Do Arnoldi Process to build H
        for j in xrange(0,m):
            p = matVec(V[:,j])
            for i in xrange(0, j):
                H[i,j] = (V[:,i].H * p)[0,0]
                p = p - H[i,j] * V[:,i]
            s = norm(p)

            #Check for "Happy Breakdown", that is we have a near-ideal representation
            #of H for Krylov Dimension < m
            if (s < btol):
                k1 = 0
                mb = j+1
                t_step = tLength - t_now
                break

            H[j+1, j] = s
            V[:, j+1] = (1.0/s) * p
        if ( k1 != 0):
            H[m+1, m] = 1.0
            avnorm = norm(matVec(V[:,m]))

        #Find the maximum timestep we can take with our Krylov matrix H such that
        # expm(t_step H) has an expected error smaller than our tolerance
        ireject = 0
        while (ireject <= MaxRejects):
            mx = mb + k1
            
            #Currently use scipy.linalg dense matrix exponentiation which
            #uses a Pade Approximant
            F = mat(expm(tSign * t_step * H[0:mx,0:mx]))

            #Treat Ideal Happy Breakdown Case First
            if k1 == 0:
                LocalError = btol
                break
            #Compute expected local error in expm(t_step H)
            else:
                phi1 = abs(Beta * F[m,0])
                phi2 = abs(Beta * F[m+1, 0] * avnorm)
                if phi1 > 10.0 * phi2:
                    LocalError = phi2
                    InvM = 1.0/m
                elif phi1 > phi2:
                    LocalError = (phi1 * phi2) / (phi1 - phi2)
                    InvM = 1.0/m
                else:
                    LocalError = phi1
                    InvM = 1.0/(m-1)
            if LocalError <= delta * t_step * tol:
                break
            else:
                t_step = gamma * t_step * (t_step * tol / LocalError)**InvM
                s = 10.0**(floor(log10(t_step)) - 1.0)
                t_step = ceil(t_step/s) * s
            if ireject == MaxRejects:
                print "Requested Tolerance in EXPV is too high"
            ireject += 1
        
        #Apply expm(t_step H) to our vector
        mx = mb + max(0, k1-1)
        w = V[:, 0:mx] * (Beta * F[0:mx,0])

        #Find new norm and "hump" error estimate
        Beta = norm(w)
        hump = max(hump, Beta)

        #Update with the timestep we were actually able to take, and suggest
        #the next time
        t_now += t_step
        t_new = gamma * t_step * (t_step * tol/LocalError) ** InvM
        s = 10.0**(floor(log10(t_new)) - 1.0)
        t_new = ceil(t_new / s) * s

        #Compare LocalError to Expected Round Off Errors
        LocalError = max(LocalError, RoundOffErr)
        TotalLocalError = TotalLocalError + LocalError
        
    FinalLocalError = TotalLocalError
    hump = hump/OriginalNormV

    #Optionally check the number of iterations it took
    #print "Number of steps: %d" % nsteps

    return w, FinalLocalError, hump
                


#Test the above routines against dense matrix exponentiation to make sure things are okay
if __name__ == "__main__":
    import time

    testCycles = 1
    for i in xrange(0, testCycles):
        nTest = int(30)
        normConst = 4.0
        t = 50.0
        print "Testing Arnoldi Exponential on Random Real Dense %dx%d" % (nTest, nTest)
        
        M = mat(normConst * random.randn(nTest,nTest))
        normM = norm(M)
        v = mat(random.randn(nTest,1))
        t = 3.0

        start_time = time.time()
        (w, error, hump) = expv(t, lambda x: M*x, v, normM)
        end_time = time.time()
        print "Krylov Time: %f" % (end_time-start_time)

        start_time = time.time()
        testExp = expm(t*M)
        testV = testExp*v
        end_time = time.time()
        print "Dense Pade Exp Time: %f" % (end_time - start_time)

        start_time = time.time()
        testExp2 = expm2(t*M)
        testV2 = testExp2*v
        end_time = time.time()
        print "Dense Eig Decomp Exp Time: %f" % (end_time - start_time)

        start_time = time.time()
        testExp3 = expm3(t*M)
        testV3 = testExp3*v
        end_time = time.time()
        print "Dense Taylor Exp Time: %f" % (end_time - start_time)
        
        relativeErr = norm(testV - w)/norm(testV)
        
        print "Relative Difference Between exp(M)*v using Krylov and Dense Pade Approx: %f" % relativeErr 
        if abs(relativeErr) > 1e-6:
            print "Tolerance Exceeded:"


        print "Testing Arnoldi Exponential on Random Dense Complex %dx%d" % (nTest, nTest)
        M = mat(complex(0.0,normConst) * random.randn(nTest,nTest))
        normM = norm(M)
        v = mat(random.randn(nTest,1), dtype=complex)

            
        relativeErr = norm(testV - w)/norm(testV)
        if abs(relativeErr) > 1e-6:
            print "Tolerance Exceeded:"

        start_time = time.time()
        (w, error, hump) = expv(t, lambda x: M*x, v, normM)
        end_time = time.time()
        print "Krylov Time: %f" % (end_time-start_time)

        start_time = time.time()
        testExp = expm(t*M)
        testV = testExp*v
        end_time = time.time()
        print "Dense Pade Exp Time: %f" % (end_time - start_time)

        start_time = time.time()
        testExp2 = expm2(t*M)
        testV2 = testExp2*v
        end_time = time.time()
        print "Dense Eig Decomp Exp Time: %f" % (end_time - start_time)

        start_time = time.time()
        testExp3 = expm3(t*M)
        testV3 = testExp3*v
        end_time = time.time()
        print "Dense Taylor Exp Time: %f" % (end_time - start_time)

        print "Relative Difference Between exp(M)*v using Krylov and Dense Pade Approx: %f" % relativeErr 

        
        nSparseDim = 500
        sparseRatio = .95
        
        print "Testing Random Sparse Matrix of Dimension %dx%d" % (nSparseDim, nSparseDim)
        M = mat(complex(0.0,normConst) * random.randn(nSparseDim,nSparseDim))
        for i in xrange(0, nSparseDim):
            for j in xrange(0, nSparseDim):
                if random.rand() < sparseRatio:
                    M[i,j] = 0
        sparseM = csr_matrix(M)
        normM = norm(M)
        v = mat(random.randn(nSparseDim,1))
        t = 3.0

        start_time = time.time()
        (w, error, hump) = expv(t, lambda x: sparseM*x, v, normM)
        end_time = time.time()
        print "Krylov Time: %f" % (end_time-start_time)

        start_time = time.time()
        testExp = expm(t*M)
        testV = testExp*v
        end_time = time.time()
        print "Dense Pade Exp Time: %f" % (end_time - start_time)

        relativeErr = norm(testV - w)/norm(testV)
        if abs(relativeErr) > 1e-6:
            print "Tolerance Exceeded:"
        print "Relative Difference Between exp(M)*v using Krylov and Dense Pade Approx: %f" % relativeErr 
