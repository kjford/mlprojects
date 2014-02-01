# Neural network implementation of sparse encoder

import numpy as np
import scipy.io
import scipy.optimize as optm
import matplotlib.pyplot as plt

# functions:

def loadImagePatches(imfile='testdata/IMAGES.mat',
        imvar='IMAGES',patchsize=8,npatches=10000,edgebuff=5,scale0to1=True):
    # open .mat file containing images in a r x c x num images array
    # load patches that are patchsize x patchsize
    # normalize scale to 0 to 1 values
    imgdict = scipy.io.loadmat(imfile)
    imgarray = imgdict[imvar]
    # get dimentions
    r = imgarray.shape[0] - 2*edgebuff - patchsize
    c = imgarray.shape[1] - 2*edgebuff - patchsize
    nimg = imgarray.shape[2]
    
    # allocate random numbers and patches arrays
    patches = np.zeros([patchsize**2,npatches])
    randrow = np.random.randint(r,size=npatches) + edgebuff
    randcol = np.random.randint(c,size=npatches) + edgebuff
    randimg = np.random.randint(nimg,size=npatches)
    
    for i in range(npatches):
        r1 = randrow[i]
        r2 = r1+patchsize
        c1 = randcol[i]
        c2 = c1 + patchsize
        imi = randimg[i]
        patchi = imgarray[r1:r2,c1:c2,imi]
        patches[:,i] = patchi.reshape(1,patchsize**2)
    
    # normalize
    # subtract mean and scale by 3 stdev's
    patches -= patches.mean(0)
    pstd = patches.std() * 3
    patches = np.maximum(np.minimum(patches, pstd),-pstd) / pstd
    
    if scale0to1:
        # Rescale from [-1,1] to [0.1,0.9]
        patches = (patches+1) *  0.4 + 0.1
    
    return patches

def squareImgPlot(I):
    # show n square images in a L x M array as single large panel image
    # where each image is L**0.5 x L**0.5 pixels
    # plotted image is M**0.5
    I = I - np.mean(I)
    (L, M)=I.shape
    sz=int(np.sqrt(L))
    buf=1
    if np.floor(np.sqrt(M))**2 != M :
        n=int(np.ceil(np.sqrt(M)))
        while M % n !=0 and n<1.2*np.sqrt(M): n+=1
        m=int(np.ceil(M/n))
    else:
        n=int(np.sqrt(M))
        m=n
    a=-np.ones([buf+m*(sz+buf)-1,buf+n*(sz+buf)-1])
    k=0
    for i in range(m):
        for j in range(n):
            if k>M: 
                continue
            clim=np.max(np.abs(I[:,k]))
            r1=buf+i*(sz+buf)
            r2=r1+sz
            c1=buf+j*(sz+buf)
            c2=c1+sz
            a[r1:r2,c1:c2]=I[:,k].reshape(sz,sz)/clim
            k+=1
        
    h = plt.imshow(a,cmap='gray',interpolation='none',vmin=-1,vmax=1)
    
def initWeights(layervec,usebias=True):
    # initialize weights to each layer of network between -r and r
    # layervec is array with size of input layer, each hidden layer, and output layer
    # outputs initialized weights rolled into a single vector
    # option to use a bias input to each layer
    r  = np.sqrt(6) / np.sqrt(np.sum(layervec[1:]))
    inweights = layervec[:-1]
    nunits = layervec[1:]
    totalW=np.multiply(inweights,nunits).sum()
    W=np.random.rand(totalW)*2*r-r
    if usebias:
        W=np.append(W,np.zeros(sum(nunits)))
    
    return W
    
def numericalGradient(J,theta,e=1e-4):
    # compute numerical gradient as slope of J at theta values
    # J is a function handle that returns a cost value (and probably gradient)
    perturb = np.zeros(np.size(theta))
    numgrad = np.zeros(np.size(theta))
    for p in range(np.size(theta)):
        perturb[p] = e
        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)
        # Compute Numerical Gradient
        numgrad[p] = (loss2[0] - loss1[0]) / (2*e)
        perturb[p] = 0
    return numgrad
        
def sparseNNCost(X,Y,theta,layervec,lam=0.0001,sparsityParam=0.01,beta=3,costtype='ss',outtype='sig'):
    # compute the cost and gradient of neural network
    # X is ndarray of n features by m examples
    # Y is ndarray of output layer size by m examples
    # note: for a sparse autoencoder X=Y
    # theta is a rolled vector of parameters
    # layervec is an list with the architecture of the network
    # lam is the regularization cost parameter
    # sparsityParam is the target sparsity value
    # beta is the sparsity cost parameter
    # defaults to a sum of squares cost (ss) of 1/m*(h(x)-y)**2
    # optional log likelihood cost (costtype='ll')
    # Defaults to a sigmoid (sig) output layer
    # optional linear (outtype='linear') output layer
    
    # get number of examples and number of layers
    m=X.shape[1]
    inweights = layervec[:-1]
    nunits = layervec[1:]
    totalW=np.multiply(inweights,nunits).sum()
    if theta.size>totalW:
        usebias=1
        B=[]
        bcount=0
    
    # perform forward pass through layers
    W=[]
    A=[X]
    wcount=0
    sigmoid = lambda x: (1+np.exp(-x))**-1
    for i in range(len(nunits)):
        nwi=inweights[i]
        nui=nunits[i]
        wi=theta[wcount:wcount+nwi*nui].reshape(nwi,nui)
        W.append(wi)
        
        a = np.dot(wi.T,A[i])
        if usebias:
            bi=theta[totalW+bcount:totalW+bcount+nui]
            a+=bi.reshape(nui,1)
            B.append(bi)
            bcount+=nui
        wcount+=nwi*nui
        if outtype=='linear' and i==len(nunits-1):
            A.append(a)
        else:
            A.append(sigmoid(a))
    # compute error    
    if costtype=='ss':
        errcost = ((A[-1] - Y)**2).sum()/(2.0*m)
    elif costtype == 'll':
        errcost = (-Y*log(A[-1]) - (1-Y)*log(1-A[-1])).sum()
    else:
        print('Error type not recognized. Using sum of squares error\n')
        errcost = ((A[-1] - Y)**2).sum()/(2.0*m)
        costtype='ss'
    
    # compute regularization cost
    regcost = 0.5 * lam * (theta[:totalW]**2).sum()
    
    # sparsity cost using KL divergence
    # for now only assessed on first hidden layer
    pj = (1.0/m)*A[1].sum(axis=1)
    
    p=sparsityParam
    KLdiv=p*np.log(p/pj) + (1-p)*np.log((1-p)/(1-pj))
    
    sparcost = beta * KLdiv.sum()
    
    # add up costs
    cost = errcost + regcost + sparcost
    #print(cost)
    
    # perform backpropagation
    if costtype=='ss':
        if outtype=='sig':
            errout = -(Y-A[-1])*A[-1]*(1-A[-1]) # vis x m
        else:
            errout = -(Y-A[-1])
    else:
        if outtype=='sig':
            errout = -(Y-A[-1])
        else:
            print('Log-likelihood error with linear outputs is not valid. Using sigmoid.')
            outtype=='sig'
            errout = -(Y-A[-1])
    # go backward through hidden layers
    layercount = range(len(A))
    revlayer = layercount[::-1][1:] #reversed count less last layer
    layererr = [errout,]
    Wgrad = W[:]
    
    if usebias:
        Bgrad = B[:]
    
    for i in revlayer:
        # err in layer is:
        # (weights transpose * err in layer+1) element wise * 
        # deriv of layer activation wrt activation fxn (sigmoid)
        
        # get outgoing weights
        wi=W[i]
        # err from layer n+1 
        erri=layererr[-1]
        # activation of layer i
        ai=A[i]
        derivi=ai*(1-ai)
        # if second layer then add sparsity err
        if i==1:
            # use pj (sparsity of layer 2 averaged over m samples (size l2)
            KLderiv = -(p/pj) + (1-p)/(1-pj);
            # need to make l2 x 1 to add to err of weights
            sparerr = beta * KLderiv.reshape(KLderiv.size,1)
            layererr.append((np.dot(wi,erri)+sparerr) * derivi)
            
        elif i>1:
            layererr.append(np.dot(wi,erri) * derivi)
        
        Wgrad[i] = np.dot(ai,erri.T)/m + lam * wi
        if usebias:
            Bgrad[i] = (erri.sum(axis=1))/m
            
    # string together gradients
    thetagrad=theta*1.
    wcount=0
    bcount=0
    for i in range(len(Wgrad)):
        nw=Wgrad[i].size
        thetagrad[wcount:nw+wcount]=Wgrad[i].reshape(nw)
        wcount+=nw
        if usebias:
            nb=Bgrad[i].size
            thetagrad[totalW+bcount:totalW+bcount+nb]=Bgrad[i].reshape(nb)
            bcount+=nb
        
    
    return(cost,thetagrad)

if __name__ == "__main__":
    # run with defaults
    p=loadImagePatches()
    l=[64,25,64]
    w0=initWeights(l)
    theta,fcost,fl=optm.fmin_l_bfgs_b(lambda x: sparseNNCost(p,p,x,l),w0)
    squareImgPlot(theta[:(64*25)].reshape(64,25))
    plt.savefig('W1.eps')
