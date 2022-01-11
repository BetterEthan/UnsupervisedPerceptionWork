import numpy
from sklearn import metrics
import copy
import numpy as np

def cal_matrix_P(X,neighbors):
    entropy=numpy.log(neighbors)
    n1,n2=X.shape
    D=numpy.square(metrics.pairwise_distances(X))
    D_sort=numpy.argsort(D,axis=1)
    P=numpy.zeros((n1,n1))
    for i in range(n1):
        Di=D[i,D_sort[i,1:]]
        P[i,D_sort[i,1:]]=cal_p(Di,entropy=entropy)
    P=(P+numpy.transpose(P))/(2*n1)
    P=numpy.maximum(P,1e-100)
    return P

def cal_entropy(D,beta):
        # P=numpy.exp(-(numpy.sqrt(D))*beta)
    P=numpy.exp(-D*beta)
    sumP=sum(P)
    sumP=numpy.maximum(sumP,1e-200)
    H=numpy.log(sumP) + beta * numpy.sum(D * P) / sumP
    return H

def cal_matrix_Q(Y):
    n1,n2=Y.shape
    D=numpy.square(metrics.pairwise_distances(Y))
    #Q=1/(1+numpy.exp(D))
    #Q=1/(1+numpy.square(D))
    #Q=1/(1+2*D)
    #Q=1/(1+0.5*D)
    Q=(1/(1+D))/(numpy.sum(1/(1+D))-n1)
    Q=Q/(numpy.sum(Q)-numpy.sum(Q[range(n1),range(n1)]))
    Q[range(n1),range(n1)]=0
    Q=numpy.maximum(Q,1e-100)
    return Q


def cal_p(D,entropy,K=50):
    beta=1.0
    H=cal_entropy(D,beta)
    error=H-entropy
    k=0
    betamin=-numpy.inf
    betamax=numpy.inf
    while numpy.abs(error)>1e-4 and k<=K:
        if error > 0:
            betamin=copy.deepcopy(beta)
            if betamax==numpy.inf:
                beta=beta*2
            else:
                beta=(beta+betamax)/2
        else:
            betamax=copy.deepcopy(beta)
            if betamin==-numpy.inf:
                beta=beta/2
            else:
                beta=(beta+betamin)/2
        H=cal_entropy(D,beta)
        error=H-entropy
        k+=1
    P=numpy.exp(-D*beta)
    P=P/numpy.sum(P)
    return P

def cal_gradients(P,Q,Y):
    n1,n2=Y.shape
    DC=numpy.zeros((n1,n2))
    for i in range(n1):
        E=(1+numpy.sum((Y[i,:]-Y)**2,axis=1))**(-1)
        F=Y[i,:]-Y
        G=(P[i,:]-Q[i,:])
        E=E.reshape((-1,1))
        G=G.reshape((-1,1))
        G=numpy.tile(G,(1,n2))
        E=numpy.tile(E,(1,n2))
        DC[i,:]=numpy.sum(4*G*E*F,axis=0)
    return DC

def cal_loss(P,Q):
    C = numpy.sum(P * numpy.log(P / Q))
    return C


def tsne(X,n=2,neighbors=30,max_iter=200):
    data=[]
    n1,n2=X.shape
    P=cal_matrix_P(X,neighbors)
    Y=numpy.random.randn(n1,n)*1e-4
    Q = cal_matrix_Q(Y)
    DY = cal_gradients(P, Q, Y)
    A=200.0
    B=0.1
    for i  in range(max_iter):
        data.append(Y)
        if i==0:
            Y=Y-A*DY
            Y1=Y
            error1=cal_loss(P,Q)
        elif i==1:
            Y=Y-A*DY
            Y2=Y
            error2=cal_loss(P,Q)
        else:
            YY=Y-A*DY+B*(Y2-Y1)
            QQ = cal_matrix_Q(YY)
            error=cal_loss(P,QQ)
            if error>error2:
                A=A*0.7
                continue
            elif (error-error2)>(error2-error1):
                A=A*1.2
            Y=YY
            error1=error2
            error2=error
            Q = QQ
            DY = cal_gradients(P, Q, Y)
            Y1=Y2
            Y2=Y
        if cal_loss(P,Q)<1e-3:
            return Y
        print ('%s iterations the error is %s, A is %s', str(i+1),str(round(cal_loss(P,Q),2)),str(round(A,3)))
    return Y

import matplotlib.pyplot as plt
def test_iris():
    data = np.load('latentDataNew32.npy')
    X=data.data
    Y=tsne(X,n=2,max_iter=300,neighbors=20)
    figure1=plt.figure()
    plt.subplot(1,2,1)
    plt.plot(Y[0:50,0],Y[0:50,1],'ro',markersize=30)
    plt.plot(Y[50:100,0],Y[50:100,1],'gx',markersize=30)
    plt.plot(Y[100:150,0],Y[100:150,1],'b*',markersize=30)
    plt.title('CUSTOM')
    plt.subplot(1,2,2)
    # t1=time.time()
    # Y1=manifold.TSNE(2).fit_transform(data.data)
    # t2=time.time()
    # print ("Sklearn TSNE cost time: %s"%str(round(t2-t1,2)))
    # plt.plot(Y1[0:50,0],Y1[0:50,1],'ro',markersize=30)
    # plt.plot(Y1[50:100,0],Y1[50:100,1],'gx',markersize=30)
    # plt.plot(Y1[100:150,0],Y1[100:150,1],'b*',markersize=30)
    # plt.title('SKLEARN')
    plt.show()


test_iris()
input(2343243)