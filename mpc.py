import numpy as np
from matplotlib import pyplot as plt


def optimal_control(n,x0,A,B,Q,R):
    C = np.zeros(shape=(n,n))
    _A = np.zeros(n)
    _Q= Q* np.eye(n)
    _R= R* np.eye(n)
    _Q[n-1,n-1] = 0 
    for i in range(n):
        _A[i] = A**(i+1) 
        for j in range(n):
            if j >= j:
                C[i,j] = B * A**(j-i)  
    H = np.dot(np.dot(C.T,_Q),C) + _R
    invH= np.linalg.inv(H)
    Ft= np.dot(np.dot(_A.T,_Q),C)
    U = -x0 * np.dot(invH,Ft.T)
    X= np.dot(C,U) + x0*_A
    
    return U, X

#not used yet
def u_constrains(n,umin,umax):
    X = np.eye(n)
    XX = np.concatenate((X,X))
    _umin = umin *np.ones(n)
    _umax = umax *np.ones(n)
    ubounds = np.concatenate((umin ,umax))
    return XX, ubounds

def mpc_regulation(n,x0,A,B,Q,R):
    U, X = optimal_control(n,x0,A,B,Q,R)
    return U[0], X[0]

def mpc_tracking(n,x0,ref,A,B,C,Q,R,S):
    _A = np.matrix([[A,B],[0,1]])
    _B = np.matrix([B,1])
    _C = np.matrix([C,0])
    _Q = np.zeros(shape= (2*n,2*n))
    _T = np.zeros(shape= (n, 2*n))
    _R = R * np.eye(n)
    __C = np.zeros(shape= (2*n,n))
    __A = np.zeros(shape=(2*n,2)) 
    for i in range(n): 
        __A[2*i:2*i+2] = np.linalg.matrix_power(_A,i+1)

    for i in np.arange(0,n-1):
        _Q[2*i:2*i+2,2*i:2*i+2] = Q*np.dot(_C.T,_C)
        _T[i,2*i:2*i+2] = Q*_C

    _Q[2*n-2:2*n,2*n-2:2*n] = S* np.dot(_C.T,_C)
    _T[n-1,2*n-2:2*n] = S*_C

    for i in range(n):
        for j in range(2*n):
            if i<=j :
                __C[2*i:2*i+2,j:j+1] = np.dot(np.linalg.matrix_power(_A,j-i),_B.T)

    H = np.linalg.multi_dot([__C.T,_Q,__C]) + _R        
    invH = np.linalg.inv(H)
    F1 = np.linalg.multi_dot([__A.T,_Q,__C])
    F2 = np.dot(-_T,__C)
    F= np.vstack([F1,F2]).T
    xref = np.concatenate([[x0,0.001],ref])
    Fx = F.dot(xref)
    du =  -np.dot(invH,Fx)
    x = np.dot(__C,du) + __A.dot([x0,0.001])
    return du , x

def test_optimal_and_regulation():
    Q = 2
    R = 1
    n = 10 
    A = 1
    B = 1
    x = np.zeros(n)
    x[0]= -1
    ur = np.zeros(n)
    xr = np.zeros(n)
    uopt ,xopt = optimal_control(n,x[0],A,B,Q,R)
    ur[0], xr[0] = mpc_regulation(n,x[0],A,B,Q,R)
    for i in range(1,n):
        ur[i] ,xr[i] = mpc_regulation(n,xr[i-1],A,B,Q,R)
    print(xopt.shape)
    print(uopt.shape)
    print("x")
    print(xopt)
    print("u")
    print(uopt)
    print ("regulation x")
    print(xr)
    print("regulation u")
    print(ur)
    plt.subplot(4,1,1)
    plt.plot(xopt)
    plt.subplot(4,1,2)
    plt.plot(uopt)
    plt.subplot(4,1,3)
    plt.plot(xr)
    plt.subplot(4,1,4)
    plt.plot(ur)
    plt.show()



def test_mcp_tracking():
    n= 10
    x0= 0.001 #np.zeros(2)
    ref = np.zeros(n)
    ref[3:5]= 4
    ref[5:7] =5
    ref[7:n]= 3
    A = 1 
    B = 1
    C = 1
    Q = 100
    R = 1
    S = 0.00001
    du , xt = mpc_tracking(n,x0,ref,A,B,C,Q,R,S)
    u = xt[1::2] +du
    x= xt[0::2]
    print("x")
    print(x)
    print("u")
    print(u)
    index = np.arange(n)
    plt.subplot(4,1,1)
    plt.bar(index , u)
    plt.subplot(4,1,2)
    plt.bar(index,du)
    plt.subplot(4,1,3)
    plt.plot(x)
    plt.plot(ref)
    plt.subplot(4,1,4)
    plt.plot(x-ref)
    plt.show()


#test_optimal_and_regulation()
#test_mcp_tracking()


