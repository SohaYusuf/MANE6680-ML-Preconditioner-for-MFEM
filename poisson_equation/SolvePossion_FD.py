# LIBRARY
# vector manipulation
import numpy as np
import math 

# THIS IS FOR PLOTTING
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt # side-stepping mpl backend
import warnings
warnings.filterwarnings("ignore")
from IPython.display import HTML
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsqr

def make_grid(N):

    x = np.linspace(0, 1, N, endpoint=True)    # make sure N grid points and last one is at 1.0
    y = np.linspace(0, 1, N, endpoint=True)    # make sure N grid points and last one is at 1.0

    h = x[1] - x[0]

    X, Y = np.meshgrid(x, y[::-1])
    
    fig = plt.figure()

    # plt.plot(x[1],y[1],'ro',label='unknown');
    plt.plot(X[1:-1,1:-1], Y[1:-1,1:-1], 'ro');

    plt.plot(np.ones(N), y, 'go', label='Boundary Condition');
    plt.plot(np.zeros(N), y, 'go');
    plt.plot(x, np.zeros(N), 'go');
    plt.plot(x, np.ones(N), 'go');
    plt.xlim((-0.1,1.1))
    plt.ylim((-0.1,1.1))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(r'Discrete Grid $\Omega_h,$ h= %s'%(h),fontsize=24,y=1.08)
    #plt.show()

    m = X.shape[1]
    
    coordinate = []
    
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            c = np.array([X[i,j], Y[i,j]], dtype=np.float32)
            coordinate.append(c)
    
    return x,y,X,Y,h,coordinate


def source(CHOICE,X,Y,N):
    
    if CHOICE == 'X+Y':
        T_b = X+Y
        f = np.zeros((N-2)**2)
    
    elif CHOICE == 'cos(Pi*X)':
        T_b = np.cos(np.pi*X)
        f = -np.pi**2*np.cos(np.pi*X[1:N-1,1:N-1]).reshape(-1,1)

    elif CHOICE == 'sin(Pi*X)':
        T_b = np.sin(np.pi*X)
        f = -np.pi**2*np.sin(np.pi*X[1:N-1,1:N-1]).reshape(-1,1)

    elif CHOICE == 'sin(Pi*X)+sin(Pi*Y)':
        T_b = np.sin(np.pi*X)+np.sin(np.pi*Y)
        f = -np.pi**2*np.sin(np.pi*X[1:N-1,1:N-1]).reshape(-1,1) - np.pi**2*np.sin(np.pi*Y[1:N-1,1:N-1]).reshape(-1,1)

    elif CHOICE == 'cos(Pi*X)+cos(Pi*Y)':
        T_b = np.cos(np.pi*X)+np.cos(np.pi*Y)
        f = -np.pi**2*np.cos(np.pi*X[1:N-1,1:N-1]).reshape(-1,1) -np.pi**2*np.cos(np.pi*Y[1:N-1,1:N-1]).reshape(-1,1)

    elif CHOICE == 'X*Y':
        T_b = X*Y
        f = np.zeros((N-2)**2)

    elif CHOICE == 'cos(Pi*X)*sin(Pi*Y)':
        T_b = np.cos(np.pi*X)*np.sin(np.pi*Y)
        f = -2*np.pi**2*np.cos(np.pi*X[1:N-1,1:N-1])*np.sin(np.pi*Y[1:N-1,1:N-1])
        f = f.reshape(-1,1)

    else:
        raise NotImplementedError("wrong CHOICE")
        
    return f, T_b


def set_boundary(w,T_b):

    w[0, :] = T_b[0, 1:-1]     # top
    w[-1, :] = T_b[-1, 1:-1]   # bottom
    w[:, 0] = T_b[1:-1, 0]     # left
    w[:, -1] = T_b[1:-1, -1]   # right
    
    return w


def plot_boundary(X,Y,w):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot a basic wireframe.
    ax.plot_wireframe(X[1:-1,1:-1], Y[1:-1,1:-1], w, color='r', rstride=10, cstride=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    plt.title(r'Boundary Values',fontsize=24,y=1.08)
    #plt.show()


def S(i,j,N):
    return (i-1)*(N-2) + j-1

def Sinv(s,N):
    j = s % (N-2) + 1
    i = 1+(s - (j - 1))//(N-2)
    return i,j


def set_interior_nodes(A,N,h):

    for i in range(2,N-2):
        for j in range(2,N-2):
            # print(i,j)

            A[S(i,j,N), S(i+1,j,N)] = 1/h**2 # the one below
            A[S(i,j,N), S(i-1,j,N)] = 1/h**2 # the one above
            A[S(i,j,N), S(i,j,N)]  = -4/h**2 
            A[S(i,j,N), S(i,j+1,N)] = 1/h**2 # the one to the right
            A[S(i,j,N), S(i,j-1,N)] = 1/h**2 # the one to the left
            
    return A


def set_top_boundary(A,N,h,b,T_b):
    
    for i in [1]:
        for j in range(2,N-2):
            # print(i,j)

            A[S(i,j,N), S(i+1,j,N)] = 1/h**2 # the one below
            # A[S(i,j), S(i-1,j)] = 1/h**2 # the one above
            A[S(i,j,N), S(i,j,N)]  = -4/h**2 
            A[S(i,j,N), S(i,j+1,N)] = 1/h**2 # the one to the right
            A[S(i,j,N), S(i,j-1,N)] = 1/h**2 # the one to the left

            # now the top one will have the boundary forcing acting on (i,j) element
            #print('A is: ',A)
            b[S(i,j,N)] += -1/h**2 * T_b[i-1,j] 
            #print('b is: ',b)
            
    return A,b


def set_bottom_boundary(A,N,h,b,T_b):
    
    for i in [N-2]:
        for j in range(2,N-2):
            # print(i,j)

            # A[S(i,j), S(i+1,j)] = 1/h**2 # the one below
            A[S(i,j,N), S(i-1,j,N)] = 1/h**2 # the one above
            A[S(i,j,N), S(i,j,N)]  = -4/h**2 
            A[S(i,j,N), S(i,j+1,N)] = 1/h**2 # the one to the right
            A[S(i,j,N), S(i,j-1,N)] = 1/h**2 # the one to the left

            # now the below one will have the boundary acting on (i,j) element
            b[S(i,j,N)] += -1/h**2 * T_b[i+1,j] 
            
    return A,b


def set_left_boundary(A,N,h,b,T_b):
    
    for j in [1]:
        for i in range(2,N-2):
            # print(i,j)

            A[S(i,j,N), S(i+1,j,N)] = 1/h**2 # the one below
            A[S(i,j,N), S(i-1,j,N)] = 1/h**2 # the one above
            A[S(i,j,N), S(i,j,N)]  = -4/h**2 
            A[S(i,j,N), S(i,j+1,N)] = 1/h**2 # the one to the right
            # A[S(i,j), S(i,j-1)] = 1/h**2 # the one to the left

            # now the left one will have the boundary acting on (i,j) element
            b[S(i,j,N)] += -1/h**2 * T_b[i,j-1] 
            
    return A,b


def set_right_boundary(A,N,h,b,T_b):
    
    for j in [N-2]:
        for i in range(2,N-2):
            # print(i,j)

            A[S(i,j,N), S(i+1,j,N)] = 1/h**2 # the one below
            A[S(i,j,N), S(i-1,j,N)] = 1/h**2 # the one above
            A[S(i,j,N), S(i,j,N)]  = -4/h**2 
            # A[S(i,j), S(i,j+1)] = 1/h**2 # the one to the right
            A[S(i,j,N), S(i,j-1,N)] = 1/h**2 # the one to the left

            # now the right one will have the boundary acting on (i,j) element
            b[S(i,j,N)] += -1/h**2 * T_b[i,j+1] 
            
    return A,b


def set_corners(A,N,h,b,T_b):
    
    ### top left
    i = 1; j = 1

    A[S(i,j,N), S(i+1,j,N)] = 1/h**2 # the one below
    # A[S(i,j), S(i-1,j)] = 1/h**2 # the one above
    A[S(i,j,N), S(i,j,N)]  = -4/h**2 
    A[S(i,j,N), S(i,j+1,N)] = 1/h**2 # the one to the right
    # A[S(i,j), S(i,j-1)] = 1/h**2 # the one to the left

    # now the above the left ones will have the boundary acting on (i,j) element
    b[S(i,j,N)]+= -1/h**2 * T_b[i-1,j] 
    b[S(i,j,N)]+= -1/h**2 * T_b[i,j-1] 


    ### top right
    i = 1; j = N-2

    A[S(i,j,N), S(i+1,j,N)] = 1/h**2 # the one below
    # A[S(i,j), S(i-1,j)] = 1/h**2 # the one above
    A[S(i,j,N), S(i,j,N)]  = -4/h**2 
    # A[S(i,j), S(i,j+1)] = 1/h**2 # the one to the right
    A[S(i,j,N), S(i,j-1,N)] = 1/h**2 # the one to the left

    # now the above the left ones will have the boundary acting on (i,j) element
    b[S(i,j,N)]+= -1/h**2 * T_b[i-1,j] 
    b[S(i,j,N)]+= -1/h**2 * T_b[i,j+1] 


    ### bottom left
    i = N-2; j = 1

    # A[S(i,j), S(i+1,j)] = 1/h**2 # the one below
    A[S(i,j,N), S(i-1,j,N)] = 1/h**2 # the one above
    A[S(i,j,N), S(i,j,N)]  = -4/h**2 
    A[S(i,j,N), S(i,j+1,N)] = 1/h**2 # the one to the right
    # A[S(i,j), S(i,j-1)] = 1/h**2 # the one to the left

    # now the above the left ones will have the boundary acting on (i,j) element
    b[S(i,j,N)]+= -1/h**2 * T_b[i+1,j] 
    b[S(i,j,N)]+= -1/h**2 * T_b[i,j-1] 


    ### bottom right
    i = N-2; j = N-2

    # A[S(i,j), S(i+1,j)] = 1/h**2 # the one below
    A[S(i,j,N), S(i-1,j,N)] = 1/h**2 # the one above
    A[S(i,j,N), S(i,j,N)]  = -4/h**2 
    # A[S(i,j), S(i,j+1)] = 1/h**2 # the one to the right
    A[S(i,j,N), S(i,j-1,N)] = 1/h**2 # the one to the left

    # now the above the left ones will have the boundary acting on (i,j) element
    b[S(i,j,N)]+= -1/h**2 * T_b[i+1,j] 
    b[S(i,j,N)]+= -1/h**2 * T_b[i,j+1] 
    
    return A,b


def plot_matrix(A,h,N,b):
    
    fig = plt.figure(figsize=(12,4));
    plt.subplot(121)
    plt.imshow(A, cmap='RdBu',vmin=-0.1/h**2,vmax=0.1/h**2);
    plt.show()

    plt.subplot(122)
    plt.imshow((b).reshape((N-2)**2,1),cmap='RdBu',vmin=-0.1/h**2,vmax=0.1/h**2);
    plt.show()


def plot_inverse(A_inv,h,A):
    
    A_inv = np.linalg.inv(A)
    plt.imshow(A_inv,cmap='RdBu',vmax=0.1*h**2,vmin=-0.1*h**2);
    plt.show()


def solve(A,b,h,T_b,N):
    
    #T_s_1d = np.linalg.lstsq(A*h**2,b*h**2)[0]
    T_s_1d = lsqr(A*h**2,b*h**2)[0]

    T_s = np.copy(T_b)
    for s in range(len(T_s_1d)):
        # print(s)
        i,j = Sinv(s,N)
        T_s[i,j] = T_s_1d[s]
        
    return T_s


def plot_results(X,Y,T_b,T_s):
    
    fig = plt.figure(figsize=(8,4));
    plt.subplot(121)
    plt.pcolor(X,Y,T_b)
    plt.title('True')

    plt.subplot(122)
    plt.pcolor(X,Y,T_s)
    plt.title('FD')


def error(T_b,T_s,h):
    r = np.linalg.norm(T_b - T_s, ord='fro') * h
    return r


def main(min_N,max_N):
    errors = []
    i = 0
    
    for no_points in range(min_N,max_N,5):

        N = no_points
        
        # CHOICE = 'cos(Pi*X)'
        # CHOICE = 'sin(Pi*X)'
        # CHOICE = 'sin(Pi*X)+sin(Pi*Y)'
        # CHOICE = 'cos(Pi*X)+cos(Pi*Y)'
        # CHOICE = 'X*Y'   
        CHOICE = 'cos(Pi*X)*sin(Pi*Y)'

        x,y,X,Y,h,coordinate = make_grid(N)

        f,T_b = source(CHOICE,X,Y,N)

        w = np.zeros((N-2,N-2))  # here we only have N-2 unknowns

        #plot_boundary(X,Y,w)

        T_b.flatten()

        M = (N-2)*(N-2)

        A = np.zeros((M,M))

        b = f    #source term

        # initilaize matrix A
        A = set_interior_nodes(A,N,h)
        A,b = set_top_boundary(A,N,h,b,T_b)
        A,b = set_bottom_boundary(A,N,h,b,T_b)
        A,b = set_left_boundary(A,N,h,b,T_b)
        A,b = set_right_boundary(A,N,h,b,T_b)
        A,b = set_corners(A,N,h,b,T_b)
        
        sA = csc_matrix(A)

        #plot_matrix(A,h,N,b)
        A_inv = np.linalg.inv(A)
        #plot_inverse(A_inv,h,A)
        T_s = solve(sA,b,h,T_b,N)
        #plot_results(X,Y,T_b,T_s)
        err = error(T_b,T_s,h)
        h_p = "{:.5f}".format(h)
        print(f'N = {no_points}, h = {h_p} ---->  error = {err}')
        errors.append(err)
        
        
        if i == 0:
            tmp1 = np.array(coordinate)
            coordinates = tmp1
            tmp2 = T_s.flatten()
            labels = tmp2
            
        else:
            coordinates = np.append(coordinates, coordinate, axis=0)
#             label = T_s.flatten()
#             labels = np.append(labels, label, axis=0)

           
        i+=1
        
    dim_coor = coordinates.shape[0]
    labels = T_b
    labels = np.reshape(labels, (dim_coor,1))
    labels = labels.astype(np.float32)
    dim_coor = coordinates.shape[0]
    #print('feature: ', coordinates)
    #print('label:', labels)
        
#     plt.figure(figsize=(8,8))

#     plt.plot(errors)
#     plt.title('Error vs Grid dimension')
#     plt.ylabel('error')
#     plt.xlabel('# grid dimension (N)')
#     plt.yscale("log") 
#     plt.xscale("log") 
#     plt.grid()
#     plt.savefig('error')
    
    return A,b,coordinates,errors,labels

if __name__ == '__main__':
    app.run(main)