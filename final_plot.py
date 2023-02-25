import numpy as np
from numpy import kron
import matplotlib.pyplot as plt
import functools as ft
from copy import deepcopy
from scipy.linalg import expm

#%%
########## Contstruction of the Hamiltonian matrix

def create_spin_operators(Ns):
    Sx = 1/2 * np.array([[0, 1], [1, 0]])
    Sy = 1j/2 * np.array([[0, -1], [1, 0]])
    Sz = 1/2 * np.array([[1, 0], [0, -1]])
    I = np.array([[1, 0], [0, 1]])
    
    
    
    #lists to store the combinations of matrices to turn into tensorial product
    X = [] #for Ns=3 this list will be [Sx, I, I, I, Sx, I, I, I, Sx]
    Y = []
    Z = []
    for i in range(Ns):
         for j in range(Ns):
             if i==j:
                 X.append(Sx)
                 Y.append(Sy)
                 Z.append(Sz)
             if i!=j:
                 X.append(I)
                 Y.append(I)
                 Z.append(I) 
     #tensorial products concatenated by ft.reduce(function, list)
    Sx_list = [ft.reduce(kron, X[i:i+Ns]) for i in range(0, Ns**2, Ns)]
    Sy_list = [ft.reduce(kron, Y[i:i+Ns]) for i in range(0, Ns**2, Ns)]
    Sz_list = [ft.reduce(kron, Z[i:i+Ns]) for i in range(0, Ns**2, Ns)]   
    
    S = []
    for i in range(Ns):
        S.append([Sx_list[i], Sy_list[i], Sz_list[i]])  
    return S, Sx_list
    
#function to describe the nearest neighbour distance considered
def NN_condition(Ns, index1, index2, distance = 1):
    bools = []
    for i in range(distance):
        bools.append(bool(index1%Ns==(index2-distance)%Ns or index1%Ns==(index2+distance)%Ns))
    return bools
#first two dimensions of the matrix is the ij index, the last two are a 3x3 matrix with the D(ij) elements, in this case a diagonal matrix 
       
def dipole_matrix(Ns, J, distance):
    D = np.zeros((Ns, Ns, 3, 3), dtype=complex) #will have to change to more dimensions if different spins have different dipole moment or interaction
    for i in range(Ns):
        for j in range(Ns):
            conditions = NN_condition(Ns, i, j, distance)
            for k in range(3):
                for l in range(3):
                    for m in range(len(conditions)): #loop through distance conditions
                        if conditions[m] and k==l: #there's xy, yz and xz interaction for the dipolar moment?
                            D[i,j, k, l] += J * 10**(-m)  #the interaction decreases one order of magnitude per unit of distance
                    #si afegeixo un elif amb distance 2 veuré més coses?
    return D

#function to multiply B * g(i)*S(i)
def product1(B, gi, Si):
    f = 0
    for i in range(3):
        gs = 0
        for j in range(3):
            gs += gi[i, j] * Si[j] #g(i)*S(i)
        f += B[0,i] * gs
    return f
#external field term in the total hamiltonian
def external_field(B, g, S, Ns):
    H = np.zeros((2**Ns, 2**Ns), dtype=complex)
    for i in range(Ns):
        H += product1(B, g, S[i])
    return H

#function to multiply S(i) * D(ij) * S(j)
def product2(Si, D, Sj):
    sds = 0
    for i in range(3):
        ds = 0
        for j in range(3):
            ds += D[i,j] * Sj[j]  
        sds += Si[i] @ ds
    return sds


#spin-spin interaction term in the hamiltonian
    
    #for a spin chain only consecutive neighbours should interact, 
        #so only neighbour i and j with periodic boundary conditions (5th line condition)
def spin_spin(D, S, Ns, distance):
    H = np.zeros((2**Ns, 2**Ns), dtype=complex)
    for i in range(Ns):
        for j in range(Ns):
            conditions = NN_condition(Ns, i, j, distance)
            for k in range(len(conditions)):
                if conditions[k]: #this condition shouldn't be necessary with the D we built, but it might increase the performance
                    H += product2(S[i], D[i,j], S[j])
    return H


def Hamiltonian(mu, B, g, S, D, Ns, distance):
    return mu * external_field(B, g, S, Ns) + 1/2 * spin_spin(D, S, Ns, distance)


#%%

########## Diagonalisation of Hamiltonian

'''eigval, eigvec = np.linalg.eig(H)
'''
#print(eigval)

#print(eigvec[:,0])

#############################################################
#from vector to number
def v2num(vector): 
    zero = np.array([[1], [0]])
    if (vector==zero).all():
        return 0
    else: 
        return 1
    
#from number to vector
def num2v(num):
    zero = np.array([[1], [0]])
    one = np.array([[0], [1]])
    if num:
        return one
    else: 
        return zero

#creating the basis for n flips

def basis(n, Ns):
    eigenvector = [num2v(1) for i in range(Ns)]
    eigenvector0 = ft.reduce(kron, eigenvector)
    #to continue, add other vectors apart from the zeroth one, corrsponding to the n excitations
    basis = [eigenvector0]
    numeric_basis = [[1 for i in range(Ns)]]
    excitations = {0 : [eigenvector]}
    for k in range(1, n+1): 
        excitations[k] = []
        prev_exc = excitations[k-1] #list with n-1 excitation basis
        #print('# excitations:', k, 'of', n, '\n', Ns,'spins:')
        for i in prev_exc:
            for j in range(Ns):
                if v2num(i[j]) != 0:
                    new_i = deepcopy(i)
                    new_i[j] = num2v(0)
                    excitations[k].append(new_i)
                    vector = ft.reduce(kron, new_i)
                    numeric_vector = [v2num(klm) for klm in new_i]
                    #to avoid adding vectors already included: (need of numeric list here since bools with arrays are tricky)
                    if numeric_vector not in numeric_basis:
                        basis.append(vector)
                        #print(numeric_vector)
                        numeric_basis.append(numeric_vector)
    return basis

def truncated_operator(basis, operator):
    dim = len(basis)
    t_H = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            t_H[i,j] = complex(basis[i].T.conjugate() @ operator @ basis[j])
    return t_H


#############################################################

#%%

########## Propagation of density matrix (state)

#spin that will be measured (index from 0 to Ns-1)


#R(theta) is a 3x3 matrix, it rotates a single spin
def rotation(theta, axis = np.array([[0, 1, 0]])):
    Sx = 1/2 * np.array([[0, 1], [1, 0]])
    Sy = 1j/2 * np.array([[0, -1], [1, 0]])
    Sz = 1/2 * np.array([[1, 0], [0, -1]]) 
    Spin = [Sx, Sy, Sz]
    Sn = np.zeros((2,2), dtype=complex)
    for i in range(3):
        Sn += Spin[i] * float(axis[0,i])
    R = expm(1j * Sn * theta/(2*np.pi))
    return R

def unitary(dt, H):
    u = expm(-1j * dt * H)
    return u

def total_propagation(state, r1, r2, H, tau, time_steps=100):
    t = np.linspace(0, tau, time_steps)
    dt = t[1]-t[0]
    #first step: rotation of pi/2 on all spins 
    s =  r1 @ state
    #second step: unitary evolution during time tau
    u = unitary(dt, H)
    for i in range(time_steps):
        s = u @ s
    #third step: rotation of pi on all spins
    s =  r2 @ s
    #fourth step: unitary evolution during time tau
    for i in range(time_steps):
        s = u @ s
    return s
    

#%%
############# Results and plot
    

def measurement(state, operator):
    v = state.T.conjugate() @ operator @ state
    return float(np.real(v))


def final_plot(spin_index, Ns, distance, muB, B, g, J, basis, figure_label, truncate=False):
    #initialise state
    state_list = [num2v(1) for i in range(Ns)]
    state = ft.reduce(kron, state_list) #initial state
    #time discretisation
    tau_i = 0
    tau_f = 6*np.pi / muB
    num_taus = 200
    
    tau = np.linspace(tau_i, tau_f, num_taus)
    
    #initialise spin operators
    S, Sx_list = create_spin_operators(Ns)
    #rotations
    I = np.eye(2)
    r1 = rotation(np.pi/2)
    R1_list = [I for i in range(Ns)]
    R1_list[spin_index] = r1
    R1 = ft.reduce(kron, R1_list)
    
    r2 = rotation(np.pi)
    R2_list = R1_list
    R2_list[spin_index] = r2
    R2 = ft.reduce(kron, R2_list)
    
    D = dipole_matrix(Ns, J, distance)
    
    H = Hamiltonian(muB, B, g, S, D, Ns, distance)
    
    final_states = []
    measurements = []
    
    if truncate:
        eigval, eigvec = np.linalg.eig(np.eye(len(basis), len(basis)))
        state = eigvec[:,0]
        t_Sx_list = [truncated_operator(basis, i) for i in Sx_list]
        t_H = truncated_operator(basis, H)
        t_R1 = truncated_operator(basis, R1)
        t_R2 = truncated_operator(basis, R2)
        
        for i in range(num_taus):
            final_states.append(total_propagation(state, t_R1, t_R2, t_H, tau[i], 300)) #modify time steps to increase the accuracy of the time discretization
            measurements.append(measurement(final_states[i], t_Sx_list[spin_index]))
    else:
        for i in range(num_taus):
            final_states.append(total_propagation(state, R1, R2, H, tau[i], 300)) #modify time steps to increase the accuracy of the time discretization
            measurements.append(measurement(final_states[i], Sx_list[spin_index]))
    
    muB_tau = [i * muB for i in tau]
    plt.plot(muB_tau, measurements, label = figure_label, lw=1)

    
spin_index = 0
distance = 2
Ns = 9
muB = 1
J = 0.1 * muB
B = np.array([[0, 0, muB]], dtype=complex)
g = 2 * np.eye(3, dtype=complex) #is negative in theory
excitations = [0,1,2,3,4]
from time import process_time 
start_time = process_time()

plt.figure()
for i in excitations:
    b = basis(i, Ns)
    if i:
        figure_label = str(i)+' excitations'
    else:
        figure_label = 'Full space'
    final_plot(spin_index, Ns, distance, muB, B, g, J, b, figure_label, i)
    print(process_time()-start_time, 's')
    start_time = process_time()
    
plt.tick_params(axis='both', direction='in')
plt.ylabel(r'$\langle \hat{S}_x^1 \rangle/\hbar$')
plt.xlabel(r'$\mu_B B_0\tau/\hbar$')
plt.title(r'$N_S=$'+str(Ns)+', $J ='+str(J)+'\mu_B B_0$')
plt.grid(alpha=0.4)
plt.rc('legend',fontsize=9) # using a size in points
plt.legend()
plt.savefig('final_plot.png', bbox_inches = "tight", dpi=900)
plt.show()
#%%
