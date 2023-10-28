"""1. matrici e norme """

import numpy as np

#help(np.linalg) # View source
#help (np.linalg.norm)
#help (np.linalg.cond)

n = 2
A = np.array([[1, 2], [0.499, 1.001]])

norm1 = np.linalg.norm(A, 1)
norm2 =  np.linalg.norm(A, 2)
normfro = np.linalg.norm(A, 'fro')
norminf = np.linalg.norm(A, np.inf)

print("Norma 1 di A\t\t =", norm1)
print("Norma 2 di A\t\t =", norm2)
print("Norma Fro di A\t\t =", normfro)
print("Norma inf di A\t\t =", norminf)

print('=====================================')

cond1 = np.linalg.cond(A, p=1)
cond2 = np.linalg.cond(A, p=2)
condfro = np.linalg.cond(A, p='fro')
condinf = np.linalg.cond(A, p=np.inf)

print ('K(A)_1 \t\t\t\t =', cond1)
print ('K(A)_2 \t\t\t\t =', cond2)
print ('K(A)_fro \t\t\t =', condfro)
print ('K(A)_inf \t\t\t =', condinf)

print('=====================================')

x = np.ones((2,1))
b = np.dot(A, x)


btilde = np.array([[3], [1.4985]])
xtilde = np.array([[2, 0.5]]).T

# # Verificare che xtilde è soluzione di A xtilde = btilde (Axtilde)
my_btilde = np.dot(A, xtilde)

print ('A*xtilde =\n', "\t" + str(btilde).replace('\n','\n\t'))
print (np.linalg.norm(btilde-my_btilde,'fro'))

deltax = np.linalg.norm((x - xtilde), 2)
deltab = np.linalg.norm((b - btilde), 2)

print ('delta x \t\t\t =', deltax)
print ('delta b \t\t\t =', deltab)

print('\n\n##########################################################################')
print('\t\t2. FATTORIZZAZIONE LU\n\n')
# """2. fattorizzazione lu"""
import numpy as np

# # crazione dati e problema test
A = np.array ([ [3,-1, 1,-2], [0, 2, 5, -1], [1, 0, -7, 1], [0, 2, 1, 1]  ])
x = np.ones((4,1))
b = np.matmul(A, x) # b = Ax

condA = np.linalg.cond(A, p=1)

print('x: \n', "\t" + str(x).replace('\n','\n\t'), '\n')
print('x.shape: ', x.shape, '\n' )
print('b: \n', "\t" + str(b).replace('\n','\n\t'), '\n')
print('b.shape: ', b.shape, '\n' )
print('A: \n', "\t" + str(A).replace('\n','\n\t'), '\n')
print('A.shape: ', A.shape, '\n' )
print('K(A)=', condA, '\n')
print('=====================================\n')

import scipy
# # help (scipy)
import scipy.linalg
# # help (scipy.linalg)
from scipy.linalg import lu_factor  as LUdec  # pivoting
from scipy.linalg import lu         as LUfull # partial pivoting

lu, piv = LUdec(A)

print('lu = \n', "\t" + str(lu).replace('\n','\n\t'), '\n')
print('piv',piv,'\n')


# # risoluzione di    Ax = b   <--->  PLUx = b 
my_x = scipy.linalg.lu_solve((lu, piv), b)

print('my_x = \n', "\t" + str(my_x).replace('\n','\n\t'), '\n')
print('norm =', scipy.linalg.norm(x-my_x, 'fro'))

print('\n=====================================\n')
print('\t# IMPLEMENTAZIONE ALTERNATIVA - 1')
# IMPLEMENTAZIONE ALTERNATIVA - 1

P, L, U = LUfull(A)
print ('A = \n', "\t" + str(A).replace('\n','\n\t'), '\n')
print ('P = \n', "\t" + str(P).replace('\n','\n\t'), '\n')
print ('L = \n', "\t" + str(L).replace('\n','\n\t'), '\n')
print ('U = \n', "\t" + str(U).replace('\n','\n\t'), '\n')
print ('P*L*U = \n', "\t" + str(np.matmul(P , np.matmul(L, U))).replace('\n','\n\t'), '\n')
print ('diff = ',   np.linalg.norm(A - np.matmul(P , np.matmul(L, U)), 'fro'  ) ) 


# if P != np.eye(n): 
# Ax = b   <--->  PLUx = b  <--->  LUx = inv(P)b  <--->  Ly=inv(P)b & Ux=y : matrici triangolari
# quindi
invP = np.linalg.inv(P)
y = scipy.linalg.solve_triangular(U, scipy.linalg.solve_triangular(L, b, lower=True), lower=True, unit_diagonal=True)
my_x = scipy.linalg.solve_triangular(U, y, lower=False)

# if P == np.eye(n): 
# Ax = b   <--->  PLUx = b  <--->  PLy=b & Ux=y
# y = scipy.linalg.solve_triangular(np.matmul(P,L) , b, lower=True, unit_diagonal=True)
# my_x = scipy.linalg.solve_triangular(U, y, lower=False)


print('\nSoluzione calcolata: \n', "\t" + str(my_x).replace('\n','\n\t'), '\n')
print('norm =', scipy.linalg.norm(x-my_x, 'fro'))

print('\n\n##########################################################################')
print('\t\t2.2 CHOLESKI CON MATRICE DI HILBERT \n\n')
"""2.2 Choleski con matrice di Hilbert"""
import matplotlib.pyplot as plt
import numpy as np
import scipy
# help (scipy)
import scipy.linalg
# help (scipy.linalg)
# help (scipy.linalg.cholesky)
# help (scipy.linalg.hilbert)

# crazione dati e problema test
n = 5
A = scipy.linalg.hilbert(n)
x = np.ones((n,1))
b = np.matmul(A, x)

condA = np.linalg.cond(A,2)

print('x: \n', "\t" + str(x).replace('\n','\n\t'), '\n')
print('x.shape: ', x.shape, '\n' )
print('b: \n', "\t" + str(b).replace('\n','\n\t'), '\n')
print('b.shape: ', b.shape, '\n' )
print('A: \n', "\t" + str(A).replace('\n','\n\t'), '\n')
print('A.shape: ', A.shape, '\n' )
print('K(A)=', condA, '\n')

print('\n=====================================')
print('\t\tDECOMPOSIZIONE DI CHOLESKI')
# decomposizione di Choleski
L = scipy.linalg.cholesky(A, lower=True)
print('L: \n', "\t" + str(L).replace('\n','\n\t'), '\n')

print('L.T*L =', scipy.linalg.norm(A-np.matmul(np.transpose(L),L)))
print('err = ', scipy.linalg.norm(A-np.matmul(np.transpose(L),L), 'fro'))

y = scipy.linalg.solve(L, b)
my_x = scipy.linalg.solve(L.T, y)
print('my_x = \n', "\t" + str(my_x).replace('\n','\n\t'), '\n')

print('norm =', np.linalg.norm(x-my_x, 'fro'))


K_A = np.zeros((6,1))
Err = np.zeros((6,1))

for n in np.arange(5,11):
    # crazione dati e problema test
    A = scipy.linalg.hilbert(n)
    x = np.ones((n,1))
    b = np.matmul(A, x)
    
    # numero di condizione 
    K_A[n-5] = np.linalg.cond(A)
    
    # fattorizzazione 
    L = scipy.linalg.cholesky(A, lower=True)
    y = scipy.linalg.solve(L, b)
    my_x = scipy.linalg.solve(L.T, y)
    
    # errore relativo
    Err[n-5] = np.linalg.norm(x-my_x, 'fro')/np.linalg.norm(x, 'fro')
  
xplot = np.arange(5,11)

# grafico del numero di condizione vs dim
plt.semilogy(xplot, K_A)
plt.title('CONDIZIONAMENTO DI A ')
plt.xlabel('dimensione matrice: n')
plt.ylabel('K(A)')
plt.grid()
plt.show()


# grafico errore in norma 2 in funzione della dimensione del sistema
plt.plot(xplot, Err)
plt.title('Errore relativo')
plt.xlabel('dimensione matrice: n')
plt.ylabel('Err= ||my_x-x||/||x||')
plt.grid()
plt.show()