import matplotlib
from scipy.sparse import diags, identity

matplotlib.use('TkAgg') #TkAgg for vizual
from matplotlib import pyplot as plt
import time
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, spilu, cg, spsolve
from numpy.linalg import norm



Lfull = sparse.load_npz("L_Matrix.npz")
Wfull = sparse.load_npz("W_Matrix.npz")
yfull = np.loadtxt("y_vector.txt")
Wfull = Wfull.tocsc()
n = 60000
L = Lfull[0:n,0:n]
W = Wfull[0:n,0:n]
y = yfull[0:n]

L = L + 1e-2 * identity(L.shape[0])
L[0,1] = 1
L[1,0] = 1
H = L.T @ L + W

b = L.T @ L @ y

t00 = time.time()

# Lets try RCM + incomplete LU decomposition
perm = scipy.sparse.csgraph.reverse_cuthill_mckee(H, symmetric_mode=True)
I,J = np.ix_(perm,perm)
# H_ref = H[I,J]
Hcoo = H.tocoo()
I = Hcoo.row
J = Hcoo.col
V = Hcoo.data
In = perm[I]
Jn = perm[J]

Hr = sparse.csc_matrix((V, (In, Jn)), shape=(n, n))
t01 = time.time()
print("reordering took {}".format(t01-t00))
ilu = spilu(Hr,fill_factor=1.0)
t02 = time.time()
print("Factorization took {}".format(t02-t01))

Mx = lambda x: ilu.solve(x)
M = LinearOperator((n, n), Mx)
br = b[perm]
xrcm,stat = cg(Hr, br, tol=1e-12, maxiter=10000, M=M)
bsol = Hr @ xrcm
Rrcm = norm(bsol - br)
t03 = time.time()
print("RCM+iLU+CG took {} s, and had residual {}".format(t03-t00,Rrcm))

#Let's try various methods and see how they do
ML_lambda = lambda x: (L.T @ (L @ x)) + (W @ x)
A = LinearOperator((n, n), ML_lambda)
M = L.T @ L

# print("Starting with the Bias problem...")
# t1 = time.time()
# bias1,stat = cg(A, b, tol=1e-12, maxiter=10000, M=M)
# bsol = H @ bias1
# R1 = norm(bsol - b)
t2 = time.time()
# print("MatVec + precond took {} s, and had residual {}".format(t2-t1,R1))

bias2,stat = cg(A, b, tol=1e-12, maxiter=10000)
bsol = H @ bias2
R2 = norm(bsol - b)
t3 = time.time()

print("MatVec took {} s, and had residual {}".format(t3-t2,R2))


# bias3,stat = cg(H, b, tol=1e-12, maxiter=10000, M=M)
# bsol = H @ bias3
# R3 = norm(bsol - b)
# t4 = time.time()
#
# print("H + precond took {} s, and had residual {}".format(t4-t3,R3))
#
#
# bias4,stat = cg(H, b, tol=1e-12, maxiter=10000)
# bsol = H @ bias4
# R4 = norm(bsol - b)
# t5 = time.time()
#
# bias = bias4
# print("H took {} s, and had residual {}".format(t5-t4,R4))


b = bias2
print("Starting with the dBias problem...")
# t1 = time.time()
# x1,stat = cg(A, b, tol=1e-12, maxiter=10000, M=M)
# bsol = H @ x1
# R = norm(bsol - b)
t2 = time.time()
# print("MatVec + precond took {} s, and had residual {}".format(t2-t1,R1))

x2,stat = cg(A, b, tol=1e-12, maxiter=10000)
bsol = H @ x2
R = norm(bsol - b)
t3 = time.time()

print("MatVec took {} s, and had residual {}".format(t3-t2,R2))

#
# x3,stat = cg(H, b, tol=1e-12, maxiter=10000, M=M)
# bsol = H @ x3
# R = norm(bsol - b)
# t4 = time.time()
#
# print("H + precond took {} s, and had residual {}".format(t4-t3,R3))
#
#
# x4,stat = cg(H, b, tol=1e-12, maxiter=10000)
# bsol = H @ x4
# R = norm(bsol - b)
# t5 = time.time()
# print("H  took {} s, and had residual {}".format(t4-t3,R3))

