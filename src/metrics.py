import torch
import utils

'''
In this file we have all the metrics we test.
For now they should all take 2 arguments and return 1 number.
Anything that is not a distance metric to be tested should not be placed here.

Warning do not import any functions as they will also show up, rather just import the module it is in and use it locally.
'''
#
def L2dist(X,Y):
    return torch.norm(X - Y)
#
# def L1dist(X,Y):
#     return torch.norm(X - Y, p=1)
#
#
# # #
# def cpd(X, Y, W=1):
#     '''
#     Cloud point distribution
#     :param X:
#     :param Y:
#     :param W:
#     :return:
#     '''
#     A = utils.getDistMatrix(X, Y, W)
#     vcol, _ = torch.min(A, dim=0)
#     mcol, _ = torch.max(A, dim=0)
#     vrow, _ = torch.min(A, dim=1)
#     mrow, _ = torch.max(A, dim=1)
#
#     d = 0.5 * (torch.sum(vcol / mcol) + torch.sum(vrow / mrow))
#
#     return d
#
# # #
# def scpd(X, Y):
#     '''
#     Symmetric cloud point distribution?
#     :param X:
#     :param Y:
#     :return:
#     '''
#     A = utils.getDistMatrix(X, Y)
#     vcol = 1 / torch.sum(1 / (A + 1e-4), dim=0)
#     mcol = 1;  # torch.sum(A,dim=0)
#     vrow = 1 / torch.sum(1 / (A + 1e-4), dim=1)
#     mrow = 1;  # torch.sum(A,dim=1)
#
#     d = 0.5 * (torch.sum(vcol / mcol) + torch.sum(vrow / mrow))
#
#     return d
#
#
# def traceDist(X, Y, W=1):
#     A = utils.getDistMatrix(X, Y, W);
#     d = torch.trace(A)
#     # d1,_ = torch.max(A,dim=1)
#     # d0,_ = torch.max(A,dim=0)
#     # D1   = torch.diag(1./d1);
#     # D0   = torch.diag(1./d0);
#     # d = 0.5*(torch.trace(D1@A) + torch.trace(A@D0));
#
#     return d
#
#
# def expDist(X, Y, W=1):
#     A = utils.getDistMatrix(X, Y, W)
#     with torch.no_grad():
#         sigma = 1 / torch.max(A.view(-1))
#     M = torch.exp(-sigma * A);
#     # d     = -torch.sum(torch.log(torch.sum(M,1)))
#     d = torch.sum(M.view(-1))
#
#     return d
# # #
# # # # #
# # # # # def probX(X, Xi, sigma):
# # # #     n = X.shape
# # # #     C = getDistMatrix(X, Xi)
# # # #     A = torch.exp(-C / sigma)
# # # #     p = torch.sum(A, 0)
# # # #     q = torch.sum(A, 1)
# # # #     return p / n[0], q / n[0]
# # #
# # #
# def sinkHorn(X, Y):
#     '''
#     Paper?
#     :param X:
#     :param Y:
#     :return:
#     '''
#     M = utils.getDistMatrix(X, Y)
#     # M = M/torch.max(M.view(-1))
#     n = M.shape
#     C = torch.ones(n[0]).to(X.device)
#     with torch.no_grad():
#         sigma = 1 / torch.max(M)
#
#     K = torch.exp(-sigma * M).to(X.device)
#     # print(K)
#     U = (torch.ones(n[0]) / n[0]).to(X.device)
#     for i in range(20):
#         Uold = U
#         U = 1 / (K @ (C / (K.t() @ U)))
#         # if torch.norm(Uold-U)/torch.norm(U) < 1e-5:
#         #    break
#
#     V = C / (K.t() @ U)
#     d = torch.sum(torch.sum(U * ((K * M) * V)))
#
#     return d

#
#
# def getProbDist(X, Y):
#     '''paper?'''
#     '''Consider changing this as it might be unstable in the current way it is written'''
#     '''Note that it is also unstable in the eps, this one is not large enough, but it is hard to say a good rule for what it should be.'''
#     eps = 1e-1
#     nx = X.shape[0]
#     ny = Y.shape[0]
#
#     X = X.view(nx, -1)
#     Y = Y.view(ny, -1)
#
#     sX = torch.sum(X ** 2, 1);
#     sY = torch.sum(Y ** 2, 1);
#
#     CXX = sX.unsqueeze(1) + sX.unsqueeze(0) - 2 * X @ X.t()
#     CXX = torch.sqrt(CXX + eps)
#
#     CYY = sY.unsqueeze(1) + sY.unsqueeze(0) - 2 * Y @ Y.t()
#     CYY = torch.sqrt(CYY + eps)
#
#     CXY = sX.unsqueeze(1) + sY.unsqueeze(0) - 2 * X @ Y.t()
#     CXY = torch.sqrt(CXY + eps)
#
#     D = (nx * ny) / (nx + ny) * (2.0 / (nx * ny) * torch.sum(CXY)
#                                  - 1.0 / nx ** 2 * (torch.sum(CXX)) - 1.0 / ny ** 2 * (torch.sum(CYY)));
#
#     return D / (nx + ny)
# #

