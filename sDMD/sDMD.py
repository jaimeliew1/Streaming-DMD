"""
Streaming DMD
A Python implementation of the streaming dynamic mode decomposition algorithm
described in the paper Liew, J. et al. "Streaming dynamic mode decomposition for
short-term forecasting in wind farms"

The algorithm performs a continuously updating dynamic mode decomposition as new
data is made available.

The equations referred to in this paper correspond to the equations in:
    Liew, J. et al. "Streaming dynamic mode decomposition for
    short-term forecasting in wind farms"

Author: Jaime Liew
License: MIT (see LICENSE.txt)
Version: 1.0
Email: jyli@dtu.dk
"""
import numpy as np


def hankel_transform(X, s):
    """
    stacks the snapshots, X, so that each new snapshot contains the previous
    s snapshots.
    args:
        X (2D array): n by m matrix.
        s (int): stack size
    returns:
        Xout (2D array): n - (k-1) by k*m matrix.
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if s == 1:
        return X
    l, m = X.shape
    w = m - (s - 1)
    out = np.zeros([l * s, w])

    for i in range(s):
        row = X[:, m - i - w : m - i]
        out[i * l : (i + 1) * l, :] = row

    return out


def truncatedSVD(X, r):
    """
    Computes the truncated singular value decomposition (SVD)
    args:
        X (2d array): Matrix to perform SVD on.
        rank (int or float): rank parameter of the svd. If a positive integer,
        truncates to the largest r singular values. If a float such that 0 < r < 1,
        the rank is the number of singular values needed to reach the energy
        specified in r. If -1, no truncation is performed.
    """

    U, S, V = np.linalg.svd(X, full_matrices=False)
    V = V.conj().T
    if r >= 1:
        rank = min(r, U.shape[1])

    elif 0 < r < 1:
        cumulative_energy = np.cumsum(S ** 2 / np.sum(S ** 2))
        rank = np.searchsorted(cumulative_energy, r) + 1

    U_r = U[:, :rank]
    S_r = S[:rank]
    V_r = V[:, :rank]

    return U_r, S_r, V_r


class sDMD_base(object):
    """
    Calculate DMD in streaming mode. 
    """

    def __init__(self, X, Y, rmin, rmax, thres=0.2, halflife=None):

        self.rmin = rmin
        self.rmax = rmax
        self.thres = thres
        self.halflife = halflife
        self.rho = 1 if halflife is None else 2 ** (-1 / halflife)

        # Eq. (2) - truncated SVD
        self.Ux, _, _ = truncatedSVD(X, rmin)
        self.Uy, _, _ = truncatedSVD(Y, rmin)

        # Eq. (3) - Mapping of input vector to reduced order space.
        X_tild = self.Ux.T @ X
        # Eq. (4) - Mapping of out vector to reduced order space.
        Y_tild = self.Uy.T @ Y

        # Eq (9) - Decomposition of transition matrix into the product of Q and Pinvx.
        self.Q = Y_tild @ X_tild.T
        self.Pinvx = X_tild @ X_tild.T
        self.Pinvy = Y_tild @ Y_tild.T

    def update(self, x, y):
        x, y = x.reshape([-1, 1]), y.reshape([-1, 1])
        status = 0

        normx = np.linalg.norm(x, ord=2, axis=0)
        normy = np.linalg.norm(y, ord=2, axis=0)

        xtilde = self.Ux.T @ x
        ytilde = self.Uy.T @ y

        # Numerator of Eq. (14) - projection error.
        ex = x - self.Ux @ xtilde
        ey = y - self.Uy @ ytilde

        #### STEP 1 - BASIS EXPANSION ####
        # Table 1: Rank augmentation of Ux
        if np.linalg.norm(ex, ord=2, axis=0) / normx > self.thres:

            u_new = ex / np.linalg.norm(ex, ord=2, axis=0)
            self.Ux = np.hstack([self.Ux, (u_new).reshape([-1, 1])])

            self.Pinvx = np.hstack([self.Pinvx, np.zeros([self.Pinvx.shape[0], 1])])
            self.Pinvx = np.vstack([self.Pinvx, np.zeros([1, self.Pinvx.shape[1]])])
            self.Q = np.hstack([self.Q, np.zeros([self.Q.shape[0], 1])])
            status = 1

        # Table 1: Rank augmentation of Uy
        if np.linalg.norm(ey, ord=2, axis=0) / normy > self.thres:
            u_new = ey / np.linalg.norm(ey, ord=2, axis=0)
            self.Uy = np.hstack([self.Uy, (u_new).reshape([-1, 1])])

            self.Pinvy = np.hstack([self.Pinvy, np.zeros([self.Pinvy.shape[0], 1])])
            self.Pinvy = np.vstack([self.Pinvy, np.zeros([1, self.Pinvy.shape[1]])])
            self.Q = np.vstack([self.Q, np.zeros([1, self.Q.shape[1]])])
            status = -1

        #### STEP 2 - BASIS POD COMPRESSION ####
        # Table 1: Rank reduction of Ux
        if self.Ux.shape[1] > self.rmax:
            eigval, eigvec = np.linalg.eig(self.Pinvx)
            indx = np.argsort(-eigval)
            eigval = eigval[indx]
            qx = eigvec[:, indx[: self.rmin]]

            self.Ux = self.Ux @ qx
            self.Q = self.Q @ qx
            self.Pinvx = np.diag(eigval[: self.rmin])
            status = 2

        # Table 1: Rank reduction of Uy
        if self.Uy.shape[1] > self.rmax:
            eigval, eigvec = np.linalg.eig(self.Pinvy)
            indx = np.argsort(-eigval)
            eigval = -np.sort(-eigval)
            qy = eigvec[:, indx[: self.rmin]]

            self.Uy = self.Uy @ qy
            self.Q = qy.T @ self.Q
            self.Pinvy = np.diag(eigval[: self.rmin])
            status = -2

        #### STEP 3 - REGRESSION UPDATE ####
        xtilde = self.Ux.T @ x
        ytilde = self.Uy.T @ y

        # Eq. (10), (11), and (12) - Rank 1 update of DMD matrices.
        self.Q = self.rho * self.Q + ytilde @ xtilde.T
        self.Pinvx = self.rho * self.Pinvx + xtilde @ xtilde.T
        self.Pinvy = self.rho * self.Pinvy + ytilde @ ytilde.T

        return status

    @property
    def rank(self):
        return self.U.shape[1]

    @property
    def A(self):
        """
        Computes the reduced order transition matrix from xtilde to ytilde.
        """
        # return self.Ux.T @ self.Uy @ self.Q @ np.linalg.pinv(self.Pinvx)
        return self.Q @ np.linalg.pinv(self.Pinvx)

    @property
    def modes(self):
        """
        Compute DMD modes and eigenvalues. The first output is the eigenmode
        matrix where the columns are eigenvectors. The second output are the
        discrete time eigenvalues. Assumes the input and output space are the
        same.
        """

        eigvals, eigvecK = np.linalg.eig(self.Ux.T @ self.Uy @ self.A)
        modes = self.Ux @ eigvecK

        return modes, eigvals


class sDMD(sDMD_base):
    """
    A wrapper class for sDMD_base. Manages the streaming data inputs and
    transforms the data to correctly represent the additional channels and delay
    states as described in Section 2.3 - State augmentation.
    """

    def __init__(self, X, rmin, rmax, Y=None, f=1, s=1, **kwargs):
        self.s = s
        self.f = f
        if Y is None:
            Y = X
        self.rolling_x = X[:, -(s + f - 1) :]

        X_hank = hankel_transform(X, s)

        X_hank = X_hank[:, :-f]
        Y_init = Y[:, f + s - 1 :]

        super().__init__(X_hank, Y_init, rmin, rmax, **kwargs)

    def update(self, x_in, y_in=None):
        if y_in is None:
            y_in = x_in
        x_in = x_in.reshape(-1, 1)
        y_in = y_in.reshape(-1, 1)

        self.rolling_x = np.hstack([self.rolling_x, x_in])

        X_hank = hankel_transform(self.rolling_x, self.s)

        self.x_buff = X_hank[:, -1]
        xnew = X_hank[:, 0]
        ynew = y_in

        status = super().update(xnew, ynew)

        self.rolling_x = self.rolling_x[:, 1:]
        return status
