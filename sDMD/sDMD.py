import numpy as np
from scipy import linalg as splinalg
from tqdm import tqdm, trange


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


def compute_svd(X, r):
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
    def __init__(
        self, X, Y, rmin, rmax, thres=0.2, halflife=None, chunksize=1, farm_cost=False
    ):

        self.rmin = rmin
        self.rmax = rmax
        self.thres = thres
        self.chunksize = chunksize
        self.farm_cost = farm_cost
        self.x_list = []
        self.y_list = []

        self.halflife = halflife
        self.rho = 1 if halflife is None else 2 ** (-1 / halflife)
        self.U, _, _ = compute_svd(X, rmin)

        X_tild = self.U.T @ X
        Y_tild = self.U.T @ Y

        self.Pinv = X_tild @ X_tild.T
        self.Q = Y_tild @ X_tild.T

    def update(self, xin, yin):
        self.x_list.append(xin)
        self.y_list.append(yin)

        status = 0
        if len(self.x_list) >= self.chunksize:
            Y = np.array(self.y_list).T

            if self.farm_cost:
                performance_measure = self.not_performing2(Y)
            else:
                performance_measure = self.not_performing(Y)
            if performance_measure > self.thres:
                status = self.update_basis(Y)

            for x, y in zip(self.x_list, self.y_list):
                self.update_single(x, y)
            self.x_list.clear()
            self.y_list.clear()
            return status

        return 0

    def not_performing(self, Y):

        Y_tild = self.U.T @ Y
        resid = Y - self.U @ Y_tild
        p = np.linalg.norm(resid, ord=2, axis=0)
        ref = np.linalg.norm(Y, ord=2, axis=0)

        ind = (p / ref).argmax()
        p_max = (p / ref)[ind]

        return p_max > self.thres

    def not_performing2(self, Y):

        Y_tild = self.U.T @ Y
        resid = (Y - self.U @ Y_tild)[-1, :]

        ind = abs(resid / Y[-1, :]).argmax()
        p_max = abs(resid / Y[-1, :])[ind]

        return p_max

    def update_basis(self, Y):
        status = 1
        if self.rank >= self.rmax:

            eigvecs = self.leading_eigvecs
            self.Pinv = eigvecs.T @ self.Pinv @ eigvecs
            self.Q = eigvecs.T @ self.Q @ eigvecs
            self.U = self.U @ eigvecs
            status = 2

        Y_tild = self.U.T @ Y
        resid = Y - self.U @ Y_tild

        p = np.linalg.norm(resid, ord=2, axis=0)
        ref = np.linalg.norm(Y, ord=2, axis=0)

        ind = (p / ref).argmax()
        p_max = (p / ref)[ind]
        u_new = resid[:, ind].reshape([-1, 1])
        for u in self.U.T:
            u = u.reshape([-1, 1])
            R = (u_new.T @ u) / (u.T @ u)

            u_new = u_new - R * u
        u_new = u_new / np.linalg.norm(u_new, ord=2)
        U = np.hstack([self.U, (u_new).reshape([-1, 1])])
        U = splinalg.orth(U)

        self.U = U
        new_shape = [x + 1 for x in self.Pinv.shape]
        tmp = np.zeros(new_shape)
        tmp[:-1, :-1] = self.Pinv
        self.Pinv = tmp
        tmp = np.zeros(new_shape)
        tmp[:-1, :-1] = self.Q
        self.Q = tmp
        return status

    def update_single(self, x, y):
        status = 0
        x, y = x.reshape([-1, 1]), y.reshape([-1, 1])
        x_tild = self.U.T @ x
        y_tild = self.U.T @ y

        # check residual of projection of y. Y is chosen as it sees the newest
        # observation.
        resid = y - self.U @ y_tild
        p = np.linalg.norm(resid, ord=2)

        self.Pinv = self.rho * self.Pinv + x_tild @ x_tild.T
        self.Q = self.rho * self.Q + y_tild @ x_tild.T

    @property
    def leading_eigvecs(self):
        vals, vecs = self.modes
        out = []
        for i in range(len(vals)):
            if i == 0 or vals[i] != vals[i - 1]:
                out.append(vecs[i])
            if len(out) == self.rmin:
                break

        out = np.column_stack(out).real
        out = splinalg.orth(out)
        return out

    @property
    def rank(self):
        return self.U.shape[1]

    @property
    def A(self):
        # do decomposition instead of inv
        return self.Q @ np.linalg.inv(self.Pinv)

    @property
    def modes(self):
        eigvals, eigvecs = splinalg.eig(self.A)
        idx = abs(eigvals).argsort()[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        return eigvals, eigvecs


class sDMD(sDMD_base):
    def __init__(self, X, rmin, rmax, f=1, s=1, **kwargs):
        self.s = s
        self.f = f

        self.rolling_x = X[:, -(s + f - 1) :]

        X_hank = hankel_transform(X, s)
        X_hank, Y_hank = X_hank[:, :-f], X_hank[:, f:]

        super().__init__(X_hank, Y_hank, rmin, rmax, **kwargs)

    def update(self, x_in):

        x_in = x_in.reshape(-1, 1)
        self.rolling_x = np.hstack([self.rolling_x, x_in])
        X_hank = hankel_transform(self.rolling_x, self.s)
        self.x_buff = X_hank[:, -1]
        xnew = X_hank[:, 0]
        ynew = X_hank[:, -1]

        status = super().update(xnew, ynew)

        self.rolling_x = self.rolling_x[:, 1:]
        return status
