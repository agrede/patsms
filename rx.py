from patsms.gemoptics import uv, asphere, dasphere, ddasphere, d_to_n, dsr, dsx
import numpy as np
import numpy.ma as ma
from scipy.optimize import fsolve, newton, minimize
import collections
from numba import jit


@jit
def oas(x, A):
    kappa = A[2]+1.
    R = A[1]
    a = np.sqrt(1.-kappa*x*x/(R*R))
    rtn = A[0]+x*x/((a+1.)*R)
    for k in range(3, A.size):
        rtn = rtn+A[k]*x**(2*(k-2))
    return rtn


@jit
def odas(x, A):
    kappa = A[2]+1.
    R = A[1]
    a = np.sqrt(1.-kappa*x*x/(R*R))
    rtn = 2.*x/((a+1.)*R)+kappa*(x/R)**3/(a*(a+1.)**2)
    for k in range(3, A.size):
        pwr = 2.*(k-2.)
        rtn = rtn+pwr*A[k]*(x**(2*(k-2)-1))
    return rtn


@jit
def oddas(x, A):
    kappa = A[2]+1.
    R = A[1]
    a = np.sqrt(1.-kappa*x*x/(R*R))
    b = 1.+a
    rtn = (2./(b*R)+5.*kappa*x**2/(b**2*a*R**3) +
           2.*kappa**2*x**4/(b**2*a**3*R**5))
    if (A.size > 3):
        rtn = rtn+2.*A[3]
        for k in range(4, A.size):
            pwr = 2.*(k-2.)
            rtn = rtn+(pwr-1.)*pwr*A[k]*(x**(2*(k-2)-2))
    return rtn


def asi(x, A, m, x0, y0):
    return (m*(x-x0)+y0-oas(x, A))


def dasi(x, A, m, x0, y0):
    return (m-odas(x, A))


def ddasi(x, A, m, x0, y0):
    return (-oddas(x, A))


def inOutPhase(rxs, ths, ns, R, X, uo=1., ux=1.):
    rtn = np.zeros((rxs.size, ths.size, 4))
    # print("R: "+", ".join(["%0.5e" % x for x in R]))
    # print("X: "+", ".join(["%0.5e" % x for x in X]))
    yxmax = oas(ux, X)

    for k2, th in enumerate(ths):
        di = np.array([np.sin(th), -np.cos(th)])
        if np.abs(di[0]) > 0:
            xxlast = np.sign(di[0])*(
                np.sign(di[0])*np.array([rxs[0], -ux])).max()
        else:
            xxlast = rxs[0]
        for k1, xr in enumerate(rxs):
            yr = oas(xr, R)
            dn = d_to_n(odas(xr, R))

            if dn[0]*np.sign(di[0]) > -di[1]:  # Ray coming from inside
                rtn[k1, k2, :] = np.nan
                continue
            ds = dsr(di, dn, ns)
            if ds[1] >= 0:
                rtn[k1, k2, :] = np.nan
                continue
            tx = (yxmax-yr)*ds[0]/ds[1]+xr
            if np.abs(tx) > ux:
                rtn[k1, k2, :] = np.nan
                continue
            if oas(tx, X) > yxmax:  # Penalize if mirror is convex
                rtn[k1, k2, :] = np.nan
                continue
            if np.abs(ds[0]) > 0:
                xxlast = np.sign(ds[0])*(
                    np.sign(ds[0])*np.array([xxlast, tx])).max()
                try:
                    xx = newton(asi, xxlast, args=(X, ds[1]/ds[0], tx, yxmax),
                                fprime=dasi, fprime2=ddasi)
                except RuntimeError:
                    xx = fsolve(asi, xxlast, args=(X, ds[1]/ds[0], tx, yxmax),
                                fprime=dasi)[0]
            else:
                xxlast = tx
                xx = tx
            yx = oas(xx, X)
            tmp = odas(xx, X)
            ds = dsx(ds, d_to_n(tmp))
            if np.abs(ds[1]) > 0:
                xo = -yx*ds[0]/ds[1]+xx
            else:
                xo = xx
            if np.abs(xo) > uo:
                rtn[k1, k2, :] = np.nan
                continue
            rtn[k1, k2, :] = np.array([xr, -ns[0]*di[0], xo, ns[1]*ds[0]])
    return ma.masked_where(np.isnan(rtn), rtn)


def phaseFill(iophs, sym=True):
    if np.all(iophs.mask):
        return 0.
    Ai = np.sum(np.diff(iophs[:, :-1, 0], axis=0) *
                np.diff(iophs[:-1, :, 1], axis=1))
    if sym:
        Ao = 2.*np.abs(iophs[:, :, 2]).max()*np.abs(iophs[:, :, 3]).max()
    else:
        Ao = ((iophs[:, :, 2].max()-iophs[:, :, 2].min()) *
              (iophs[:, :, 3].max()-iophs[:, :, 3].min()))
    if Ao <= 0:
        return 0.
    else:
        return np.min(np.array([1., np.abs(Ai/Ao)]))  # Catch greater than 1 error


def optASRXFit(A, rxs, ths, ns, NR):
    R = A[:NR]
    X = A[NR:]
    return (1.-phaseFill(inOutPhase(rxs, ths, ns, R, X)))


def optASRX(R0, X0, ns, betam, Nx=101, Np=101, method='L-BFGS-B'):
    A0 = np.hstack((R0, X0))
    bnds = [(None, None) for x in A0]
    NR = R0.size
    bnds[0] = (0.01, 5.)
    bnds[NR] = (-5., -0.01)
    bnds[1] = (-20., -0.01)
    bnds[NR+1] = (0.01, 20)
    rxs = np.linspace(-1., 1., Nx)
    ths = np.arcsin(np.linspace(0., np.sin(betam), Np))
    return minimize(optASRXFit, A0, args=(rxs, ths, ns, NR),
                    bounds=bnds, method=method)
