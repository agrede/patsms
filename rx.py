from patsms.gemoptics import uv, asphere, dasphere, ddasphere, d_to_n, dsr, dsx
import numpy as np
import numpy.ma as ma
from scipy.optimize import newton, minimize
import collections


def oas(x, A):
    il = isinstance(x, collections.Iterable)
    if (not il):
        x = np.array(x)
    rtn = A[0] + asphere(x, A[1], A[2], A[3:])
    if (il):
        return rtn
    else:
        return rtn[0]


def odas(x, A):
    il = isinstance(x, collections.Iterable)
    if (not il):
        x = np.array(x)
    rtn = dasphere(x, A[1], A[2], A[3:])
    if (il):
        return rtn
    else:
        return rtn[0]


def oddas(x, A):
    il = isinstance(x, collections.Iterable)
    if (not il):
        x = np.array(x)
    rtn = ddasphere(x, A[1], A[2], A[3:])
    if (il):
        return rtn
    else:
        return rtn[0]


def inOutPhase(rxs, ths, ns, R, X, uo=1., ux=1.):
    rtn = np.zeros((rxs.size, ths.size, 4))
    yxmax = oas(ux, X)
    for k2, th in enumerate(ths):
        di = np.array([np.sin(th), -np.cos(th)])
        if np.abs(di[0]) > 0:
            xxlast = (np.sign(di[0])*np.array([rxs[0], -ux])).max()
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
                xxlast = (np.sign(ds[0])*np.array([xxlast, tx])).max()
                xx = newton(
                    lambda x: ds[1]/ds[0]*(x-tx)+yxmax-oas(x, X),
                    xxlast,
                    fprime=lambda x: ds[1]/ds[0]-odas(x, X),
                    fprime2=lambda x: -oddas(x, X))
            else:
                xxlast = tx
                xx = tx
            yx = oas(xx, X)
            ds = dsx(ds, d_to_n(odas(xx, X)))
            if np.abs(ds[1]) > 0:
                xo = -yx*ds[0]/ds[1]+xx
            else:
                xo = xx
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
    return np.min(np.array([1., np.abs(Ai/Ao)]))  # Catch greater than 1 error


def optASRXFit(A, rxs, ths, ns, NR):
    R = A[:NR]
    X = A[NR:]
    return (1.-phaseFill(inOutPhase(rxs, ths, ns, R, X)))


def optASRX(R0, X0, ns, betam, Nx=101, Np=101):
    A0 = np.hstack((R0, X0))
    NR = R0.size
    rxs = np.linspace(-1., 1., Nx)
    ths = np.arcsin(np.linspace(0., np.sin(betam), Np))
    return minimize(optASRXFit, A0, args=(rxs, ths, ns, NR))
