from patsms.gemoptics import uv, asphere, dasphere, ddasphere, d_to_n, dsr, dsx
import numpy as np
import numpy.ma as ma
from scipy.optimize import fsolve, newton, minimize, basinhopping
import collections
import matplotlib.pyplot as plt
from numba import jit


@jit
def oas(x, A):
    """Asphere for optimizations"""
    kappa = A[2]+1.
    R = A[1]
    a = np.sqrt(1.-kappa*x*x/(R*R))
    rtn = A[0]+x*x/((a+1.)*R)
    for k in range(3, A.size):
        rtn = rtn+A[k]*x**(2*(k-2))
    return rtn


@jit
def odas(x, A):
    """First derivative asphere for optimizations"""
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
    """Second derivative asphere for optimizations"""
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
    """Asphere for solutions"""
    return (m*(x-x0)+y0-oas(x, A))


def dasi(x, A, m, x0, y0):
    """First derivative for solutions"""
    return (m-odas(x, A))


def ddasi(x, A, m, x0, y0):
    """Second derivative for solutions"""
    return (-oddas(x, A))


def inOutPhase(rxs, ths, ns, R, X, uo=1., ux=1.):
    """Ray tracing with phase space output
    rxs -- array of positions
    ths -- array of angles
    ns -- array of refractive indices
    R -- Refractive element parameters
    X -- Reflexive element parameters
    uo -- half-width of max output angle
    ux -- half-width of max mirror width
    """
    rtn = np.zeros((rxs.size, ths.size, 4))
    if X[2]+1. > 0:
        uxb = np.sqrt(X[1]**2/(1.+X[2]))
        if ux >= uxb:
            ux = uxb-np.finfo(type(uxb)).eps
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
            if yr < 0:  # Penalize if lens drops below image plane
                rtn[k1, k2, :] = np.nan
                continue
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
                    print(np.array([xxlast, ds[1]/ds[0], tx, yxmax]))
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
            xxlast = xx
            rtn[k1, k2, :] = np.array([xr, -ns[0]*di[0], xo, ns[1]*ds[0]])
    return ma.masked_where(~np.isfinite(rtn), rtn)


def phaseFill(iophs, sym=True):
    """Phase Filling factor metric
    iophs -- result from inOutPhase
    sym -- If only half of the angles were used
    """
    if np.all(iophs.mask):
        return 0.
    Ai = np.sum(np.diff(iophs[:, :-1, 0], axis=0) *
                np.diff(iophs[:-1, :, 1], axis=1))
    if sym:
        Ao = 2.*np.abs(iophs[:, :, 2]).max()*np.abs(iophs[:, :, 3]).max()
    else:
        Ao = ((iophs[:, :, 2].max()-iophs[:, :, 2].min()) *
              (iophs[:, :, 3].max()-iophs[:, :, 3].min()))
    try:
        if Ao <= 0:
            rtn = 0.
        else:
            rtn = np.min(np.array([1., np.abs(Ai/Ao)]))
    except:
        rtn = 0.
    if np.isfinite(rtn):
        return rtn
    else:
        return 0.


def phaseOrient(iophs):
    if np.all(iophs.mask):
        return 0.
    ks = np.argmin(np.abs(iophs[:, :, 2] -
                          np.atleast_2d(iophs[:, :, 2].mean(axis=0))), axis=0)
    r0 = np.transpose(
        np.atleast_3d(
            ma.vstack([iophs[k2, k1, [2, 3]] for k1, k2 in enumerate(ks)])),
        (2, 0, 1))
    ado = np.abs(iophs[:, :, [2, 3]]-r0)
    return (np.sum(ado[:, :, 1]) /
            np.sum(np.sqrt(np.sum(np.power(ado, 2), axis=2))))


def inPhaseUse(iophs):
    """Number of fraction of rays that were dropped"""
    return ((~iophs[:, :, 0].mask).sum()/iophs[:, :, 0].size)


def spotSize(iophs):
    """1 sigma method for spot size"""
    if np.all(iophs.mask):
        return iophs[:, :, 2].size
    rtn = (iophs[:, :, 2].std(axis=0)).sum()/iophs.shape[1]
    if np.isfinite(rtn):
        return rtn
    else:
        return iophs[:, :, 2].size


def spotSizeWeighted(iophs, ths):
    """Angle weighted method for spot size"""
    if np.all(iophs.mask):
        return (iophs[:, :, 2].size*np.cosh(2.*ths/ths.max()))
    rtn = (iophs[:, :, 2].std(axis=0) *
           (np.cosh(2.*ths/ths.max())) /
           ((~iophs[:, :, 2].mask).sum(axis=0)+1.)).sum()
    if np.isfinite(rtn):
        return rtn
    else:
        return (iophs[:, :, 2].size*np.cosh(2.*ths/ths.max()))


def spotSizeUniform(iophs):
    """Method for uniformity of spot size"""
    if np.all(iophs.mask):
        return iophs[:, :, 2].size
    tmp = (iophs[:, :, 2].std(axis=0))/((~iophs[:, :, 2].mask).sum(axis=0)+1.)
    rtn = tmp.max()*tmp.std()*100.
    if np.isfinite(rtn):
        return rtn
    else:
        return iophs[:, :, 2].size


def spotSize90(iophs):
    """90% of rays spot size"""
    if np.all(iophs.mask):
        return iophs[:, :, 2].size
    rtn = 0.
    for k in range(iophs.shape[1]):
        xs = ma.compressed(iophs[:, k, 2]).copy()
        n = int((iophs.shape[0]-1)*(0.1-1.+xs.size/iophs.shape[0]))
        if xs.size <= 1:
            rtn = rtn + iophs.shape[0]
        elif xs.size < 2*n or n < 1:
            rtn = rtn + (xs.max()-xs.min())
        else:
            xs.sort()
            rtn = rtn + (xs[-n:]-xs[:n]).min()

    if np.isfinite(rtn):
        return rtn
    else:
        return iophs[:, :, 2].size


def posOrient(iophs):
    if np.all(iophs.mask):
        return 0.
    ks = np.argmin(np.abs(iophs[:, :, 3] -
                          np.atleast_2d(iophs[:, :, 3].mean(axis=1))), axis=1)
    r0 = np.transpose(
        np.atleast_3d(
            ma.vstack([iophs[k1, k2, [2, 3]] for k1, k2 in enumerate(ks)])),
        (0, 2, 1))
    ado = np.abs(iophs[:, :, [2, 3]]-r0)
    return (np.sum(ado[:, :, 0]) /
            np.sum(np.sqrt(np.sum(np.power(ado, 2), axis=2))))


def optScale(A0, NR):
    """Scaling method for parameters"""
    A = A0.copy()
    A[:NR] = A0[:NR]*np.hstack((0.1*np.ones(3), np.arange(1, NR-2)*2.))
    A[NR:] = A0[NR:]*np.hstack((0.1*np.ones(3), np.arange(1, NR-2)*2.))
    return A


def optUnScale(A0, NR):
    """Remove parameter scale"""
    A = A0.copy()
    A[:NR] = A0[:NR]/np.hstack((0.1*np.ones(3), np.arange(1, NR-2)*2.))
    A[NR:] = A0[NR:]/np.hstack((0.1*np.ones(3), np.arange(1, NR-2)*2.))
    return A


def optASRXFit(A, rxs, ths, ns, NR, scale):
    """Objective function
    A -- All RX optic parameters
    rxs -- positions for simulation
    ths -- angles for simulation
    ns -- refractive indices
    NR -- Number of parameters for optic
    scale -- Boolean for parameter scaling
    """
    if scale:
        A0 = optUnScale(A, NR)
    else:
        A0 = A.copy()
    R = A0[:NR]
    X = A0[NR:]
    print(",".join(["%e" % x for x in R]))
    print(",".join(["%e" % x for x in X]))
    iop = inOutPhase(rxs, ths, ns, R, X, uo=2.)
    # return (1.-phaseFill(iop))*spotSizeUniform(iop)*(1.-inPhaseUse(iop))
    # return (1.-phaseFill(iop))*spotSizeWeighted(iop, ths)*(1.-inPhaseUse(iop))
    # return (1.-phaseFill(iop))*spotSize(iop)*(1.-inPhaseUse(iop))
    return (1.-phaseFill(iop))*spotSize90(iop)*(1.-inPhaseUse(iop))
    # return spotSizeWeighted(iop, ths)


def optASRX(R0, X0, ns, betam, Nx=51, Np=51, method='L-BFGS-B', scale=True):
    """Call optimization
    R0 -- Initial lens parameters (Thickness, radius, conic, aspheres)
    X0 -- Initial mirror parameters
    ns -- refractive indicies ([Source index, optic index])
    Nx -- Number of positions to launch rays
    Np -- Number of angles (phases) to launch rays
    method -- Optimization algorithm
    scale -- Scale parameters (Removes the multiple effect from the power
             in a derivative)
    """
    NR = R0.size
    if scale:
        A0 = optScale(np.hstack((R0, X0)), NR)
    else:
        A0 = np.hstack((R0, X0)).copy()
    bnds = [(None, None) for x in A0]
    bnds[0] = (0.01, 5.)
    bnds[NR] = (-5., -0.01)
    bnds[1] = (-20., -0.01)
    bnds[NR+1] = (0.01, 20.)
    rxs = np.linspace(-1., 1., Nx)
    ths = np.arcsin(np.linspace(0., np.sin(betam), Np))
    o = minimize(optASRXFit, A0, args=(rxs, ths, ns, NR, scale),
                 bounds=bnds, method=method)
    if scale:
        A = optUnScale(o.x, NR)
    else:
        A = o.x.copy()
    R0 = A[:NR]
    X0 = A[NR:]
    return (R0, X0, o)


def bhASRX(R0, X0, ns, betam, Nx=101, Np=101, method='L-BFGS-B'):
    """Basin hopping"""
    A0 = np.hstack((R0, X0))
    bnds = [(None, None) for x in A0]
    NR = R0.size
    bnds[0] = (0.01, 5.)
    bnds[NR] = (-5., -0.01)
    bnds[1] = (-200., -0.01)
    bnds[NR+1] = (0.01, 200)
    rxs = np.linspace(-1., 1., Nx)
    ths = np.arcsin(np.linspace(0., np.sin(betam), Np))
    o = basinhopping(optASRXFit, A0,
                     minimizer_kwargs={
                         'args': (rxs, ths, ns, NR),
                         'bounds': bnds,
                         'method': method})
    R0 = o.x[:NR]
    X0 = o.x[NR:]
    return (R0, X0, o)


def plotRX(xs, R, X, opt='k'):
    """Plots the lens"""
    return plt.plot(
        xs,
        np.vstack([np.array([oas(x, R), oas(x, X)]) for x in xs]),
        opt)


def plotPhase(iophs):
    """Plots the output phase"""
    return plt.pcolormesh(iophs[:, :, 2], iophs[:, :, 3], iophs[:, :, 1])


def plotPhaseUse(iophs):
    """Plots the input phase"""
    return plt.pcolormesh(iophs[:, :, 0], iophs[:, :, 1], iophs[:, :, 1].mask)
