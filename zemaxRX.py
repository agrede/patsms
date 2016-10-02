# Copyright (C) 2016 Alex J. Grede
# GPL v3, See LICENSE.txt for details
# This function is part of PATSMS (https://github.com/agrede/patsms)
import numpy as np
import numpy.ma as ma
from scipy.optimize import minimize
import pyzdde.zdde as pyz
import pyzdde.arraytrace as at
import os


def optScale(A0, NR):
    A = A0.copy()
    A[:NR] = A0[:NR]*np.hstack((0.1*np.ones(3), np.arange(1, NR-2)*2.))
    A[NR:] = A0[NR:]*np.hstack((0.1*np.ones(3), np.arange(1, NR-2)*2.))
    return A


def optUnScale(A0, NR):
    A = A0.copy()
    A[:NR] = A0[:NR]/np.hstack((0.1*np.ones(3), np.arange(1, NR-2)*2.))
    A[NR:] = A0[NR:]/np.hstack((0.1*np.ones(3), np.arange(1, NR-2)*2.))
    return A


class ZERX():
    """
    Optimize RX concentrators with ZEMAX
    """

    ln = None
    rd = None
    Omega = 1.
    NperBeta = 0
    n = 1.
    X = None
    Y = None
    pys = None
    pxs = None
    varOff = 1.
    scale = True
    R0 = None
    X0 = None
    Rf = None
    Xf = None
    o = None

    def __init__(self, R0, X0, n, scale=True, ln=None):
        if (ln is None):
            self.ln = pyz.createLink()
        else:
            self.ln = ln
        self.n = n
        self.scale = scale
        self.R0 = R0.copy()
        self.X0 = X0.copy()
        self.Rf = R0.copy()
        self.Xf = X0.copy()
        filename = os.path.abspath('./SequentialForProgram.zmx')
        self.ln.zLoadFile(filename)
        self.ln.zGetUpdate()
        if not self.ln.zPushLensPermission():
            return None
        self.ln.zPushLens(1)
        self.updateRX(R0, X0)

    def updateRX(self, R, X):
        # Radius
        self.ln.zSetSurfaceData(1, 2, 1./R[1])
        self.ln.zSetSurfaceData(4, 2, 1./X[1])
        # Conic
        self.ln.zSetSurfaceData(1, 6, R[2])
        self.ln.zSetSurfaceData(4, 6, X[2])
        # Asphere
        for k, v in enumerate(R[3:]):
            self.ln.zSetSurfaceParameter(1, k+1, v)
        for k, v in enumerate(X[3:]):
            self.ln.zSetSurfaceParameter(4, k+1, v)
        # Thickness
        self.ln.zSetSurfaceData(3, 3, X[0])
        self.ln.zPushLens(1)
        self.ln.zGetRefresh()
        thick = self.ln.zGetSurfaceData(1, 3)
        self.ln.zSetSurfaceData(2, 3, R[0]-thick)
        self.ln.zPushLens(1)
        self.ln.zGetRefresh()
        return True

    def setSpace(self, alpha, beta, Nxy, Na, Nb):
        xs = np.atleast_2d(np.linspace(-1., 1., Nxy))
        ys = np.atleast_2d(np.linspace(-1., 1., Nxy)).T
        self.pxs = np.linspace(-np.sin(alpha), np.sin(alpha), Na)
        self.pys = np.linspace(0., np.sin(beta), Nb)
        R = (np.sqrt(np.power(xs, 2)+np.power(ys, 2)))
        self.X = ma.masked_where(R > 1., xs*np.ones(ys.shape))
        self.Y = ma.masked_where(R > 1., ys*np.ones(xs.shape))
        self.Omega = 4.*np.pi*self.pys[-1]*self.pxs[-1]
        self.NperBeta = np.sum(~self.X.mask)*Na
        self.NBeta = Nb
        return True

    def setRays(self):
        z = self.ln.zGetSurfaceData(1, 3)
        self.rd = at.getRayDataArray(self.NperBeta*self.NBeta,
                                     tType=1, startSurf=0)
        k = 0
        for py in self.pys:
            for px in self.pxs:
                pz = np.sqrt(1.-py**2+px**2)
                for x, y in zip(self.X.compressed(),
                                self.Y.compressed()):
                    self.rd[k].z = z
                    self.rd[k].x = x
                    self.rd[k].y = y
                    self.rd[k].l = px
                    self.rd[k].m = py
                    self.rd[k].n = pz
                    k = k+1
        return True

    def phaseFill(self):
        xmx = 0.
        ymx = 0.
        pxmx = 0.
        pymx = 0.
        Nerr = 0
        for k in range(1, len(self.rd)):
            if self.rd[k].error == 0:
                if np.abs(self.rd[k].x) > xmx:
                    xmx = np.abs(self.rd[k].x)
                if np.abs(self.rd[k].y) > ymx:
                    ymx = np.abs(self.rd[k].y)
                if np.abs(self.rd[k].l) > pxmx:
                    pxmx = np.abs(self.rd[k].l)
                if np.abs(self.rd[k].m) > pymx:
                    pymx = np.abs(self.rd[k].m)
            else:
                Nerr = Nerr + 1
        Ao = 16.*(self.n)**2*xmx*ymx*pxmx*pymx
        inPhaseUse = 1.-Nerr/(len(self.rd)-1)
        Ai = self.Omega*inPhaseUse
        return (Ai/Ao, inPhaseUse)

    def spotSize(self):
        ks = np.arange(self.NperBeta)
        varSum = np.array([0., 0.])
        for n in range(int((len(self.rd)-1)/self.NperBeta)):
            tks = ks+1+self.NperBeta*n
            pos = np.vstack(
                [
                    np.array([self.rd[k].x, self.rd[k].y])
                    for k in tks if self.rd[k].error == 0
                ])
            Ntr = np.sum([(self.rd[k].error == 0) for k in tks])
            var = np.power(pos.std(axis=0), 2)
            varSum = (varSum +
                      (1.+self.varOff)*var /
                      (self.varOff+Ntr/self.NperBeta))
        return (np.sqrt(varSum.sum()))

    def rxObjective(self, A):
        NR = self.R0.size
        if self.scale:
            A0 = optUnScale(A, NR)
        else:
            A0 = A.copy()
        R = A0[:NR]
        X = A0[NR:]
        self.updateRX(R, X)
        self.setRays()
        at.zArrayTrace(self.rd)
        PF, PIF = self.phaseFill()
        spt = self.spotSize()
        return (1.-PF)*(1.1-PIF)*spt

    def opt(self, method='L-BFGS-B'):
        NR = self.R0.size
        if self.scale:
            A0 = optScale(np.hstack((self.R0, self.X0)), NR)
        else:
            A0 = np.hstack((self.R0, self.X0)).copy()
        bnds = [(None, None) for x in A0]
        bnds[0] = (0.01, 5.)
        bnds[NR] = (0.01, 5.)
        bnds[1] = (0.01, 20.)
        bnds[NR+1] = (-20., -0.01)
        o = minimize(self.rxObjective, A0, bounds=bnds,
                     method=method)
        if self.scale:
            A = optUnScale(o.x, NR)
        else:
            A = o.x.copy()
        self.Rf = A[:NR]
        self.Xf = A[NR:]
        self.o = o.copy()
        return (self.Rf, self.Xf, self.o)
