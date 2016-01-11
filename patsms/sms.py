from gemoptics import uv, dsr, dsx, dnr, dnx, opl, trace_stack, d_to_n
from numpy import sin, cos, pi, sqrt, array, vstack, hstack, ones, zeros, fliplr, inner, where, argsort, dstack
from numpy.linalg import norm
from scipy.optimize import fsolve
from scipy.interpolate import PiecewisePolynomial, PchipInterpolator


def wa(wr, wx, hx, ni, no, thetai):
    """Returns width of absorber
    wr -- width of refractive element
    wx -- width of reflective element
    hx -- height of reflective element (below absorber)
    ni -- index of incident rays
    no -- index of output rays
    """
    return fsolve(lambda wa: wr/wa*ni/no*sin(thetai) -
                  ((wx-wa)/(2.*sqrt(((wx-wa)/2.)**2+hx**2))),
                  wr*ni/no*sin(thetai))[0]


def solve_rx(ri, di, rf, n, l):
    """Returns position and normal of a mirror
    ri -- initial ray position
    di -- initial ray direction
    rf -- final ray position
    n -- index of refraction
    l -- optical path length
    """
    mi = fsolve(lambda x: l-n*(x+norm(rf-(ri+x*di))),
                (l/n-norm(rf-ri)))[0]
    rx = ri+mi*di
    nx = dnx(di, uv(rf-rx))
    return (rx, nx)


def solve_rr(rf, df, ri, di, ns, l):
    """Returns position and normal of refractive surface
    rf -- final ray position
    df -- final ray direction
    ri -- reference position for opl
    di -- initial ray direction
    ns -- indices of refraction
    l -- optical path length
    """
    mf = fsolve(lambda x: l+inner((rf-x*df)-ri, [1., -1.]*di) -
                ns[1]*x, l/ns[1])[0]
    rr = rf-mf*df
    nr = dnr(di, df, ns)
    return (rr, nr)


def sms_rx_inf_source(nr, nx, ystack, nstack, wr=1.,
                      ni=1., thetai=9.35e-3, Nmax=200, hx=0.):
    """Returns design rr, rx, dnr, dnx"""
    ystack = hstack((ystack, [0.]))
    ns = hstack((ni, nr, nstack, nx))
    rr = [array([wr/2., ystack[0]])]
    rx = [array([wr/2., -hx])]
    di = array([[sin(thetai), -cos(thetai)],
                [-sin(thetai), -cos(thetai)]])
    tmp = wa(wr, wr, hx, ns[0], ns[-1], thetai)
    ra = array([[-tmp/2., 0.], [tmp/2., 0.]])
    nr = [dnr(di[0, :], array([0., -1.]), ns[:2])]
    nx = [dnx(uv(rx[-1]-rr
                 [-1]), uv(ra[1, :]-rx[-1]))]
    l = [opl(vstack((rr[0],
                     vstack(((wr/2.*ones(ystack.size)), ystack)).T,
                     rx[0], ra[1, :])), hstack((ns[1:], ns[[-1]])))]
    k = 0
    while k < Nmax and rr[-1][0] >= 0. and rx[-1][0] >= 0.:
        k = k+1
        ds = dsr(di[1, :], nr[-1], ns[:2])
        rs, ds, dl = trace_stack(ystack, ns[1:], rr[-1], ds)
        trx, tnx = solve_rx(rs, ds, ra[0, :], ns[-1],
                            l[-1]+ns[0]*2.*rr[-1][0]*sin(thetai)-dl)
        rx.append(trx)
        nx.append(tnx)
        ds = dsx(uv(rx[-1]-ra[1, :]), nx[-1])
        rs, ds, dl = trace_stack(ystack[::-1], ns[:0:-1],
                                 rx[-1], ds)
        dl = dl + opl(vstack((ra[1, :], rx[-1])), array(ns[-1:]))
        trr, tnr = solve_rr(rs, -ds, rr[-1], di[0, :], ns[:2], l[-1]-dl)
        rr.append(trr)
        nr.append(tnr)
        l.append(dl+opl(vstack((rs, rr[-1])), array(ns[1:2])))

    return (vstack(rr), vstack(rx), vstack(nr), vstack(nx), hstack(l))


def fit_surf(r, n, sym=True):
    """Returns PiecewisePolynomial fit for surface
    r -- coordinates of surface
    n -- surface normal for each coordinate
    sym -- treat as symmetrical
    """
    if sym:
        kp = where(r[:, 0] >= 0)[0]
        x = hstack((-r[kp, 0], r[kp, 0]))
        ks = argsort(x)
        x = x[ks]
        y = vstack((
            hstack((r[kp, 1], r[kp, 1]))[ks],
            hstack((n[kp, 0]/n[kp, 1], -n[kp, 0]/n[kp, 1]))[ks])).T
    else:
        ks = argsort(r[:, 0])
        x = r[ks, 0]
        y = vstack((r[ks, 1], -n[ks, 0]/n[ks, 1])).T
    return PiecewisePolynomial(x, y)


def fit_surf_spline(r, n, sym=True):
    """Returns PchipInterpolator fit for surface
    r -- coordinates of surface
    n -- surface normal for each coordinate
    sym -- treat as symmetrical
    """
    if sym:
        kp = where(r[:, 0] >= 0)[0]
        x = hstack((-r[kp, 0], r[kp, 0]))
        ks = argsort(x)
        x = x[ks]
        y = hstack((r[kp, 1], r[kp, 1]))[ks]
    else:
        ks = argsort(r[:, 0])
        x = r[ks, 0]
        y = r[ks, 1]
    return PchipInterpolator(x, y)


def trace_rx(fr, fx, ystack, nstack, nr, nx, xs, thetai,
             wr=1., hx=0., ni=1., dfr=None, dfx=None):
    ystack = hstack((ystack, [0.]))
    di = array([sin(thetai), -cos(thetai)])
    ns = hstack((ni, nr, nstack, nx))
    rtn = zeros((3+ystack.size, 2, xs.size))
    if dfr is None:
        dfr = lambda x: fr.derivative(x)
    if dfx is None:
        dfx = lambda x: fx.derivative(x)
    for k, x in enumerate(xs):
        rr = array([x, fr(x)])
        ds = dsr(di, d_to_n(dfr(x)), ns[:2])
        tmp, ds, dl = trace_stack(ystack, ns[1:], rr, ds, return_all=True)
        if ds[0] == 0. or ds[1] >= 0.:
            next
        tx = (-hx-tmp[-1, 1])*ds[0]/ds[1]+tmp[-1, 0]
        if tx < -wr/2. or tx > wr/2.:
            next
        xx = fsolve(lambda x: ds[1]/ds[0]*(x-tmp[-1, 0]) +
                    tmp[-1, 1]-fx(x), tx)[0]
        rx = array([xx, fx(xx)])
        ds = dsx(ds, d_to_n(dfx(xx)))
        ra = -rx[1]/ds[1]*ds+rx
        rtn[:, :, k] = vstack((tmp, rx, ra))
    return rtn
