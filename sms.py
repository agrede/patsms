from patsms.gemoptics import uv, dsr, dsx, dnr, dnx, opl, trace_stack, d_to_n, find_angle
from numpy import sin, cos, pi, sqrt, array, vstack, hstack, ones, zeros, fliplr, inner, where, argsort, dstack, arctan, abs, nan, sign, any
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


def solve_rr_rev(rf, df, ri, di, ns, l, thetai):
    """Returns position and normal of refractive surface
    rf -- final ray position
    df -- final ray direction
    ri -- reference position for opl
    di -- initial ray direction
    ns -- indices of refraction
    l -- optical path length
    """
    mf = fsolve(lambda x: l +
                ns[0]*inner((rf-x*df)-ri,
                            array([sin(thetai), cos(thetai)])) -
                ns[1]*x, l/ns[1])[0]
    rr = rf-mf*df
    nr = dnr(di, df, ns)
    return (rr, nr)


def sms_rx_inf_source(nr, nx, ystack, nstack, wr=1., wx=1.,
                      ni=1., thetai=4.66e-3, Nmax=200, hx=0.):
    """Returns design rr, rx, dnr, dnx"""
    ystack = hstack((ystack, [0.]))
    ns = hstack((ni, nr, nstack, nx))
    rr = [array([wr/2., ystack[0]])]
    rx = [array([wx/2., -hx])]
    di = array([[sin(thetai), -cos(thetai)],
                [-sin(thetai), -cos(thetai)]])
    tmp = wa(wr, wx, hx, ns[0], ns[-1], thetai)
    # thetao = arctan((2.*hx)/wr)
    # tmp = wr*ni*sin(thetai)/(nx*sin(thetao))
    ra = array([[-tmp/2., 0.], [tmp/2., 0.]])
    nr = [dnr(di[0, :],
              find_angle(hstack((ystack, [-hx])),
                         hstack((ns[1:], ns[[-1]])),
                         (wx-wr)/2.),
              ns[:2])]
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


def sms_rx_inf_source_rev(nr, nx, ystack, nstack, hr, hx, thetao,
                          wr=1., ni=1., thetai=4.66e-3, Nmax=200):
    """Returns design rr, rx, dnr, dnx"""
    ns = hstack((ni, nr, nstack, nx))
    ystack = hstack((ystack, [0.]))
    di = array([[sin(thetai), -cos(thetai)],
                [-sin(thetai), -cos(thetai)]])
    tmp = wr/(ns[1]*sin(thetao)/(ns[0]*sin(thetai)))
    ra = array([[-tmp/2., 0.], [tmp/2., 0.]])
    rr = [array([0., hr])]
    nr = [array([0., 1.])]
    ds = dsr(di[0, :], nr[-1], ns[:2])
    rs, ds, dl = trace_stack(ystack, ns[1:], rr[-1], ds)
    rx = [rs+ds*(-hx/ds[1])]
    nx = [dnx(ds, uv(ra[1, :]-rx[-1]))]
    l = [dl+opl(vstack((rs, rx[-1], ra[1, :])), array([ns[-1], ns[-1]]))]
    k = 0
    while k < Nmax and rr[-1][0] <= wr/2.:
        k = k+1
        ds = dsx(uv(rx[-1]-ra[0, :]), nx[-1])
        rs, ds, dl = trace_stack(ystack[::-1], ns[:0:-1],
                                 rx[-1], ds)
        dl = dl + opl(vstack((ra[0, :], rx[-1])), array(ns[-1:]))
        trr, tnr = solve_rr_rev(rs, -ds, rr[0], di[1, :],
                                ns[:2], l[0]-dl, thetai)
        rr.append(trr)
        nr.append(tnr)
        l.append(dl+opl(vstack((rs, rr[-1])), array(ns[1:2])))
        ds = dsr(di[0, :], nr[-1], ns[:2])
        rs, ds, dl = trace_stack(ystack, ns[1:], rr[-1], ds)
        dl = dl-ns[0]*inner(rr[-1]-rr[0],
                            array([-sin(thetai), cos(thetai)]))
        trx, tnx = solve_rx(rs, ds, ra[1, :], ns[-1], l[0]-dl)
        rx.append(trx)
        nx.append(tnx)
    return (vstack(rr), vstack(rx), vstack(nr), vstack(nx), hstack(l))


def sms_rx_inf_source_cont(nr, nx, ystack, nstack, rr_0, rr_next,
                           nr_next, l_0, thetao, wr=1., ni=1.,
                           thetai=4.66e-3, Nmax=200):
    ns = hstack((ni, nr, nstack, nx))
    ystack = hstack((ystack, [0.]))
    di = array([[sin(thetai), -cos(thetai)],
                [-sin(thetai), -cos(thetai)]])
    tmp = wr/(ns[1]*sin(thetao)/(ns[0]*sin(thetai)))
    ra = array([[-tmp/2., 0.], [tmp/2., 0.]])
    rr = [rr_0, rr_next]
    nr = [array([0., 1.]), nr_next]
    l = [l_0]
    ds = dsr(di[0, :], nr[-1], ns[:2])
    rs, ds, dl = trace_stack(ystack, ns[1:], rr[-1], ds)
    dl = dl-ns[0]*inner(rr[-1]-rr[0],
                        array([-sin(thetai), cos(thetai)]))
    trx, tnx = solve_rx(rs, ds, ra[1, :], ns[-1], l[0]-dl)
    rx = [trx]
    nx = [tnx]
    k = 0
    while k < Nmax and rr[-1][0] <= wr/2.:
        k = k+1
        ds = dsx(uv(rx[-1]-ra[0, :]), nx[-1])
        rs, ds, dl = trace_stack(ystack[::-1], ns[:0:-1],
                                 rx[-1], ds)
        dl = dl + opl(vstack((ra[0, :], rx[-1])), array(ns[-1:]))
        trr, tnr = solve_rr_rev(rs, -ds, rr[0], di[1, :],
                                ns[:2], l[0]-dl, thetai)
        rr.append(trr)
        nr.append(tnr)
        l.append(dl+opl(vstack((rs, rr[-1])), array(ns[1:2])))
        ds = dsr(di[0, :], nr[-1], ns[:2])
        rs, ds, dl = trace_stack(ystack, ns[1:], rr[-1], ds)
        dl = dl-ns[0]*inner(rr[-1]-rr[0],
                            array([-sin(thetai), cos(thetai)]))
        trx, tnx = solve_rx(rs, ds, ra[1, :], ns[-1], l[0]-dl)
        rx.append(trx)
        nx.append(tnx)
    return (vstack(rr)[1:, :], vstack(rx),
            vstack(nr)[1:, :], vstack(nx), hstack(l))


def sms_rx_inf_source_ang(nr, nx, ystack, nstack, hr, fr, fx,
                          thetas, thetao, dfr=None, dfx=None,
                          wr=1., ni=1., thetai=4.66e-3, Nmax=200):
    """Returns design rr, rx, dnr, dnx"""
    ns = hstack((ni, nr, nstack, nx))
    ystack = hstack((ystack, [0.]))
    di = array([[sin(thetas+thetai), -cos(thetas+thetai)],
                [sin(thetas-thetai), -cos(thetas-thetai)]])
    tmpa = trace_rx(fr, fx, ystack, nstack, nr, nx, array([0.]),
                    thetas+thetai, wr=wr, dfr=dfr, dfx=dfx)
    tmpb = trace_rx(fr, fx, ystack, nstack, nr, nx, array([0.]),
                    thetas-thetai, wr=wr, dfr=dfr, dfx=dfx)
    ra = vstack((tmpa[-1, :, 0], tmpb[-1, :, 0]))
    l0 = [opl(tmpa, array(ns[1:])), opl(tmpb, array(ns[1:]))]
    rr = [array([0., hr])]
    nr = [array([0., 1.])]
    ds = dsr(di[0, :], nr[-1], ns[:2])
    rs, ds, dl = trace_stack(ystack, ns[1:], rr[-1], ds)
    rx = []
    nx = [dnx(ds, uv(ra[1, :]-rx[-1]))]
    l = [dl+opl(vstack((rs, rx[-1], ra[1, :])), array([ns[-1], ns[-1]]))]
    k = 0
    while k < Nmax and rr[-1][0] <= wr:
        k = k+1
        ds = dsx(uv(rx[-1]-ra[0, :]), nx[-1])
        rs, ds, dl = trace_stack(ystack[::-1], ns[:0:-1],
                                 rx[-1], ds)
        dl = dl + opl(vstack((ra[0, :], rx[-1])), array(ns[-1:]))
        trr, tnr = solve_rr_rev(rs, -ds, rr[0], di[1, :],
                                ns[:2], l[0]-dl, thetai)
        rr.append(trr)
        nr.append(tnr)
        l.append(dl+opl(vstack((rs, rr[-1])), array(ns[1:2])))
        ds = dsr(di[0, :], nr[-1], ns[:2])
        rs, ds, dl = trace_stack(ystack, ns[1:], rr[-1], ds)
        dl = dl-ns[0]*inner(rr[-1]-rr[0],
                            array([-sin(thetai), cos(thetai)]))
        trx, tnx = solve_rx(rs, ds, ra[1, :], ns[-1], l[0]-dl)
        rx.append(trx)
        nx.append(tnx)
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
        kn = where(r[:, 0] > 0)[0]
        x = hstack((-r[kn, 0], r[kp, 0]))
        ks = argsort(x)
        x = x[ks]
        y = hstack((r[kn, 1], r[kp, 1]))[ks]
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
        dn = d_to_n(dfr(x))
        if dn[0]*sign(di[0]) > -di[1]:
            rtn[:, :, k] = nan
            continue
        ds = dsr(di, d_to_n(dfr(x)), ns[:2])
        tmp, ds, dl = trace_stack(ystack, ns[1:], rr, ds, return_all=True)
        if ds[1] >= 0:
            rtn[:, :, k] = nan
            continue
        tx = (-hx-tmp[-1, 1])*ds[0]/ds[1]+tmp[-1, 0]
        if tx < -wr/2. or tx > wr/2.:
            rtn[:, :, k] = nan
            continue
        if ds[0] == 0:
            xx = tmp[-1, 0]
        else:
            xx = fsolve(lambda x: ds[1]/ds[0]*(x-tmp[-1, 0]) +
                        tmp[-1, 1]-fx(x), tx)[0]
            if abs(ds[0]) < 0.01:
                xx = fsolve(lambda x: ds[1]/ds[0]*(x-tmp[-1, 0]) +
                            tmp[-1, 1]-fx(x), xx)[0]
        if (abs(xx) > wr/2.):
            rtn[:, :, k] = nan
            continue
        rx = array([xx, fx(xx)])
        print(rx)
        print(ds)
        print(d_to_n(dfx(xx)))
        ds = dsx(ds, d_to_n(dfx(xx)))
        print(ds)
        if ds[1] == 0:
            ra = array([rx[0], 0.])
        else:
            ra = -rx[1]/ds[1]*ds+rx
        rtn[:, :, k] = vstack((tmp, rx, ra))
    return rtn
