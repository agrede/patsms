from gemoptics import uv, dsr, dsx, dnr, dnx, opl, trace_stack
from numpy import sin, cos, pi, sqrt, array, vstack, hstack, ones, fliplr, inner
from numpy.linalg import norm
from scipy.optimize import fsolve


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
    nx = [dnx(uv(rx[-1]-rr[-1]), uv(ra[1, :]-rx[-1]))]
    l = [opl(vstack((rr[0],
                     vstack(((wr/2.*ones(ystack.size)), ystack)).T,
                     rx[0], ra[1, :])), hstack((ns[1:], ns[[-1]])))]
    k = 0
    while k < Nmax and rr[-1][0] >= 0. and rx[-1][0] >= 0.:
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
