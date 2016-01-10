from gemoptics import uv, dsr, dsx, dnr, dnx, opl, trace_stack
from numpy import sin, sqrt
from numpy.linalg import norm, inner
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


def rx(ri, di, rf, n, l):
    """Returns position and normal of a mirror
    ri -- initial ray position
    di -- initial ray direction
    rf -- final ray position
    n -- index of refraction
    l -- optical path length
    """
    mi = fsolve(lambda x: l-n*(x+norm(rf-(ri+x*di))),
                (l/n-norm(rf-ri)))[0]
    trx = ri+mi*di
    nx = dnx(di, uv(rf-trx))
    return (trx, nx)


def rr(rf, df, ri, di, ns, l):
    """Returns position and normal of refractive surface
    rf -- final ray position
    df -- final ray direction
    ri -- reference position for opl
    di -- initial ray direction
    ns -- indices of refraction
    l -- optical path length
    """
    mf = fsolve(lambda x: l-inner((rf-x*df)-ri, [1., -1.]*di) -
                ns[1]*x, l/ns[1])[0]
    trr = rf-mf*df
    nr = dnr(di, df, ns)
    return (trr, nr)
