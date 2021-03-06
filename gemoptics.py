# Copyright (C) 2016 Alex J. Grede
# GPL v3, See LICENSE.txt for details
# This function is part of PATSMS (https://github.com/agrede/patsms)
from numpy import sqrt, power, eye, vstack, atleast_2d, diff, array, sign, zeros, inner, dot, arctan, sin, cos, arange, sum
from numpy.linalg import norm
from scipy.optimize import fsolve
from numba import jit


def uv(x):
    """Return unit vector in direction of x"""
    return x/norm(x)


def dsr(di, dn, n):
    """Return unit vector of refracted ray
    di -- incident ray unit vector
    dn -- surface normal unit vector
    n -- array of refractive indices
    """
    r = n[0]/n[1]
    c = -inner(di, dn)
    return uv(r*di+(r*c-sqrt(1.-power(r, 2)*(1.-power(c, 2))))*dn)


def dsx(di, dn):
    """Return unit vector of reflected ray
    di -- incident ray unit vector
    dn -- surface normal unit vector
    """
    return uv(di-2.*inner(di, dn)*dn)


def R(di, ds):
    """Reflection matrix
    di -- incident ray unit vector
    ds -- reflected ray unit vector
    """
    dn = uv(ds-di)
    return eye(2)-2.*dot(dn, dn.T)


def dnr(di, ds, n):
    """Surface normal of refractive surface
    di -- incident ray unit vector
    ds -- refracted ray unit vector
    n -- array of refractive indices
    """
    n = atleast_2d(n)*[1., -1.]
    return uv(dot(n, vstack((di, ds)))[0, :])


def dnx(di, ds):
    """Surface normal of reflective surface
    di -- incident ray unit vector
    ds -- refracted ray unit vector
    """
    return uv(ds-di)


def opl(p, n):
    """Returns the optical path length
    p -- array of N rows of points [x, y]
    n -- array of N-1 values of the index of refraction
    """
    return inner(norm(diff(p, axis=0), axis=1), n)


def trace_stack(ys, ns, r, dr, return_all=False):
    """ Trace stack of layers and return final position, direction, and opl
    ys -- array of N y values for the stack
    ns -- array of N+1 refractive indices
    r -- starting position
    dr -- starting direction unit vector
    """
    dn = array([0, -sign(dr[1])])
    rs = vstack((r, vstack((zeros(ys.size), ys)).T))
    for k, y in enumerate(ys):
        if abs(dr[1]) > 0:
            rs[k+1, 0] = (y-r[1])*dr[0]/dr[1]+r[0]
        else:
            rs[k+1, 0] = r[0]
        dr = dsr(dr, dn, ns[[k, k+1]])
    if return_all:
        return (rs, dr, opl(rs, array(ns[:-1])))
    else:
        return (rs[-1, :], dr, opl(rs, array(ns[:-1])))


def d_to_n(dydx):
    """Return normal vector from surface derivative"""
    return uv([-dydx, 1.])


def find_angle(ys, ns, dx):
    """Find angle for stack that changes ray by dx"""
    r0 = array([0., 0.])
    th = fsolve(lambda x: dx -
                trace_stack(ys, ns, r0,
                            array([sin(x[0]), -cos(x[0])]))[0][0],
                arctan(dx/(ys[0]-ys[-1])))[0]
    return array([sin(th), -cos(th)])


def asphere(r, R, kappa, alpha):
    kappa = kappa+1.
    alpha = atleast_2d(alpha)
    r = atleast_2d(r).T
    powers = atleast_2d(2*(1+arange(alpha.size)))
    a = sqrt(1.-kappa*power(r/R, 2))
    return (power(r[:, 0], 2)/((a[:, 0]+1.)*R) +
            sum(alpha*power(r, powers), axis=1))


def dasphere(r, R, kappa, alpha):
    """First derivative"""
    kappa = kappa+1.
    alpha = atleast_2d(alpha)
    r = atleast_2d(r).T
    powers = atleast_2d(2*(1+arange(alpha.size)))
    a = sqrt(1.-kappa*power(r[:, 0]/R, 2))
    return (
        2.*r[:, 0]/((a+1.)*R) +
        kappa*power(r[:, 0]/R, 3)/(power(a+1., 2)*a) +
        sum(powers*alpha*power(r, powers-1), axis=1))


def ddasphere(r, R, kappa, alpha):
    """Second derivative"""
    kappa = kappa+1.
    alpha = atleast_2d(alpha)
    r = atleast_2d(r).T
    powers = atleast_2d(2*(1+arange(alpha.size)))
    a = sqrt(1.-kappa*power(r[:, 0]/R, 2))
    b = 1.+a
    return (
        2./(b*R) +
        5.*kappa*power(r, 2)/(power(b, 2)*a*power(R, 3)) +
        2.*power(kappa, 2)*power(r, 4)/(power(b, 3)*power(a, 2)*power(R, 5)) +
        power(kappa, 2)*power(r, 4)/(power(b, 2)*power(a, 3)*power(R, 5)) +
        sum((powers-1.)*powers*alpha*power(r, powers-2), axis=1))
