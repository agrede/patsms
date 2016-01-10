from numpy import sqrt, power, eye, vstack, atleast_2d, diff
from numpy.linalg import norm, inner, dot


def uv(x):
    return x/norm(x)


def dsr(di, dn, n):
    r = n[0]/n[1]
    c = -inner(di, dn)
    return uv(r*di+(r*c-sqrt(1.-power(r, 2)*(1.-power(c, 2))))*dn)


def dsx(di, dn):
    return uv(di-2.*inner(di, dn)*dn)


def R(di, ds):
    dn = uv(ds-di)
    return eye(2)-2.*dot(dn, dn.T)


def dnr(di, ds, n):
    n = atleast_2d(n)*[1., -1.]
    return uv(dot(n, vstack(di, ds))[0, :])


def opl(p, n):
    return inner(norm(diff(p, axis=0), axis=1), n)
