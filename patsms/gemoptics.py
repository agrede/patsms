from numpy import sqrt, power, eye, vstack, atleast_2d, diff, array, sign, zeros, inner, dot
from numpy.linalg import norm


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
    return uv(dot(n, vstack(di, ds))[0, :])


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


def trace_stack(ys, ns, r, dr):
    """ Trace stack of layers and return final position, direction, and opl
    ys -- array of N y values for the stack
    ns -- array of N+1 refractive indices
    r -- starting position
    dr -- starting direction unit vector
    """
    dn = array([0, sign(dr[1])])
    rs = vstack((r, vstack((zeros(ys.size), ys)).T))
    for k, y in enumerate(ys):
        rs[k, 0] = (y-r[1])*dr[0]/dr[1]+r[0]
        dr = dsr(dr, dn, ns[[k, k+1]])
    return (rs[-1, :], dr, opl(rs, ns[:-1]))
