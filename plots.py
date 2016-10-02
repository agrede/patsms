# Copyright (C) 2016 Alex J. Grede
# GPL v3, See LICENSE.txt for details
# This function is part of PATSMS (https://github.com/agrede/patsms)
from specPlot.plotspec import data_ranges, mkplot_axis
import numpy as np
import numpy.ma as ma
from jinja2 import Environment, FileSystemLoader
import os
import re

LATEX_SUBS = (
    (re.compile(r'\\'), r'\\textbackslash'),
    (re.compile(r'([{}_#%&$])'), r'\\\1'),
    (re.compile(r'~'), r'\~{}'),
    (re.compile(r'\^'), r'\^{}'),
    (re.compile(r'"'), r"''"),
    (re.compile(r'\.\.\.+'), r'\\ldots'),
)


def escape_tex(value):
    newval = value
    for pattern, replacement in LATEX_SUBS:
        newval = pattern.sub(replacement, newval)
    return newval

texenv = Environment(
    autoescape=False,
    loader=FileSystemLoader(
        os.path.join(os.path.dirname(__file__), "templates")))
texenv.block_start_string = '((*'
texenv.block_end_string = '*))'
texenv.variable_start_string = '((('
texenv.variable_end_string = ')))'
texenv.comment_start_string = '((='
texenv.comment_end_string = '=))'
texenv.filters['escape_tex'] = escape_tex
texenv.lstrip_blocks = True
texenv.trim_blocks = True


def mkphasePlot(pth, iop, Nstep=10,
                labels={'xlabel': {'text': 'Position',
                                   'symbol': 'x/u_i'},
                        'ylabel': {'text': 'Momentum',
                                   'symbol': 'p'},
                        'zlabel': {'text': 'Inc. Angle',
                                   'symbol': '\\beta',
                                   'units': '\\degree'}}):
    limits = {}
    ticks = {}
    (t_ticks, t_limits) = mkplot_axis('x', iop[:, :, 2])
    ticks.update(t_ticks)
    limits.update(t_limits)
    (t_ticks, t_limits) = mkplot_axis('y', iop[:, :, 3])
    ticks.update(t_ticks)
    limits.update(t_limits)
    k = np.where(np.all(~iop[:, :, 1].mask, axis=1))[0][0]
    zs = iop[k, ::Nstep, 1]
    (zmin, zmax, zjmin, zjmax, zstep, zstepm) = data_ranges(zs, 7, 7)
    limits['zmin'] = zmin
    limits['zmax'] = zmax
    x = np.linspace(iop[:, :, 2].min(), iop[:, :, 2].max(),
                    np.floor(iop.shape[1]/2.)+1)
    xa = ma.reshape(iop[:, :, 2], (-1,))
    pa = ma.reshape(iop[:, :, 3], (-1,))
    pn = np.array([pa[np.where((x[n] <= xa)*(xa < x[n+1]))[0]].min()
                   for n in range(x.size-1)])
    px = np.array([pa[np.where((x[n] <= xa)*(xa < x[n+1]))[0]].max()
                   for n in range(x.size-1)])
    np.savetxt(pth+"_Lines.csv",
               np.hstack((iop[:, ::Nstep, 2], iop[:, ::Nstep, 3])),
               delimiter=',')
    np.savetxt(pth+"_Bound.csv",
               np.vstack((x[1:-2], pn[1:-1], px[1:-1])).T,
               delimiter=',')
    template = texenv.get_template('phasePlot.tex')

    f = open(pth+".tex", 'w')
    f.write(
        template.render(limits=limits,
                        ticks=ticks, labels=labels, zs=zs))
    f.close()
