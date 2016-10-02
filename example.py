import numpy as np
import patsms.rx as rx
from pprint import pprint


betam = np.deg2rad(60.) # max angle of incidence
ns = np.array([1., 1.5]) # refractive indicies

# Initial lens to test
R0 = np.array([3.3/5., -1., -1., 0., 0., 0., 0.])
X0 = np.array([-4.3/5., 7./5., 0.7, 0., 0., 0., 0.])

# Optimize
(R, X, o) = rx.optASRX(R0, X0, ns, betam)

pprint(o)
