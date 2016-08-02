Planar angular tracking simaltanious multiple surface design
============================================================

For SMS design and RX design based on phase filling and aspheres

Dependencies
------------
* [numba](http://numba.pydata.org/)
* [numpy](http://www.numpy.org/)
* [scipy](http://www.scipy.org/)
* [matplotlib](http://matplotlib.org/)
* [jinja](http://jinja.pocoo.org/) Only for plots.py

Files
-----
* `rx.py` Ray tracing and optimization of RX concentrator
  + `optASRX`: optimization method
  + `optASRXFit`: Objective function
  + `spotSize90`: For the spot size component to objective
  + `inPhaseUse`: For the dropped rays component to the objective
  + `phaseFill`: For the phase filling component to the objective
  + `inOutPhase`: Ray tracing and output needed to calculate objective
* `gemoptics.py` Geometric optics and other utility functions
* `zemaxRX.py` Older ZDDE interface to zemax
* `plots.py` Plotting functions for phase space
* `sms.py` SMS method for RX concentrator
