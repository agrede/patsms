Planar angular tracking simaltanious multiple surface design
============================================================

[![DOI](https://zenodo.org/badge/69797744.svg)](https://zenodo.org/badge/latestdoi/69797744)

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
* `example.py` generates the 2D RX concentrator used in Grede, Price, and Giebink (2016)
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

Acknowledgements
----------------
This work was funded in part by the National Science Foundation under Grant No. CBET-1508968 and the Advanced Research Projects Agency-Energy (ARPA-E), U.S. Department of Energy, under Award No. DE-AR0000626. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.
