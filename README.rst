===========================
Japanese Immigration Policy
===========================
"The Dynamic Effects of Changes to Japanese Immigration Policy"


Abstract
========
This paper uses a single-sector dynamic general equilibrium (DGE) model to analyze Japanese immigration policy. We examine the effects on output, consumption, factor prices, and utility. We do this for both steady states and for transition paths


Contributors
============
- Scott C. Bradford
- Kerk L. Phillips


License
=======
The following copyright license restrictions apply:

Copyright: K. Phillips.  Feel free to copy, modify and use at your own risk.  However, you are not allowed to sell this software or otherwise impinge on its free distribution.


Contents
========
Ths repository contains Python code for each of the four scenarios laid out in the paper.  Each of these programs writes a .pkl file with the results of the simulation.  JapImm_4plots.py pulls data from .pkl files created by these programs and plots the impulse responese for each scenario on the same graph.  JapImmMC.py takes information from .pkl files and then runs a set of Monte Carlo simulations.

Supporting code from the LinApp package and DSGEmoments.py are also included.  The most recent versions of these can be found at https://github.com/kerkphil/DSGE-Utilities.