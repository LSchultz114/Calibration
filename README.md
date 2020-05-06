# Calibration
Miscellaneous work regarding calibration research

Jupyter notebooks created using miscellaneous codes during research efforts on calibration. 

Includes:
- A basic bayesian optimization example using a Gaussian Process prior (uses only numpy and pyplot)
- The same example implemented in a python package called George (https://github.com/dfm/george)
- The George implementation of the example with the Expected Improvement utility implemented for Active Learning (a semi-supervised method)

Currently, active subspaces does not import correctly into py3 due to changes in relative importing; attached is a version that was edited, though the author is in the process of correcting the issue.
Download the active subspaces software and install, refer or replace the site-package file in your python library with the one found here