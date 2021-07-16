

# Streaming Dynamic Mode Decomposition (sDMD)

This repository contains a Python implementation of Streaming Dynamic Mode Decomposition (sDMD) as described in the paper [Liew, J. et al. -  *Streaming dynamic mode decomposition for short-term forecasting in wind farms*](). The algorithm is suited to performing DMD analysis on-the-fly. 

The implementation draws from concepts from similar works by [Hemani, M. et al. -  *Dynamic mode decomposition for large and streaming datasets*](http://dx.doi.org/10.1063/1.4901016), and [Zhang, H. et al. - *Online dynamic mode decomposition for time-varying systems*](https://doi.org/10.1137/18M1192329)


# Citation

[![DOI](https://zenodo.org/badge/337083998.svg)](https://zenodo.org/badge/latestdoi/337083998)

If sDMD played a role in your research, please cite it. This software can be
cited as:
  Jaime Liew, 2021. jaimeliew1/Streaming-DMD: Version 1.0. [doi:10.5281/zenodo.4646750](http://doi.org/10.5281/zenodo.4646750)

For LaTeX users:

.. code-block:: latex

  @software{jaime_liew_2021_4646750,
    author       = {Jaime Liew},
    title        = {Streaming-DMD},
    year         = 2021,
    publisher    = {Zenodo},
    version      = {v1.0},
    doi          = {10.5281/zenodo.4646750},
    url          = {https://doi.org/10.5281/zenodo.4646750}
  }
    
    
# Installation
Users who want to run sDMD should download the source code and install the package using `pip`, as shown below.

```
git clone https://github.com/jaimeliew1/Streaming-DMD.git
pip install -e streaming_dmd
```
