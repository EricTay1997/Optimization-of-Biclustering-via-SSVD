# Implementation and Optimization of Biclustering via Sparse Singular Value Decomposition: STA 663 Final Project
### Eric Tay and Cathy Shi
#### April 27, 2021

### Description
This repository contains a report detailing our implementation, optimization and discussion of the SSVD algorithm in "Biclustering via Sparse Singular Value Decomposition" [1]. We have uploaded our implementation to [PyPI], and provided code to install the package and reproduce our results.

### Files
- `algorithms.py` contains an implementation of the original and optimized SSVD algorithm, which has been uploaded to [PyPI].
- `STA 663L Final Submission.ipynb` downloads the algorithms from [PyPI], and reproduces the results in `STA 663L Final Submission.pdf`.
- `STA 663L Final Submission.pdf` is the report for our final project.

### Package for SSVD
The original and optimized version of SSVD algorithm is now uploaded to [PyPI](https://test.pypi.org/project/ssvd-pkg-cathy10/) and ready for installation via the  command:

`pip install ssvd-pkg-cathy10`

The package can then be used as such:

`from ssvd_pkg import ssvd_original, ssvd_new`

`X = np.random.rand(100, 100)`

`u_original, v_original, s_original = ssvd_original(X) # This runs the original algorithm`

`u, v, s = ssvd_new(X) # This runs the optimized algorithm`

### References

[1] Lee et al. (2010), "Biclustering via Sparse Singular Value Decomposition", Biometrics 66, pp 1087-1095.
