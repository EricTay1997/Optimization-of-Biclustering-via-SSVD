# Implementation and Optimization of Biclustering via Sparse Singular Value Decomposition: STA 663 Final Project
### Eric Tay and Cathy Shi
#### April 27, 2021

### Files
- `algorithms.py` contains an implementation of the original and optimized SSVD algorithm, which has been uploaded to [PyPI].
- `STA 663L Final Submission.ipynb` downloads the algorithms from [PyPI], and reproduces the results in `STA 663L Final Submission.pdf`.
- `STA 663L Final Submission.pdf` is the report for our final project.

### Package for SSVD
The optimized version of SSVD algorithm is now uploaded to [PyPI](https://test.pypi.org/project/ssvd-pkg-cathy10/) and ready for installation via the following command lines:

\begin{lstlisting}
pip install ssvd-pkg-cathy10
import ssvd_pkg
\end{lstlisting}

The the function "ssvd" can be called by inputting the matrix of interest and get the values of decomposition $u$, $v$, $s$.
