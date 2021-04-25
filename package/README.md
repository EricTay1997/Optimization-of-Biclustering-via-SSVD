# SSVD Algorithm

This package contains the Sparse singular value decomposition (SSVD) algorithm, optimized by Numba, a JIT (just-in-time) compiler for Python code. The basic algorithm was written according to the paper "Biclustering via Sparse Singular Value Decomposition" by Mihee Lee, Haipeng Shen, Jianhua Z. Huang and J. S. Marron, published on Biometrics in Dec. 2010.



This SSVD algorithm is an efficient iterative algorithm for computing the sparse singular vectors. It uses sparsity-infucing regularization penalties via penaltiy parameters to produce sparse singular vectors, which is effective in biclustering--a method that allows for simultaneous identification of distinctive “checkerboard” patterns in data matrices, or sets of rows (or samples) and sets of columns (or variables) in the matrices that are significantly associated.