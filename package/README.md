# SSVD Algorithm

This package contains three functions that can realize the Sparse singular value decomposition (SSVD) algorithm. `ssvd_original` performs the SSVD algorithm optimized by using `sparsesvd` from `sparsesvd`. `ssvd_new` is optimized by using Numba, a JIT (just-in-time) compiler for Python code, which is much faster than the `ssvd_original`.

## Algorithm overview
The basic algorithm was written according to the paper "Biclustering via Sparse Singular Value Decomposition" by Mihee Lee, Haipeng Shen, Jianhua Z. Huang and J. S. Marron, published on Biometrics in Dec. 2010. This SSVD algorithm is an efficient iterative algorithm for computing the sparse singular vectors. It uses sparsity-infucing regularization penalties via penaltiy parameters to produce sparse singular vectors, which is effective in biclustering--a method that allows for simultaneous identification of distinctive “checkerboard” patterns in data matrices, or sets of rows (or samples) and sets of columns (or variables) in the matrices that are significantly associated.

## Installation 

In order to install SSVD, you’ll need sparsesvd, Numba, NumPy, Scipy and Cython.
Install SSVD and its dependencies with:

```python
pip install numba
pip install numpy
pip install scipy
pip install cython
pip install sparsesvd
pip install -i https://test.pypi.org/simple/ SSVD-pkg-cathy10
```

## Usage

```python
ssvd_original(X, tol = 1e-3, lambda_us = None, lambda_vs = None, gamma1s = [2], gamma2s = [2], max_iter = 20, sparse_decomp = False)

ssvd_new(X, BIC_v = BIC_v, BIC_u = BIC_u, tol = 1e-3, lambda_us = None, lambda_vs = None, gamma1s=None, gamma2s=None, max_iter = 20)
```


### Arguments

#### ssvd_original

- X: input matrix with type `double` elements to decompose
- tol: tolerance threshold used for convergence, type `double`
- lambda_us, lambda_vs: nonnegative penalty parameters which will be calculated inside of the function if not given; each is aan array with values of type `double`
- gamma1s, gamma2s: nonnegative parameters that controls for the weights in the adaptive lasso fit; default is 2; each consists a `double` in a list format
- max_iter: maximum iteration allowed; type `int`
- sparse_decomp: default is False; if True, uses `sparsesvd` function to optimize SVD decomposition instead of using the default option of using `numpy.linalg.svd`; type `boolean`

#### ssvd_new

- X: input matrix with type `double` elements to decompose
- tol: tolerance threshold used for convergence, type `double`
- BIV_v, BIV_u: embedded functions that users cannot change
- lambda_us, lambda_vs: nonnegative penalty parameters which will be calculated inside of the function if not given
- gamma1s, gamma2s: nonnegative parameters that controls for the weights in the adaptive lasso fit; default is 2; each consists a `double` in a list format
- max_iter: maximum iteration allowed; type `int`

### Output

One SSVD layer in terms of `u`,`v`,`s`, where `u`,`v` are arrays with type double and `s` is a scalar with type double.
