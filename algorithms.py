import numpy as np
import numba
from numba import jit

def ssvd_original(X, tol = 1e-3, lambda_us = None, lambda_vs = None, gamma1s = [2], gamma2s = [2], max_iter = 20):
    
    def BIC_v(lambda_v):
        v = abs(v_hat) - lambda_v*w2/2
        df = np.sum(v > 0)
        v[v <= 0] = 0
        v *= np.sign(v_hat)
        return (np.linalg.norm(X - u @ v.T)**2/sigma_sq_hat + np.log(nd)*df, v)
    
    def BIC_u(lambda_u):
        u = abs(u_hat) - lambda_u*w1/2
        df = np.sum(u > 0)
        u[u <= 0] = 0
        u *= np.sign(u_hat)
        return (np.linalg.norm(X - u @ v.T)**2/sigma_sq_hat + np.log(nd)*df, u)
    
    U, S, V = np.linalg.svd(X, full_matrices = False)
    u = U[:,0][:,None]
    v = V[0,:][:,None]
    s = S[0]
    n, d = X.shape
    nd = n*d
    iter_num = 0
    is_lambda_us_given = lambda_us is not None
    is_lambda_vs_given = lambda_vs is not None
    
    while iter_num < max_iter:
        
        # Update v
        v_hat = X.T @ u
        if all(v_hat == 0):
            v_new = v_hat
        else:
            sigma_sq_hat = np.linalg.norm(X - u @ v_hat.T)**2/(nd-d)
            min_BIC_val = float('inf')
            for gamma2 in gamma2s:
                w2 = np.abs(v_hat)**-gamma2
                if not is_lambda_vs_given:
                    lambda_vs = abs(v_hat)*2/w2
                BIC_vals, v_vals = list(zip(*list(map(BIC_v, lambda_vs))))
                if np.min(BIC_vals) < min_BIC_val:
                    min_BIC_val = np.min(BIC_vals)
                    v_new = v_vals[np.argmin(BIC_vals)]
                    if not all(v_new == 0):
                        v_new = v_new/np.linalg.norm(v_new)
        delta_v_norm = np.linalg.norm(v_new - v)
        v = v_new
        
        # Update u
        u_hat = X @ v
        if all(u_hat == 0):
            u_new = u_hat
        else:
            sigma_sq_hat = np.linalg.norm(X - u_hat @ v.T)**2/(nd-d)
            min_BIC_val = float('inf')
            for gamma1 in gamma1s:
                w1 = np.abs(u_hat)**-gamma1
                if not is_lambda_us_given:
                    lambda_us = abs(u_hat)*2/w1
                BIC_vals, u_vals = list(zip(*list(map(BIC_u, lambda_us))))
                if np.min(BIC_vals) < min_BIC_val:
                    min_BIC_val = np.min(BIC_vals)
                    u_new = u_vals[np.argmin(BIC_vals)]
                    if not all (u_new == 0):
                        u_new = u_new/np.linalg.norm(u_new)
        delta_u_norm = np.linalg.norm(u_new - u)
        u = u_new
        
        iter_num += 1
    
        if (delta_v_norm < tol) and (delta_u_norm < tol):
            break
    
    if (delta_v_norm >= tol) or (delta_u_norm >= tol):
        print("Failed to converge in {} iterations. Try increasing tolerance, or increasing the maximum number of iterations.".format(iter_num))
    
    return u, v, s

@jit(nopython=True)
def BIC_v(lambda_v, v_hat, w2, u, sigma_sq_hat, nd, X):
    v = np.abs(v_hat) - lambda_v*w2/2
    df = np.sum(v > 0)
    
    vcopy = v.ravel()
    
    for i in range(vcopy.shape[0]):
        if vcopy[i] <= 0:
            vcopy[i] = 0
    v = vcopy.reshape(v.shape)
    
    v *= np.sign(v_hat)
    return (np.linalg.norm(X - u @ v.T)**2/sigma_sq_hat + np.log(nd)*df, v)

@jit(nopython=True)
def BIC_u(lambda_u, u_hat, w1, v, sigma_sq_hat, nd, X):
    u = np.abs(u_hat) - lambda_u*w1/2
    df = np.sum(u > 0)
    
    ucopy = u.ravel()
    for i in range(ucopy.shape[0]):
        if ucopy[i] <= 0:
            ucopy[i] = 0
    u = ucopy.reshape(u.shape)

    u *= np.sign(u_hat)
    return (np.linalg.norm(X - u @ v.T)**2/sigma_sq_hat + np.log(nd)*df, u)

@jit(nopython=True)
def ssvd_new(X, BIC_v = BIC_v, BIC_u = BIC_u, tol = 1e-3, lambda_us = None, lambda_vs = None, gamma1s=None, gamma2s=None, max_iter = 20):
    
    if gamma1s is None:
        gamma1s = [2]
    
    if gamma2s is None:
        gamma2s = [2]
    
    
    U, S, V = np.linalg.svd(X, full_matrices = False)
    u = U[:,0].copy()
    u = u.reshape((u.shape[0],1))
    v = V[0,:].copy()
    v = v.reshape((v.shape[0],1))
    s = S[0]
    n, d = X.shape
    nd = n*d
    iter_num = 0
    is_lambda_us_given = lambda_us is not None
    is_lambda_vs_given = lambda_vs is not None
   
    while iter_num < max_iter:
        
        # Update v
        v_hat = X.T @ u
    
        if not v_hat.any():
            v_new = v_hat
            
        else:
            sigma_sq_hat = np.linalg.norm(X - u @ v_hat.T)**2/(nd-d)
            min_BIC_val = 1e8 
            for gamma2 in gamma2s:
                w2 = np.abs(v_hat)**-gamma2
                if not is_lambda_vs_given:
                    lambda_vs = np.abs(v_hat)*2/w2
  
                BIC_vals_lst=[0.]*lambda_vs.shape[0]
                v_vals=[np.zeros((lambda_vs.shape[0],1))]*lambda_vs.shape[0]
                for i in range(lambda_vs.shape[0]):
                    bic_val, v_val = BIC_v(lambda_v=lambda_vs[i,0], v_hat=v_hat, w2=w2, u=u, sigma_sq_hat=sigma_sq_hat, nd=nd, X=X)
                    BIC_vals_lst[i] = bic_val
                    v_vals[i] = v_val
                
                BIC_vals = np.array(BIC_vals_lst)
                if np.min(BIC_vals) < min_BIC_val:
                    min_BIC_val = np.min(BIC_vals)
                    v_new = v_vals[np.argmin(BIC_vals)]
                    
                    if v_new.any():
                        v_new = v_new/np.linalg.norm(v_new)
        delta_v_norm = np.linalg.norm(v_new - v)
        v = v_new
        
        # Update u
        u_hat = X @ v
        
        if not u_hat.any():
            u_new = u_hat
        else:
            sigma_sq_hat = np.linalg.norm(X - u_hat @ v.T)**2/(nd-d)
            min_BIC_val = 1e8 
            for gamma1 in gamma1s:
                w1 = np.abs(u_hat)**-gamma1
                if not is_lambda_us_given:
                    lambda_us = np.abs(u_hat)*2/w1
            
                BIC_vals_lst=[0.]*lambda_us.shape[0]
                u_vals=[np.zeros((lambda_us.shape[0],1))]*lambda_us.shape[0]
                for i in range(lambda_us.shape[0]):
                    bic_val, u_val = BIC_u(lambda_u = lambda_us[i,0], u_hat=u_hat, w1=w1, v=v, sigma_sq_hat=sigma_sq_hat, nd=nd, X=X)
                    BIC_vals_lst[i] = bic_val
                    u_vals[i] = u_val
                    
                BIC_vals = np.array(BIC_vals_lst)
                if np.min(BIC_vals) < min_BIC_val:
                    min_BIC_val = np.min(BIC_vals)
                    u_new = u_vals[np.argmin(BIC_vals)]
                  
                    if u_new.any():
                        u_new = u_new/np.linalg.norm(u_new)
        delta_u_norm = np.linalg.norm(u_new - u)
        u = u_new
        
        iter_num += 1
    
        if (delta_v_norm < tol) and (delta_u_norm < tol):
            break
    
    if (delta_v_norm >= tol) or (delta_u_norm >= tol):
        
        print("Failed to converge in", iter_num, "iterations. Try increasing tolerance, or increasing the maximum number of iterations.")
    
    return u, v, s

