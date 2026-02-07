#!/usr/bin/env python
# coding: utf-8

"""
PNMTF_alpha - Penalized Non-negative Matrix Tri-Factorization with α-divergence
"""

import numpy as np
from sklearn.utils import check_random_state, check_array
from numpy.random import rand
from datetime import datetime

class PNMTF_alpha:

    def __init__(self,
                 n_row_clusters=2, 
                 n_col_clusters=2, 
                 alpha=1.5,  # α-divergence parameter (default > 1)
                 delta=0,    # row orthogonality penalty
                 beta=0,     # column orthogonality penalty
                 R_init=None,
                 C_init=None,
                 max_iter=100,
                 n_init=1,
                 tol=1e-9,
                 random_state=None):
        
        self.n_row_clusters = n_row_clusters
        self.n_col_clusters = n_col_clusters
        self.alpha = alpha
        self.delta = delta
        self.beta = beta
        self.R_init = R_init
        self.C_init = C_init
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = check_random_state(random_state)
        
        # Results
        self.R = None          # Row clustering matrix
        self.C = None          # Column clustering matrix  
        self.B = None          # Basis matrix
        self.row_labels_ = None
        self.column_labels_ = None
        self.reconstructed_matrix = None
        self.criterions = []
        self.criterion = -np.inf
        self.n_iter_ = 0
        self.runtime = None

    def fit(self, X, y=None):
        """
        Fit PNMTF_alpha model to data matrix X.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data matrix to be factorized
            
        Returns
        -------
        self
        """
        X = check_array(X, accept_sparse=False, dtype=np.float64, 
                       ensure_2d=True, ensure_min_features=self.n_col_clusters,
                       ensure_min_samples=self.n_row_clusters)
        
        X = np.asarray(X, dtype=np.float64)
        n, m = X.shape
        g = self.n_row_clusters
        s = self.n_col_clusters
        
        # Check non-negativity
        if np.any(X < 0):
            raise ValueError("Input matrix X must be non-negative")
        
        best_criterion = -np.inf
        best_R, best_C, best_B = None, None, None
        
        random_state = check_random_state(self.random_state)
        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
        
        for seed in seeds:
            R, C, B, criterion, n_iter = self._fit_single(X, seed)
            
            if criterion > best_criterion:
                best_criterion = criterion
                best_R, best_C, best_B = R, C, B
                self.n_iter_ = n_iter
        
        # Store best results
        self.R = best_R
        self.C = best_C  
        self.B = best_B
        self.criterion = best_criterion
        
        # Get hard clustering assignments (Algorithm step 15)
        self.row_labels_ = np.argmax(best_R, axis=1) + 1
        self.column_labels_ = np.argmax(best_C, axis=1) + 1
        
        # Reconstructed matrix
        self.reconstructed_matrix = best_R @ best_B @ best_C.T
        
        return self

    def _fit_single(self, X, random_state):
        """
        Single run of PNMTF_alpha algorithm.
        """
        np.random.seed(random_state)
        
        n, m = X.shape
        g = self.n_row_clusters
        s = self.n_col_clusters
        
        # Step 7: Initialize R, C, B
        if self.R_init is None:
            R = np.random.rand(n, g).astype(np.float64)
            R = R / R.sum(axis=1, keepdims=True)  # Normalize rows
        else:
            R = self.R_init.copy()
            
        if self.C_init is None:
            C = np.random.rand(m, s).astype(np.float64)
            C = C / C.sum(axis=1, keepdims=True)  # Normalize rows
        else:
            C = self.C_init.copy()
        
        # B = R^T X C (as in Algorithm step 7)
        B = R.T @ X @ C
        
        # Identity matrices for orthogonality terms
        I_g = np.eye(g)
        I_s = np.eye(s)
        
        # Convergence tracking
        prev_criterion = -np.inf
        criterion_history = []
        start_time = datetime.now()
        
        # Main optimization loop (Algorithm steps 8-13)
        for iteration in range(self.max_iter):
            # Step 9: Update R (fixing C and B)
            X_hat = R @ B @ C.T
            X_hat = np.maximum(X_hat, 1e-10)  # Avoid division by zero
            
            # Equation (8) from the algorithm (simplified form)
            # Note: Actual equation (8) would need to be provided
            # Using generalized multiplicative update for α-divergence
            if self.delta == 0:
                # Without orthogonality penalty
                num_R = ((X / X_hat)**self.alpha) @ C @ B.T
                den_R = np.ones((n, m)) @ C @ B.T
            else:
                # With orthogonality penalty
                num_R = ((X / X_hat)**self.alpha) @ C @ B.T
                den_R = np.ones((n, m)) @ C @ B.T + 2 * self.delta  * R @ (np.ones((g, g)) - I_g)
            
            R *= (num_R / np.maximum(den_R, 1e-10))**(1/self.alpha)
            R = np.maximum(R, 1e-10)
            
            # Normalize R rows
            R = R / R.sum(axis=1, keepdims=True)
            
            # Step 10: Update B (fixing R and C)
            X_hat = R @ B @ C.T
            X_hat = np.maximum(X_hat, 1e-10)
            
            # Equation (9)
            num_B = R.T @ ((X / X_hat)**self.alpha) @ C
            den_B = R.T @ np.ones((n, m)) @ C
            
            B *= (num_B / np.maximum(den_B, 1e-10))**(1/self.alpha)
            B = np.maximum(B, 1e-10)
            
            # Step 11: Update C (fixing R and B)
            X_hat = R @ B @ C.T
            X_hat = np.maximum(X_hat, 1e-10)
            
            # Equation (10)
            if self.beta == 0:
                num_C = ((X / X_hat).T)**self.alpha @ R @ B
                den_C = np.ones((m, n)) @ R @ B
            else:
                num_C = ((X / X_hat).T)**self.alpha @ R @ B
                den_C = np.ones((m, n)) @ R @ B + 2 * self.beta  * C @ (np.ones((s, s)) - I_s)
            
            C *= (num_C / np.maximum(den_C, 1e-10))**(1/self.alpha)
            C = np.maximum(C, 1e-10)
            
            # Normalize C rows
            C = C / C.sum(axis=1, keepdims=True)
            
            # Step 12: Compute criterion (α-divergence with penalties)
            X_hat = R @ B @ C.T
            X_hat = np.maximum(X_hat, 1e-10)
            
            # α-divergence term
            if abs(self.alpha - 1.0) < 1e-10:
                # KL divergence (α→1)
                div_term = np.sum(X * np.log(X / X_hat) - X + X_hat)
            else:
                # General α-divergence
                term1 = (X**(self.alpha) * X_hat**(1 - self.alpha) - self.alpha * X + (self.alpha - 1) * X_hat)
                div_term = np.sum(term1) / (self.alpha * (self.alpha - 1))
            
            # Orthogonality penalty terms
            orth_R = self.delta * np.trace(R.T @ R @ (np.ones((g, g)) - I_g))
            orth_C = self.beta * np.trace(C.T @ C @ (np.ones((s, s)) - I_s))
            
            criterion = -div_term - orth_R - orth_C
            criterion_history.append(criterion)
            
            # Check convergence
            if iteration > 0 and abs(criterion - prev_criterion) < self.tol:
                break
                
            prev_criterion = criterion
        
        # Step 14: Probabilistic interpretation
        D_R = np.diag(R.sum(axis=0))
        D_C = np.diag(C.sum(axis=0))
        
        R = R @ np.diag((B @ D_C) @ np.ones(s))
        B = D_R @ B @ D_C
        C = C @ np.diag(np.ones(g).T @ D_R @ B)
        
        # Normalize for probability interpretation
        R = R / R.sum(axis=1, keepdims=True)
        C = C / C.sum(axis=1, keepdims=True)
        
        runtime = (datetime.now() - start_time).total_seconds()
        self.runtime = runtime
        self.criterions = criterion_history
        
        return R, C, B, criterion, iteration + 1

    def transform(self, X=None):
        """
        Transform data or get factorization matrices.
        """
        if X is None:
            return self.R, self.B, self.C
        else:
            # For new data, you would project using learned basis
            # This is a simplified version
            return self.R

    def fit_transform(self, X):
        """
        Fit to data and return the row clustering matrix.
        """
        self.fit(X)
        return self.R