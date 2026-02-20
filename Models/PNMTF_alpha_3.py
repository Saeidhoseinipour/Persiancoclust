#!/usr/bin/env python
# coding: utf-8

"""
PNMTF_alpha: Penalized Non-negative Matrix Tri-Factorization with alpha-divergence.

Reference: Algorithm 1 — PNMTF_alpha

Approximation: X ≈ R B C^T
  - R ∈ R+^{n×g}  : row cluster membership matrix
  - B ∈ R+^{g×s}  : core cluster association matrix
  - C ∈ R+^{m×s}  : column cluster membership matrix

Update rules use α-divergence, matching equations (8), (9), (10).

Author: Hoseinipour Saeid <saeidhoseinipour@aut.ac.ir>
"""

import numpy as np
from numpy import linalg
from numpy.random import rand
from datetime import datetime
from sklearn.utils import check_random_state, check_array

from ..initialization import random_init
from ..io.input_checking import check_positive


class PNMTF_alpha:
    """
    Penalized Non-negative Matrix Tri-Factorization with α-divergence.

    Factorizes X ≈ R B C^T where:
        R ∈ R+^{n×g}  row cluster matrix
        B ∈ R+^{g×s}  core matrix
        C ∈ R+^{m×s}  column cluster matrix

    Parameters
    ----------
    n_row_clusters : int
        Number of row clusters (g).
    n_col_clusters : int
        Number of column clusters (s).
    alpha : float
        α-divergence parameter. α=1 → KL-divergence, α=2 → Euclidean.
    delta : float ≥ 0
        Row regularization parameter (orthogonality of R).
    beta : float ≥ 0
        Column regularization parameter (orthogonality of C).
    max_iter : int
        Maximum number of iterations N.
    n_init : int
        Number of random initializations.
    tol : float
        Convergence tolerance.
    random_state : int or None
        Random seed.
    epsilon : float
        Small value to avoid division by zero.
    """

    def __init__(
        self,
        n_row_clusters=2,
        n_col_clusters=2,
        alpha=1.0,          # α-divergence parameter
        delta=0.0,          # row regularization δ ≥ 0
        beta=0.0,           # col regularization β ≥ 0
        R_init=None,        # optional initialization for R
        B_init=None,        # optional initialization for B
        C_init=None,        # optional initialization for C
        max_iter=100,
        n_init=1,
        tol=1e-9,
        random_state=None,
        epsilon=1e-10,      # numerical stability floor
    ):
        self.n_row_clusters = n_row_clusters
        self.n_col_clusters = n_col_clusters
        self.alpha = alpha
        self.delta = delta
        self.beta = beta
        self.R_init = R_init
        self.B_init = B_init
        self.C_init = C_init
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = check_random_state(random_state)
        self.epsilon = epsilon

        # Outputs
        self.R = None
        self.B = None
        self.C = None
        self.row_labels_ = None
        self.column_labels_ = None
        self.criterion = -np.inf
        self.criterions = []
        self.runtime = None
        self.soft_matrix = None
        self.hard_matrix = None
        self.orthogonality_R = None
        self.orthogonality_C = None
        self.MSE = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y=None):
        """
        Fit PNMTF_alpha to data matrix X.

        Parameters
        ----------
        X : array-like, shape (n, m), non-negative.
        """
        check_array(
            X,
            accept_sparse=False,
            dtype="numeric",
            ensure_2d=True,
            ensure_min_samples=self.n_row_clusters,
            ensure_min_features=self.n_col_clusters,
        )
        check_positive(self.n_row_clusters, "n_row_clusters")
        check_positive(self.n_col_clusters, "n_col_clusters")

        if np.any(X < 0):
            raise ValueError("X must be non-negative (X ∈ R+^{n×m}).")

        X = X.astype(float)

        criterion = -np.inf
        criterions = []
        row_labels_ = None
        column_labels_ = None
        runtime = None

        random_state = check_random_state(self.random_state)
        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)

        for seed in seeds:
            self._fit_single(X, seed)

            if np.isnan(self.criterion):
                raise ValueError(
                    "Criterion is NaN. Check that X contains no zeros "
                    "or unexpected NaN values."
                )

            if self.criterion > criterion:
                criterion = self.criterion
                criterions = self.criterions
                row_labels_ = self.row_labels_
                column_labels_ = self.column_labels_
                runtime = self.runtime

        self.random_state = random_state
        self.criterion = criterion
        self.criterions = criterions
        self.row_labels_ = row_labels_
        self.column_labels_ = column_labels_
        self.runtime = runtime

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fit_single(self, X, random_state=None):
        """
        Single run of PNMTF_alpha (Algorithm 1 in the paper).

        Notation follows the paper exactly:
            R  ↔  row cluster matrix       (n × g)
            B  ↔  core matrix              (g × s)
            C  ↔  column cluster matrix    (m × s)
        """
        n, m = X.shape
        g = self.n_row_clusters
        s = self.n_col_clusters
        alpha = self.alpha
        delta = self.delta
        beta = self.beta
        eps = self.epsilon

        # ---- Constant matrices ----------------------------------------
        # E_nm : n×m matrix of ones  (Algorithm 1, below eq. 10)
        E_nm = np.ones((n, m))

        # Identity matrices
        I_g = np.eye(g)
        I_s = np.eye(s)

        # Ψ_g = 1_g 1_g^T - I_g  (orthogonality penalty for R)
        # Φ_s = 1_s 1_s^T - I_s  (orthogonality penalty for C)
        Psi_g = np.ones((g, g)) - I_g   # Ψ_g
        Phi_s = np.ones((s, s)) - I_s   # Φ_s

        # ---- Initialization  (step 9 in Algorithm 1) ------------------
        # R^(0), C^(0) random non-negative
        # B^(0) = R^(0)^T X C^(0)   ← paper's initialization
        R = rand(n, g) if self.R_init is None else self.R_init.copy()
        C = rand(m, s) if self.C_init is None else self.C_init.copy()

        if self.B_init is None:
            B = R.T @ X @ C          # B^(0) = R^(0)^T X C^(0)
        else:
            B = self.B_init.copy()

        # Ensure strict positivity for numerical stability
        R = np.maximum(R, eps)
        B = np.maximum(B, eps)
        C = np.maximum(C, eps)

        # ---- Tracking -------------------------------------------------
        c_init = -np.inf
        c_list = []
        orth_R_list = []
        orth_C_list = []
        iteration = 0
        start_time = datetime.now()

        # ---- Main loop  (steps 10-15) ---------------------------------
        while True:

            # -- Update R  (eq. 8) --------------------------------------
            #
            #           [ (X ⊘ RBC^T)^α ⊙ CB^T ]
            # R ← R ⊙  ─────────────────────────────
            #           [ E_nm CB^T + 2δ R Ψ_g  ]_+
            #
            if self.R_init is None:
                XBC = np.maximum(R @ B @ C.T, eps)        # n×m
                ratio_alpha = (X / XBC) ** alpha           # (X ⊘ RBC^T)^α, n×m

                numer_R = ratio_alpha @ (C @ B.T)          # n×m @ m×g = n×g
                denom_R = E_nm @ (C @ B.T) + 2 * delta * (R @ Psi_g)
                denom_R = np.maximum(denom_R, eps)

                R = np.maximum(R * (numer_R / denom_R), eps)

            # -- Update C  (eq. 9) --------------------------------------
            #
            #           [ ((X ⊘ RBC^T)^α)^T ⊙ RB ]
            # C ← C ⊙  ───────────────────────────────
            #           [ E_nm^T RB + 2β C Φ_s    ]_+
            #
            if self.C_init is None:
                XBC = np.maximum(R @ B @ C.T, eps)        # n×m (recompute with new R)
                ratio_alpha = (X / XBC) ** alpha           # n×m

                numer_C = ratio_alpha.T @ (R @ B)          # m×n @ n×s = m×s
                denom_C = E_nm.T @ (R @ B) + 2 * beta * (C @ Phi_s)
                denom_C = np.maximum(denom_C, eps)

                C = np.maximum(C * (numer_C / denom_C), eps)

            # -- Update B  (eq. 10) -------------------------------------
            #
            #           [ R^T (X ⊘ RBC^T)^α C ]
            # B ← B ⊙  ──────────────────────────
            #           [ R^T E_nm C           ]_+
            #
            if self.B_init is None:
                XBC = np.maximum(R @ B @ C.T, eps)        # n×m (recompute with new R,C)
                ratio_alpha = (X / XBC) ** alpha           # n×m

                numer_B = R.T @ ratio_alpha @ C            # g×n @ n×m @ m×s = g×s
                denom_B = R.T @ E_nm @ C                   # g×s
                denom_B = np.maximum(denom_B, eps)

                B = np.maximum(B * (numer_B / denom_B), eps)

            # -- Probabilistic normalization  (step 16) -----------------
            #
            # D_R = diag(1_n^T R),  D_C = diag(1_m^T C)
            # R ← R diag(B D_C 1_s)
            # B ← D_R B D_C
            # C ← C diag(1_g^T D_R B)
            #
            D_R = np.diag(R.sum(axis=0))                  # diag(1_n^T R), g×g
            D_C = np.diag(C.sum(axis=0))                  # diag(1_m^T C), s×s

            ones_s = np.ones(s)
            ones_g = np.ones(g)

            scale_R = np.diag(B @ D_C @ ones_s)           # diag(B D_C 1_s)
            R = R @ scale_R

            B = D_R @ B @ D_C

            scale_C = np.diag(ones_g @ D_R @ B)           # diag(1_g^T D_R B)
            C = C @ scale_C

            # Clip after normalization
            R = np.maximum(R, eps)
            B = np.maximum(B, eps)
            C = np.maximum(C, eps)

            # -- Orthogonality tracking ---------------------------------
            orth_R_list.append(linalg.norm(R.T @ R - I_g, "fro"))
            orth_C_list.append(linalg.norm(C.T @ C - I_s, "fro"))

            # -- α-divergence criterion ---------------------------------
            #
            # D_α(X || RBC^T) + δ·trace(R Ψ_g R^T) + β·trace(C Φ_s C^T)
            #
            XBC = np.maximum(R @ B @ C.T, eps)
            c = (
                self._alpha_divergence(X, XBC, alpha)
                + delta * np.trace(R @ Psi_g @ R.T)
                + beta * np.trace(C @ Phi_s @ C.T)
            )

            iteration += 1
            c_list.append(c)

            # -- Convergence check  (step 14) ---------------------------
            if np.abs(c - c_init) <= self.tol or iteration >= self.max_iter:
                break

            c_init = c

        end_time = datetime.now()

        # -- Hard cluster assignments  (step 17) ------------------------
        #
        # k* = argmax_k R_{ik},   h* = argmax_h C_{jh}
        #
        row_labels = np.argmax(R, axis=1)         # shape (n,)
        col_labels = np.argmax(C, axis=1)         # shape (m,)

        R_hard = np.zeros_like(R)
        R_hard[np.arange(n), row_labels] = 1.0

        C_hard = np.zeros_like(C)
        C_hard[np.arange(m), col_labels] = 1.0

        soft_matrix = R @ B @ C.T
        hard_matrix = R_hard @ B @ C_hard.T
        N_total = n * m

        # -- Store results ----------------------------------------------
        self.R = R_hard
        self.B = B
        self.C = C_hard
        self.soft_matrix = soft_matrix
        self.hard_matrix = hard_matrix
        self.row_labels_ = (row_labels + 1).tolist()     # 1-indexed
        self.column_labels_ = (col_labels + 1).tolist()  # 1-indexed
        self.criterion = c
        self.criterions = c_list
        self.orthogonality_R = orth_R_list
        self.orthogonality_C = orth_C_list
        self.MSE = (
            linalg.norm(X - hard_matrix, "fro") ** 2 / N_total
        )
        self.runtime = [str(end_time - start_time)]
        self.max_iter = iteration

    # ------------------------------------------------------------------
    # α-divergence
    # ------------------------------------------------------------------

    @staticmethod
    def _alpha_divergence(X, X_hat, alpha, eps=1e-10):
        """
        Generalized α-divergence D_α(X || X_hat).

        Special cases:
            α → 1 : KL divergence  Σ [X log(X/X̂) - X + X̂]
            α = 2 : Euclidean       0.5 Σ (X - X̂)²
            α = 0 : Itakura-Saito

        General form (α ≠ 0, 1):
            D_α = (1 / (α(α-1))) Σ [X^α X̂^{1-α} - αX + (α-1)X̂]
        """
        X_hat = np.maximum(X_hat, eps)

        if np.isclose(alpha, 1.0):
            # KL divergence
            X_safe = np.maximum(X, eps)
            return np.sum(X_safe * np.log(X_safe / X_hat) - X_safe + X_hat)

        elif np.isclose(alpha, 2.0):
            # Euclidean / Frobenius
            return 0.5 * np.sum((X - X_hat) ** 2)

        elif np.isclose(alpha, 0.0):
            # Itakura-Saito
            X_safe = np.maximum(X, eps)
            return np.sum(X_safe / X_hat - np.log(X_safe / X_hat) - 1.0)

        else:
            # General α-divergence
            X_safe = np.maximum(X, eps)
            term = (
                (X_safe ** alpha) * (X_hat ** (1 - alpha))
                - alpha * X_safe
                + (alpha - 1) * X_hat
            )
            return np.sum(term) / (alpha * (alpha - 1))