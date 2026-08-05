import numpy as np
from scipy.special import psi, gammaln, logsumexp
from scipy.special import gamma as gamma_func
from scipy.special import digamma as digamma_func

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "6"

from sklearn.cluster import KMeans
from scipy.stats import wishart
from scipy.stats import betanbinom as BNB

# A 'tight' implementation of the CAVI for Dynamic MFMFA algorithm.

class CAVI_MFMFA:
    """
    Coordinate Ascent Variational Inference for a
    Dynamic Mixture of Finite Mixtures of Factor Analysers.
    
    This implementation follows the manuscript, using the 'shared component'
    assumption for the variational distribution.
    """
    def __init__(self, Y, T, H, a0=1, b0=0.3, a_xi=1., b_xi=1.):
        """
        Initialize the model and variational parameters.

        Args:
            Y (np.ndarray): The data, shape (N, p).
            T (int): Truncation level for the number of components K.
            H (int): Number of factors for each component.
            a0, b0 (float): Hyperparameters for Gamma prior on alpha_eta.
            a_xi, b_xi (float): Hyperparameters for Gamma prior on xi precision.
        """
        self.Y = Y
        self.N, self.p = Y.shape
        self.T = T
        self.H = H

        # Hyperparameters
        self.a0, self.b0 = a0, b0
        self.b0_mu = np.zeros(self.p)
        self.B0_mu_inv = np.eye(self.p) * 1e-6 # Vague prior
        self.a_xi, self.b_xi = a_xi, b_xi
        
        self.alpha_lambda = 1
        self.alpha_pi = 4
        self.beta_pi = 3
        BNBprob = np.zeros(T)
        for t in range(T):
            BNBprob[t] = BNB.pmf(t, self.alpha_lambda, self.alpha_pi, self.beta_pi)
        self.kappa_prior = BNBprob/np.sum(BNBprob)
        #self.kappa_prior = np.ones(self.T) / np.sum(np.ones(self.T)) # uniform prior
        
        self.logkappa_prior = np.log(self.kappa_prior)
        #print(f"logprior:{np.log(self.kappa_prior)}")
        
        # --- Variational Parameters Initialization ---
        
                # q(mu_k): Normal distributions (T of them)
        self.m_mu_k = np.zeros((self.T, self.p))
        self.Sigma_mu_k = np.array([np.eye(self.p) for _ in range(self.T)])
                           
        # Using KMeans for a smarter start
        print("Initializing parameters with KMeans...")
        
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial.distance import cdist

        # 1. Run KMeans for all k and extract both labels and centers
        all_centers = []
        initial_labels = []
        
        for k in range(1, self.T + 1):
            kmeans = KMeans(n_clusters=min(k, self.N), n_init=10, random_state=42)
            labels = kmeans.fit_predict(self.Y)
            initial_labels.append(labels)
            all_centers.append(kmeans.cluster_centers_)
            
        # 2. Use the largest model (k = T) as the anchor reference
        ref_centers = all_centers[-1] # shape (T, p)
        
        # Track sums and counts to compute the average for each of the T components
        sum_mu = np.copy(ref_centers)
        count_mu = np.ones(self.T)
        
        initial_relabels = list(initial_labels)  

        # 3. Align smaller models (k < T) to the reference model
        for k_idx in range(self.T - 1):
            k = k_idx + 1
            centers = all_centers[k_idx] # shape (k, p)
            
            # Compute distance matrix between current centers and reference centers
            cost_matrix = cdist(centers, ref_centers, metric='euclidean')
            
            # Find optimal unique matching: maps row (current cluster) to col (ref cluster)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Create a mapping dictionary for labels: current_label -> ref_label
            label_mapping = dict(zip(row_ind, col_ind))
            
            # Align the initial labels for this model size to match the reference slots
            initial_relabels[k_idx] = np.array([label_mapping[l] for l in initial_relabels[k_idx]])
            
            # Accumulate centers into their matched reference slots
            for r, c in zip(row_ind, col_ind):
                sum_mu[c] += centers[r]
                count_mu[c] += 1

        # 4. Initialize self.m_mu_k with the matched averages
        self.m_mu_k = sum_mu / count_mu[:, np.newaxis] 
        #print(f"initial_means:{self.m_mu_k}")

        # q(K): Categorical distribution over {1, ..., T}
        self.kappa = self.kappa_prior
        
        # q(alpha_eta): Gamma distribution
        self.a_eta_hat = self.a0
        self.b_eta_hat = self.b0

        # q(eta^[k]): Dirichlet distributions
        E_alpha_eta = self.a_eta_hat / self.b_eta_hat
        self.alpha_eta_k_hat = [(E_alpha_eta / k) * np.ones(k) for k in range(1, self.T + 1)]

        # q(S_n^[k]): Responsibilities for each model size k
        self.r_nk = [np.zeros((self.N, k)) for k in range(1, self.T + 1)]
        self.log_rho_nk = [np.zeros((self.N, k)) for k in range(1, self.T + 1)] 
        for kappa in range(0, self.T):
                for k in range(0, kappa):
                    for n in range(0, self.N):
                        self.r_nk[kappa][n, k] = 1/(k + 1)
        
        # q(Lambda_k): Factor loadings (T of them)
        self.m_lambda_k = np.random.randn(self.T, self.p, self.H) * 0.1
        self.Sigma_lambda_k = np.array([np.eye(self.H) for _ in range(self.T * self.p)]) \
                                  .reshape(self.T, self.p, self.H, self.H)

        # q(Xi_k): Idiosyncratic variances (T of them)
        self.a_xi_k_hat = np.ones((self.T, self.p)) * 2.0
        self.b_xi_k_hat = np.ones((self.T, self.p)) * 1.0

        # Pre-computed expectations (to be updated each iteration)
        self.E_Omega_k_inv = np.array([np.eye(self.p) for _ in range(self.T)])
        self.E_log_det_Omega_k = np.zeros(self.T)
        print("Initialization complete.")


    def _update_q_S(self):
        """Update variational pmf of allocations S_n^[k] (responsibilities)."""
        E_log_eta_k = [psi(self.alpha_eta_k_hat[k-1]) - psi(np.sum(self.alpha_eta_k_hat[k-1]))
                       for k in range(1, self.T + 1)]
        
        # E[(y-mu_k)^T Omega_k^-1 (y-mu_k)]
        # = Tr(E[Omega_k^-1] E[mu_k mu_k^T]) - 2 E[y^T Omega_k^-1 mu_k] + E[y^T Omega_k^-1 y]
        E_mu_mu_T = self.Sigma_mu_k + np.einsum('ki,kj->kij', self.m_mu_k, self.m_mu_k)
        term1 = np.einsum('kij,kji->k', self.E_Omega_k_inv, E_mu_mu_T)
        term2 = -2 * np.dot(self.Y, self.E_Omega_k_inv.transpose(0, 2, 1)) # N x T x p
        term2 = np.einsum('nti,ti->nt', term2, self.m_mu_k)
        term3 = np.einsum('ni,kij,nj->nk', self.Y, self.E_Omega_k_inv, self.Y)
        E_mahalanobis = term1[np.newaxis, :] + term2 + term3
        for k in range(1, self.T + 1):
            log_rho_nk = E_log_eta_k[k-1][np.newaxis, :k] - 0.5 * self.E_log_det_Omega_k[np.newaxis, :k] \
                         - 0.5 * E_mahalanobis[:, :k]
            #print(f"log_rho_nk:{log_rho_nk}")
            
            # Stabilize and normalize
            log_r_nk = log_rho_nk - logsumexp(log_rho_nk, axis=1, keepdims=True)
            self.r_nk[k-1] = np.exp(log_r_nk)


    def _update_q_eta(self):
        """Update variational pdf of mixture weights eta^[k]."""
        E_alpha_eta = self.a_eta_hat / self.b_eta_hat
        for k in range(1, self.T + 1):
            N_k = self.r_nk[k-1].sum(axis=0) # shape (k,)
            self.alpha_eta_k_hat[k-1] = (E_alpha_eta / k) + N_k

    def _update_q_alpha_eta(self):
        """Update variational pdf of concentration parameter alpha_eta."""
        E_alpha_eta = self.a_eta_hat / self.b_eta_hat
        expect_K = 0
        a_term = 0
        b_term = 0
        for k in range(1, self.T + 1):
            expect_K += self.kappa[k-1] * k
            a_term += self.kappa[k-1] * psi(E_alpha_eta / k + 1)
            E_log_eta = psi(self.alpha_eta_k_hat[k-1]) - psi(np.sum(self.alpha_eta_k_hat[k-1]))
            b_term += self.kappa[k-1] * (1/k) * np.sum(E_log_eta)
        self.a_eta_hat = self.a0 + expect_K - 1 + E_alpha_eta * (psi(E_alpha_eta + 1) - a_term)
        self.b_eta_hat = self.b0 - b_term

    def _update_q_K(self):
        """Update variational pmf of the number of components K."""
        log_tilde_kappa = np.zeros(self.T)
        log_tilde_kappa_alt = np.zeros(self.T)
        E_alpha_eta = self.a_eta_hat / self.b_eta_hat
        
        # Approx for E[log Gamma(x)] ~= gammaln(E[x])
        E_log_Gamma_alpha = gammaln(E_alpha_eta)

        for k in range(1, self.T + 1):
            E_log_eta = psi(self.alpha_eta_k_hat[k-1]) - psi(np.sum(self.alpha_eta_k_hat[k-1]))
            E_log_Gamma_alpha_k = gammaln(E_alpha_eta / k + 1)
            per_point_log_evidence = logsumexp(self.log_rho_nk[k - 1], axis=1)
            data_fit_term = np.sum(per_point_log_evidence)
            alpha_eta_terms = k * (psi(self.a_eta_hat) - np.log(self.b_eta_hat) ) - k * np.log(k) - k * E_log_Gamma_alpha_k - E_alpha_eta * psi(E_alpha_eta / k + 1) * (psi(self.a_eta_hat) - np.log(self.b_eta_hat) - np.log(E_alpha_eta)) + (E_alpha_eta / k - 1) * np.sum(E_log_eta)
            log_tilde_kappa[k-1] = self.logkappa_prior[k-1] + alpha_eta_terms + data_fit_term
                               
            #print(f"+++++\nkappa = {k}\n+++++")    
            #print(f"alpha_eta_terms:{alpha_eta_terms}")
            #print(f"alpha_eta_terms:{alpha_eta_terms}")
            #print(f"data_fit_term:{data_fit_term}")
            #print(f"prior_term:{self.logkappa_prior[k-1]}")

        log_tilde_kappa = log_tilde_kappa - logsumexp(log_tilde_kappa)
        self.kappa = np.exp(log_tilde_kappa)
        self.kappa =  np.ones(self.T) / np.sum(np.ones(self.T))
        
        #print(f"K_pmf1: {self.kappa}")
        #print(f"K_pmf2: {self.kappa_alt}")
    

    def _update_q_mu(self):
        """Update variational pdf of cluster means mu_k."""
        # Calculate weighted statistics
        weighted_N_k = np.zeros(self.T)
        weighted_Y_sum_k = np.zeros((self.T, self.p))

        for k in range(1, self.T + 1):
            N_k_kappa = self.r_nk[k-1].sum(axis=0) # shape (k,)
            Y_sum_k_kappa = self.r_nk[k-1].T @ self.Y # shape (k, p)
            
            weighted_N_k[:k] += self.kappa[k-1] * N_k_kappa
            weighted_Y_sum_k[:k, :] += self.kappa[k-1] * Y_sum_k_kappa
        
        for k in range(self.T):
            precision_k = self.B0_mu_inv + weighted_N_k[k] * self.E_Omega_k_inv[k]
            self.Sigma_mu_k[k] = np.linalg.inv(precision_k)
            
            mean_term = self.B0_mu_inv @ self.b0_mu + self.E_Omega_k_inv[k] @ weighted_Y_sum_k[k]
            self.m_mu_k[k] = self.Sigma_mu_k[k] @ mean_term

    def _update_q_Omega(self):
        """Update variational pdfs for FA parameters Lambda_k and Xi_k."""
        # Calculate weighted sufficient statistics
        # weighted_N_k = [np.zeros((self.T, k)) for k in range(1, self.T + 1)]
        # for k_model in range(1, self.T + 1):
        #     weighted_N_k[k_model-1] = self.r_nk[k_model-1].sum(axis=0)
            
        # E_mu_mu_T = self.Sigma_mu_k + np.einsum('ki,kj->kij', self.m_mu_k, self.m_mu_k)
        
        # for kappa in range(self.T):
        #     for k in range(kappa):
            
        #         if weighted_N_k[kappa, k] < 1e-6: continue # Skip empty clusters

        #         # Aggregate responsibilities for this component
        #         r_nk_agg = np.zeros(self.N)
        #         r_nk_agg = self.r_nk[k_model-1][:, k]
                
        #     y_centered = self.Y - self.m_mu_k[kappa, k]
            
        #     # --- Update q(Lambda_k) ---
        #     E_Xi_inv = self.a_xi_k_hat[k] / self.b_xi_k_hat[k]
            
        #     for j in range(self.p):
        #         S_lambda_inv = np.eye(self.H) + weighted_N_k[k] * E_Xi_inv[j] * \
        #                        (self.m_lambda_k[k, j, :].T @ self.m_lambda_k[k, j, :]) # approx
        #         self.Sigma_lambda_k[k, j] = np.linalg.inv(S_lambda_inv)
                
        #         sum_term = np.sum([r_nk_agg[n] * y_centered[n, j] * self.m_lambda_k[k,j,:] for n in range(self.N)], axis=0) # approx
        #         self.m_lambda_k[k, j] = E_Xi_inv[j] * self.Sigma_lambda_k[k, j] @ sum_term

        #     # --- Update q(Xi_k) ---
        #     E_Lambda_Lambda_T = self.Sigma_lambda_k[k] + \
        #                         np.einsum('ih,ij->ihj', self.m_lambda_k[k], self.m_lambda_k[k])
            
        #     # S_k diagonal elements: shape (p,)
        #     diag_S_k = np.sum((r_nk_agg[:, np.newaxis] * y_centered) * y_centered, axis=0)
            
        #     # Cross term diagonal elements: shape (p,)
        #     # Replaces the broken matrix multiplication with an explicit row-by-row dot product
        #     weighted_y_sum = y_centered.T @ r_nk_agg # shape (p,)
        #     diag_cross_term = 2 * np.einsum('jh,j->j', self.m_lambda_k[k], weighted_y_sum) # Note: review if factor expectations E[Z] are missing here
            
        #     # Expected value of Lambda * Lambda^T trace for each j: shape (p,)
        #     diag_lambda_term = np.einsum('jhh->j', E_Lambda_Lambda_T)

        #     self.a_xi_k_hat[k] = self.a_xi + weighted_N_k[k] / 2
            
        #     # Combine them directly into the (p,) vector
        #     diag_term = diag_S_k - diag_cross_term + weighted_N_k[k] * diag_lambda_term            
        #     self.b_xi_k_hat[k] = np.clip(self.b_xi + 0.5 * diag_term, a_min=1e-10, a_max=None)
            
        #     # --- Update expectations for Omega_k ---
        #     E_Xi_inv = self.a_xi_k_hat[k] / self.b_xi_k_hat[k]
        #     E_log_Xi = psi(self.a_xi_k_hat[k]) - np.log(self.b_xi_k_hat[k])
            
        #     E_Lambda = self.m_lambda_k[k]
        #     E_Lambda_T_Xi_inv_Lambda = E_Lambda.T @ np.diag(E_Xi_inv) @ E_Lambda # approx
            
        #     M = np.eye(self.H) + E_Lambda_T_Xi_inv_Lambda
        #     M_inv = np.linalg.inv(M)
            
        #     self.E_Omega_k_inv[k] = np.diag(E_Xi_inv) - np.diag(E_Xi_inv) @ E_Lambda @ M_inv @ E_Lambda.T @ np.diag(E_Xi_inv)
        #     self.E_log_det_Omega_k[k] = np.linalg.slogdet(M)[1] + np.sum(E_log_Xi)

    def compute_elbo(self):
        """Compute the Evidence Lower Bound (ELBO) to track convergence."""
        # # A simplified ELBO. A full one is very lengthy.
        # # E[log p(Y|...)]
        # expected_log_likelihood = 0
        # for k_model in range(1, self.T + 1):
        #     log_lik_k_model = 0
        #     for k_comp in range(k_model):
        #         E_mahalanobis = np.sum((self.Y - self.m_mu_k[k_comp]) * (self.E_Omega_k_inv[k_comp] @ (self.Y - self.m_mu_k[k_comp]).T).T, axis=1)
        #         log_lik_k_comp = -0.5 * (self.p * np.log(2*np.pi) + self.E_log_det_Omega_k[k_comp] + E_mahalanobis)
        #         log_lik_k_model += self.r_nk[k_model-1][:, k_comp] * log_lik_k_comp
        #     expected_log_likelihood += self.kappa[k_model-1] * np.sum(log_lik_k_model)
        
        # # - E[log q(S)]
        # entropy_S = 0
        # for k_model in range(1, self.T + 1):
        #     entropy_S -= self.kappa[k_model-1] * np.sum(self.r_nk[k_model-1] * np.log(self.r_nk[k_model-1] + 1e-9))

        # return expected_log_likelihood + entropy_S

    def fit(self, max_iter=100, tol=1e-5, verbose=False):
        """Run the CAVI algorithm."""
        elbo_history = []
        respon_history = []
        for i in range(max_iter):
            # Update all parameters
            self._update_q_Omega()
            if np.any(np.isnan(self.E_Omega_k_inv)): print(f"NaN in Omega at Iter {i}"); break
                
            self._update_q_mu()
            if any(np.isnan(mu).any() for mu in self.m_mu_k): print(f"NaN in mu at Iter {i}"); break                
            respon_old = self.r_nk.copy()
            self._update_q_S()
            if any(np.any(np.isnan(r)) for r in self.r_nk): print(f"NaN in S at Iter {i}"); break            
            self._update_q_eta()
            self._update_q_alpha_eta()
            self._update_q_K()

            # Compute ELBO
            elbo = self.compute_elbo()
            elbo_history.append(elbo)

            if verbose:
                print(f"\n--- Iteration {i+1} Diagnostics ---")
                #print(f"Kappa pmf:\n{self.kappa}")
                #print(f"Kappa range:       [{self.kappa:.4f}, {self.kappa.max():.4f}]")
                #print(f"Responsibilities:         \n{self.r_nk[1][0,0:5]}")
                #for kappa in range(1, self.T):
                #print(f"Eta:         \n{(model.alpha_eta_k_hat[9] - 1) / (np.sum(model.alpha_eta_k_hat[9]) + 1)}")
                #print(f"a_eta: {self.a_eta_hat} b_eta: {self.b_eta_hat}")
                #print(f"Assignments: {np.bincount(np.argmax(self.r_nk[4], axis=1))}")
                #print(f"Cluster centers: \n{model.m_mu_k}")
                for kappa in range(1, T_truncation + 1):
                    print(f"Assignments for kappa={kappa}: {np.bincount(np.argmax(model.r_nk[kappa - 1], axis=1))}")



            #if verbose:
            #    print(f"Iter {i+1}/{max_iter}, ELBO: {elbo:.4f}, Most likely K: {np.argmax(self.kappa) + 1}")
            
            # Check for convergence
            #if i > 0 and np.abs(elbo - elbo_history[-2]) < tol:
            if i > 0:
                abs_change = []
                for k in range(1, self.T):
                    abs_change.append(np.max(np.abs(self.r_nk[k - 1] - respon_old[k - 1]))) 
                abs_change_max = np.max(abs_change)
                if abs_change_max < tol:
                    print("Converged.")
                    break
        
        return elbo_history

# --- Example Usage ---
if __name__ == '__main__':
    # 1. Generate synthetic data
    N = 30 # Number of data points
    p = 2   # Dimensionality
    
    # True parameters for 3 clusters
    true_K = 3
    true_means = np.array([
        [3] * p,
        [-3] * p,
        [0] * p
    ])
    
    # Simple covariance structures for illustration
    true_covs = [
        np.eye(p) * 1.0,
        np.eye(p) * 1.0,
        np.eye(p) * 1.0
    ]
    
    true_weights = np.array([0.4, 0.4, 0.2])
    
    Y = np.zeros((N, p))
    labels = np.random.choice(true_K, size=N, p=true_weights)
    for k in range(true_K):
        indices = np.where(labels == k)[0]
        Y[indices] = np.random.multivariate_normal(true_means[k], true_covs[k], size=len(indices))

#    print(f"Generated means are {true_means} and covar are {true_covs}")
    print(f"Generated data with N={N}, p={p}, true K={true_K}")

    # 2. Run the CAVI algorithm
    T_truncation = 10 # Set truncation level higher than true K
    H_factors = 3    # Max number of factors
    
    model = CAVI_MFMFA(Y, T=T_truncation, H=H_factors)
    #np.seterr(all='raise')
    elbo_history = model.fit(max_iter=100, tol=1e-4)

    # 3. Inspect the results
    print("\n--- Results ---")
    print(f"\n--({len(elbo_history)} iterations)--")
    print(f"Final posterior over K (kappa):\n{np.round(model.kappa, 3)}")
    final_K = np.argmax(model.kappa) + 1
    #print(f"Most probable number of clusters: {final_K}")
    # Get final cluster assignments based on the most likely model size
    final_assignments = np.argmax(model.r_nk[final_K - 1], axis=1)
    #print(f"Number of points assigned to each cluster: {np.bincount(final_assignments)}")
    print(f"Cluster centers: \n{model.m_mu_k}")
    
    # for kappa in range(1, T_truncation + 1):
    #     print(f"Cluster centers for kappa={kappa}: {model.m_mu_k[kappa - 1]}")
    # for kappa in range(1, T_truncation + 1):
    #     print(f"Weights for kappa={kappa}: {(model.alpha_eta_k_hat[kappa - 1] - 1) / (np.sum(model.alpha_eta_k_hat[kappa - 1]) + 1)}")
    for kappa in range(1, T_truncation + 1):
        print(f"Assignments for kappa={kappa}: {np.bincount(np.argmax(model.r_nk[kappa - 1], axis=1))}")
