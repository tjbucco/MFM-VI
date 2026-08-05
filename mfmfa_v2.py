import numpy as np
from scipy.special import psi, gammaln, logsumexp
from sklearn.cluster import KMeans
from scipy.stats import betanbinom as BNB
from scipy.linalg import cholesky, solve_triangular

# Set environment variables to prevent over-threading with numpy/scipy
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "6"

class CAVI_MFM:
    """
    Coordinate Ascent Variational Inference for a
    Dynamic Mixture of Finite Mixtures (MFM) of Gaussians.
    
    This implementation follows the manuscript, using the 'shared component'
    assumption for the variational distribution and a Normal-Inverse-Wishart
    model for the cluster parameters (mean and covariance).
    """
    def __init__(self, Y, T, a0=1.0, b0=0.3, lambda0=1e-6, nu0=None, b0_mu=None, B0=None):
        """
        Initialize the model and variational parameters.

        Args:
            Y (np.ndarray): The data, shape (N, p).
            T (int): Truncation level for the number of components K.
            a0, b0 (float): Hyperparameters for Gamma prior on alpha_eta.
            lambda0 (float): Hyperparameter for the Normal-Inverse-Wishart prior.
            nu0 (float): Hyperparameter for the Normal-Inverse-Wishart prior.
            b0_mu (np.ndarray): Hyperparameter for the Normal-Inverse-Wishart prior.
            B0 (np.ndarray): Hyperparameter for the Normal-Inverse-Wishart prior.
        """
        self.Y = Y
        self.N, self.p = Y.shape
        self.T = T

        # --- Hyperparameters ---
        self.a0, self.b0 = a0, b0
        
        # Hyperparameters for Normal-Inverse-Wishart prior p(mu, Omega)
        self.lambda0 = lambda0
        self.b0_mu = np.mean(Y, axis=0) if b0_mu is None else b0_mu
        self.nu0 = self.p + 2 if nu0 is None else nu0
        if B0 is None:
            # A reasonable default for B0 based on data covariance
            emp_cov = np.cov(Y, rowvar=False)
            if self.p == 1: emp_cov = np.array([[emp_cov]])
            self.B0 = emp_cov * (self.nu0 - self.p - 1)
        else:
            self.B0 = B0

        # Prior for K (number of components)
        self.alpha_lambda_prior = 1
        self.alpha_pi_prior = 4
        self.beta_pi_prior = 3
        BNBprob = np.array([BNB.pmf(t, self.alpha_lambda_prior, self.alpha_pi_prior, self.beta_pi_prior) for t in range(T)])
        self.log_kappa_prior = np.log(BNBprob / np.sum(BNBprob))
        
        # --- Variational Parameters Initialization ---
        
        # q(K): Categorical distribution over {1, ..., T}
        self.kappa_log_prob = np.copy(self.log_kappa_prior)
        self.kappa_prob = np.exp(self.kappa_log_prob)
        
        # q(alpha_eta): Gamma distribution
        self.a_eta_hat = self.a0
        self.b_eta_hat = self.b0

        # q(eta^[k]): Dirichlet distributions for k=1..T
        E_alpha_eta = self.a_eta_hat / self.b_eta_hat
        self.alpha_eta_k_hat = [(E_alpha_eta / k) * np.ones(k) for k in range(1, self.T + 1)]

        # q(S_n^[k]): Responsibilities r_nk^[k] for each model size k
        self.r_nk = [np.zeros((self.N, k)) for k in range(1, self.T + 1)]
        
        # q(mu_k, Omega_k): Normal-Inverse-Wishart distributions (T of them)
        self.hat_lambda_k = np.zeros(self.T)
        self.hat_nu_k = np.zeros(self.T)
        self.hat_b_k = np.zeros((self.T, self.p))
        self.hat_B_k = np.zeros((self.T, self.p, self.p))

        print("Initializing parameters with KMeans...")
        self._initialize_with_kmeans()

        # Pre-computed expectations (to be updated each iteration)
        self.E_mu_k = np.zeros((self.T, self.p))
        self.E_Omega_k_inv = np.zeros((self.T, self.p, self.p))
        self.E_log_det_Omega_k = np.zeros(self.T)
        self.trace_term = np.zeros(self.T)
        self._compute_expectations() # Compute initial expectations
        
        print("Initialization complete.")

    def _initialize_with_kmeans(self):
        """Use KMeans to provide a smart initialization for variational parameters."""
        # Use the largest possible model (T clusters) for a stable start
        n_clusters_init = min(self.T, self.N)
        kmeans = KMeans(n_clusters=n_clusters_init, n_init=10, random_state=42)
        labels = kmeans.fit_predict(self.Y)
        
        for k in range(self.T):
            if k < n_clusters_init:
                points_in_k = self.Y[labels == k]
                Nk = points_in_k.shape[0]
                if Nk > 0:
                    y_bar_k = np.mean(points_in_k, axis=0)
                    Sk = (points_in_k - y_bar_k).T @ (points_in_k - y_bar_k)
                else: # Empty cluster
                    Nk = 1e-6 # Avoid division by zero
                    y_bar_k = self.b0_mu
                    Sk = np.zeros((self.p, self.p))

                # Initialize responsibilities for the largest model
                self.r_nk[self.T-1][labels==k, k] = 1.0

                # Initialize NW-IW params based on manuscript update rules
                self.hat_lambda_k[k] = self.lambda0 + Nk
                self.hat_nu_k[k] = self.nu0 + Nk
                self.hat_b_k[k] = (self.lambda0 * self.b0_mu + Nk * y_bar_k) / self.hat_lambda_k[k]
                self.hat_B_k[k] = self.B0 + Sk + (self.lambda0 * Nk / self.hat_lambda_k[k]) * \
                                   np.outer(y_bar_k - self.b0_mu, y_bar_k - self.b0_mu)
            else: # For components beyond what KMeans found
                self.hat_lambda_k[k] = self.lambda0
                self.hat_nu_k[k] = self.nu0
                self.hat_b_k[k] = self.b0_mu
                self.hat_B_k[k] = self.B0

        # Propagate responsibilities to smaller models for a reasonable start
        for kappa in range(1, self.T):
            self.r_nk[kappa-1] = self.r_nk[self.T-1][:, :kappa]
            row_sums = self.r_nk[kappa-1].sum(axis=1, keepdims=True)
            self.r_nk[kappa-1] /= np.where(row_sums > 0, row_sums, 1)

    def _compute_expectations(self):
        """
        Compute required expectations from the variational distributions.
        This is called after updating the variational parameters for mu_k and Omega_k.
        """
        # E[mu_k] from q(mu_k) is simply the mean of the Normal distribution
        self.E_mu_k = self.hat_b_k
        
        for k in range(self.T):
            # E[Omega_k^-1] from q(Omega_k) ~ IW(B_k_hat, nu_k_hat)
            self.E_Omega_k_inv[k] = self.hat_nu_k[k] * np.linalg.inv(self.hat_B_k[k])
            
            # E[log|Omega_k|]
            log_det_B_k = np.linalg.slogdet(self.hat_B_k[k])[1]
            digamma_term = np.sum(psi(0.5 * (self.hat_nu_k[k] + 1 - np.arange(1, self.p + 1))))
            self.E_log_det_Omega_k[k] = self.p * np.log(2) + log_det_B_k - digamma_term
            
            # Trace term from the Mahalanobis distance expansion
            # Tr(E[Omega_k^-1] Cov_q(mu_k))
            # Cov_q(mu_k) = (1/lambda_k_hat) * E[Omega_k]
            # E[Omega_k] = B_k_hat / (nu_k_hat - p - 1)
            # --> Tr(nu_k_hat * inv(B_k_hat) * (1/lambda_k_hat) * B_k_hat / (nu_k_hat - p - 1))
            if self.hat_nu_k[k] > self.p + 1:
                self.trace_term[k] = self.p * self.hat_nu_k[k] / (self.hat_lambda_k[k] * (self.hat_nu_k[k] - self.p - 1))
            else:
                # Fallback for numerical stability if nu is too small
                self.trace_term[k] = self.p / self.hat_lambda_k[k]

    def _update_q_S(self):
        """Update variational pmf of allocations S_n^[k] (responsibilities r_nk)."""
        E_log_eta_k = [psi(self.alpha_eta_k_hat[k-1]) - psi(np.sum(self.alpha_eta_k_hat[k-1]))
                       for k in range(1, self.T + 1)]
        
        # Calculate the expected Mahalanobis distance for all N points and T components
        # E[(y-mu_k)^T Omega_k^-1 (y-mu_k)] =
        # (y-E[mu_k])^T E[Omega_k^-1] (y-E[mu_k]) + Tr(E[Omega_k^-1] Cov_q(mu_k))
        E_mahalanobis = np.zeros((self.N, self.T))
        for k in range(self.T):
            Y_centered = self.Y - self.E_mu_k[k] # N x p
            quadratic_term = np.sum((Y_centered @ self.E_Omega_k_inv[k]) * Y_centered, axis=1) # N,
            E_mahalanobis[:, k] = quadratic_term + self.trace_term[k]

        # Update responsibilities for each model size kappa
        for kappa in range(1, self.T + 1):
            log_rho_nk = (E_log_eta_k[kappa-1][np.newaxis, :kappa]
                          - 0.5 * self.E_log_det_Omega_k[np.newaxis, :kappa]
                          - 0.5 * E_mahalanobis[:, :kappa])
            
            # Stabilize and normalize
            log_r_nk = log_rho_nk - logsumexp(log_rho_nk, axis=1, keepdims=True)
            self.r_nk[kappa-1] = np.exp(log_r_nk)

    def _update_q_eta(self):
        """Update variational pdf of mixture weights eta^[k]."""
        E_alpha_eta = self.a_eta_hat / self.b_eta_hat
        for k in range(1, self.T + 1):
            Nk_kappa = self.r_nk[k-1].sum(axis=0) # shape (k,)
            self.alpha_eta_k_hat[k-1] = (E_alpha_eta / k) + Nk_kappa

    def _update_q_alpha_eta(self):
        """Update variational pdf of concentration parameter alpha_eta using Taylor approx."""
        E_alpha_eta = self.a_eta_hat / self.b_eta_hat
        if E_alpha_eta <= 0: E_alpha_eta = 1e-6 # Safeguard

        E_K = np.sum(self.kappa_prob * np.arange(1, self.T + 1))
        
        psi_term_sum = np.sum([self.kappa_prob[k-1] * psi(E_alpha_eta / k + 1) for k in range(1, self.T + 1)])
        
        log_eta_term_sum = np.sum([
            self.kappa_prob[k-1] * (1/k) * np.sum(psi(self.alpha_eta_k_hat[k-1]) - psi(np.sum(self.alpha_eta_k_hat[k-1])))
            for k in range(1, self.T + 1)
        ])

        A = E_alpha_eta * (psi(E_alpha_eta + 1) - psi_term_sum)
        
        self.a_eta_hat = self.a0 + E_K - 1 + A
        self.b_eta_hat = self.b0 - log_eta_term_sum
        
        if self.a_eta_hat <= 0: self.a_eta_hat = 1e-6
        if self.b_eta_hat <= 0: self.b_eta_hat = 1e-6

    def _update_q_K(self):
        """Update variational pmf of the number of components K."""
        log_tilde_kappa = np.zeros(self.T)
        E_alpha_eta = self.a_eta_hat / self.b_eta_hat
        if E_alpha_eta <= 0: E_alpha_eta = 1e-6 # Safeguard
        
        E_log_alpha_eta = psi(self.a_eta_hat) - np.log(self.b_eta_hat)

        for kappa in range(1, self.T + 1):
            # Term from p(eta | alpha, K)
            log_p_eta_term = (gammaln(E_alpha_eta) - kappa * gammaln(E_alpha_eta / kappa) +
                              (E_alpha_eta / kappa - 1) * np.sum(psi(self.alpha_eta_k_hat[kappa-1]) - psi(np.sum(self.alpha_eta_k_hat[kappa-1]))))

            # Data fit term from p(y, S | eta, mu, Omega)
            log_lik_per_point = logsumexp(
                np.log(self.r_nk[kappa-1] + 1e-300), # Using log of responsibilities
                axis=1
            )
            data_fit_term = np.sum(log_lik_per_point)
            
            # The manuscript uses a more complex Taylor expansion for the update of K.
            # A simpler, often effective approximation is to plug in expectations of other variables.
            # Let's use the simpler version for stability.
            log_tilde_kappa[kappa-1] = self.log_kappa_prior[kappa-1] + log_p_eta_term + data_fit_term
                               
        self.kappa_log_prob = log_tilde_kappa - logsumexp(log_tilde_kappa)
        self.kappa_prob = np.exp(self.kappa_log_prob)

    def _update_q_mu_Omega(self):
        """Update variational pdf of cluster params (mu_k, Omega_k) for all k."""
        # Calculate effective responsibilities w_nk (shape N x T)
        w_nk = np.zeros((self.N, self.T))
        for kappa in range(1, self.T + 1):
            w_nk[:, :kappa] += self.kappa_prob[kappa-1] * self.r_nk[kappa-1]

        # Calculate weighted sufficient statistics
        N_prime_k = w_nk.sum(axis=0) # shape (T,)
        
        # Avoid division by zero for empty/unused components
        safe_N_prime_k = np.where(N_prime_k > 1e-9, N_prime_k, 1e-9)
        
        y_bar_prime_k = (w_nk.T @ self.Y) / safe_N_prime_k[:, np.newaxis] # shape (T, p)

        for k in range(self.T):
            if N_prime_k[k] < 1e-9:
                # Reset unused components to prior
                self.hat_lambda_k[k] = self.lambda0
                self.hat_nu_k[k] = self.nu0
                self.hat_b_k[k] = self.b0_mu
                self.hat_B_k[k] = self.B0
                continue

            # Weighted covariance matrix S'_k
            y_centered = self.Y - y_bar_prime_k[k]
            S_prime_k = (y_centered.T * w_nk[:, k]) @ y_centered

            # Update NW-IW parameters using manuscript equations
            self.hat_lambda_k[k] = self.lambda0 + N_prime_k[k]
            self.hat_nu_k[k] = self.nu0 + N_prime_k[k]
            self.hat_b_k[k] = (self.lambda0 * self.b0_mu + N_prime_k[k] * y_bar_prime_k[k]) / self.hat_lambda_k[k]
            
            term4_num = self.lambda0 * N_prime_k[k]
            term4_den = self.hat_lambda_k[k]
            y_diff = y_bar_prime_k[k] - self.b0_mu
            
            self.hat_B_k[k] = self.B0 + S_prime_k + (term4_num / term4_den) * np.outer(y_diff, y_diff)
    
    def fit(self, max_iter=100, tol=1e-5, verbose=False):
        """Run the CAVI algorithm."""
        responsibilities_history = []
        for i in range(max_iter):
            # Store old responsibilities for convergence check
            r_nk_old = [r.copy() for r in self.r_nk]
            
            # --- CAVI Updates ---
            self._update_q_S()
            self._update_q_eta()
            self._update_q_alpha_eta()
            self._update_q_K()
            self._update_q_mu_Omega()
            
            # This must be called after updating q(mu,Omega) to propagate expectations
            self._compute_expectations()

            if verbose:
                print(f"\n--- Iteration {i+1} ---")
                print(f"Posterior p(K): {np.round(self.kappa_prob, 3)}")
                # Show assignments for the most probable K
                most_probable_K = np.argmax(self.kappa_prob) + 1
                assignments = np.argmax(self.r_nk[most_probable_K-1], axis=1)
                print(f"Assignments for K={most_probable_K}: {np.bincount(assignments, minlength=most_probable_K)}")

            # Check for convergence based on max change in responsibilities
            if i > 0:
                max_abs_change = 0
                for kappa in range(self.T):
                    change = np.max(np.abs(self.r_nk[kappa] - r_nk_old[kappa]))
                    if change > max_abs_change:
                        max_abs_change = change
                
                if max_abs_change < tol:
                    print(f"Converged after {i+1} iterations (max change in responsibilities < {tol}).")
                    break
        else:
            print(f"Reached max iterations ({max_iter}).")
        
        # Return final responsibilities for inspection
        return self.r_nk


# --- Example Usage ---
if __name__ == '__main__':
    # 1. Generate synthetic data for arbitrary dimension p
    rng = np.random.default_rng(42)

    N = 150   # Number of data points
    p = 20     # Dimensionality (set this to any positive integer)
    true_K = 3

    # Helper: random SPD covariance of size (p, p)
    def random_spd(p, rng, eig_min=0.3, eig_max=2.0):
        # Random orthogonal matrix via QR
        A = rng.standard_normal((p, p))
        Q, _ = np.linalg.qr(A)
        # Positive eigenvalues
        eigvals = rng.uniform(eig_min, eig_max, size=p)
        return (Q @ np.diag(eigvals) @ Q.T)

    # Generate well-separated means in R^p
    mean_radius = 6.0
    true_means = rng.standard_normal((true_K, p))
    # Normalize to unit vectors and scale so clusters are separated
    norms = np.linalg.norm(true_means, axis=1, keepdims=True) + 1e-12
    true_means = mean_radius * (true_means / norms)

    # Generate random SPD covariances
    true_covs = [random_spd(p, rng) for _ in range(true_K)]

    # Mixture weights
    true_weights = np.array([0.4, 0.4, 0.2])
    true_weights /= true_weights.sum()

    # Sample data
    Y = np.zeros((N, p))
    labels = rng.choice(true_K, size=N, p=true_weights)
    for k in range(true_K):
        idx = np.where(labels == k)[0]
        if len(idx) == 0:
            continue
        Y[idx] = rng.multivariate_normal(true_means[k], true_covs[k], size=len(idx))

    print(f"Generated data with N={N}, p={p}, true K={true_K}")

    # 2. Run the CAVI algorithm
    T_truncation = 10  # Set truncation level higher than true K

    model = CAVI_MFM(Y, T=T_truncation)
    model.fit(max_iter=100, tol=1e-4, verbose=False)

    # 3. Inspect the results
    print("\n--- Final Results ---")
    print(f"Final posterior over K (kappa):\n{np.round(model.kappa_prob, 3)}")
    final_K = np.argmax(model.kappa_prob) + 1
    print(f"Most probable number of clusters: {final_K}")

    # Final cluster assignments based on the most likely model size
    final_assignments = np.argmax(model.r_nk[final_K - 1], axis=1)
    print(f"\nNumber of points assigned to each cluster (for K={final_K}):")
    print(np.bincount(final_assignments, minlength=final_K))

    print("\nEstimated cluster centers (for all T components):")
    # Sort centers by the number of assignments for clarity (using the largest K=T_truncation)
    counts = np.bincount(np.argmax(model.r_nk[T_truncation - 1], axis=1), minlength=T_truncation)
    sorted_indices = np.argsort(-counts)
    print(np.round(model.hat_b_k[sorted_indices], 2))