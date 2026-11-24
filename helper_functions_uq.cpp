#include <RcppArmadillo.h>
#ifdef _OPENMP
#include <omp.h>
#endif
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

// Scalable componentwise variance corrector. Species are processed in blocks,
// so no n x p matrix is held in memory. `observed` may be an empty matrix,
// meaning every cell is observed. The loadings are the last k columns of BL.

static arma::mat cell_weights(const arma::mat& X_tilde, const arma::mat& BL_b,
                              const arma::mat& observed, arma::uword j0) {
  arma::mat mu = 1.0 / (1.0 + arma::exp(-(X_tilde * BL_b.t())));
  arma::mat v  = mu % (1.0 - mu);
  if (observed.n_elem > 0) v %= observed.cols(j0, j0 + BL_b.n_rows - 1);
  return v;
}

// Symmetric inverse that never throws (exceptions must not escape OpenMP
// regions): Cholesky inverse of the symmetrized matrix; if that fails, retry with
// a diagonal jitter eps * mean|diag|, eps = 1e-10, ..., 1e-4, which keeps the
// result positive definite; the pseudo-inverse is the last resort.
static arma::mat sym_inv(const arma::mat& X) {
  const arma::mat Xs = 0.5 * (X + X.t());
  arma::mat R;
  if (arma::inv_sympd(R, Xs)) return R;
  const double s = std::max(arma::mean(arma::abs(Xs.diag())), 1e-12);
  for (double eps = 1e-10; eps <= 1.0001e-4; eps *= 10)
    if (arma::inv_sympd(R, Xs + eps * s * arma::eye(Xs.n_rows, Xs.n_cols))) return R;
  return arma::pinv(Xs);
}

// Solve A X = B for a general square A without throwing: direct solve, then the
// same diagonal jitter ladder as sym_inv, then the pseudo-inverse.
static arma::mat safe_solve(const arma::mat& A, const arma::mat& B) {
  arma::mat X;
  if (arma::solve(X, A, B, arma::solve_opts::no_approx)) return X;
  const double s = std::max(arma::mean(arma::abs(A.diag())), 1e-12);
  for (double eps = 1e-10; eps <= 1.0001e-4; eps *= 10)
    if (arma::solve(X, A + eps * s * arma::eye(A.n_rows, A.n_cols), B, arma::solve_opts::no_approx)) return X;
  return arma::pinv(A) * B;
}

// Row l2_j = vec(lambda_j lambda_j^T), so that  W * L2  stacks
// sum_j w_ij lambda_j lambda_j^T  over sites.
static arma::mat outer_rows(const arma::mat& Lam) {
  const arma::uword k = Lam.n_cols;
  arma::mat L2(Lam.n_rows, k * k);
  for (arma::uword a = 0; a < k; ++a)
    for (arma::uword b = 0; b < k; ++b)
      L2.col(a * k + b) = Lam.col(a) % Lam.col(b);
  return L2;
}

// Conditional covariance of each site latent factor, flattened:
//   V_eta_i = (sum_j v_ij lambda_j lambda_j^T + I_k / v_M)^{-1},  row i = vec(V_eta_i).
// [[Rcpp::export]]
arma::mat uq_site_cov_cpp(const arma::mat& X_tilde, const arma::mat& BL,
                          const arma::mat& observed, int k, double v_M,
                          int block = 5000) {
  const arma::uword n = X_tilde.n_rows, p = BL.n_rows, d = BL.n_cols;
  arma::mat A(n, k * k, arma::fill::zeros);
  for (arma::uword j0 = 0; j0 < p; j0 += block) {
    const arma::uword j1 = std::min<arma::uword>(j0 + block, p) - 1;
    const arma::mat BL_b = BL.rows(j0, j1);
    A += cell_weights(X_tilde, BL_b, observed, j0) * outer_rows(BL_b.cols(d - k, d - 1));
  }
  arma::mat V_eta(n, k * k);
  const arma::mat prior = arma::eye(k, k) / v_M;
  for (arma::uword i = 0; i < n; ++i)
    V_eta.row(i) = arma::vectorise(sym_inv(arma::reshape(A.row(i), k, k) + prior)).t();
  return V_eta;
}

// Per-species diagonals of
//   V_cond = (X~^T diag(v_j) X~ + diag(1/sigma_sq))^{-1}
//   V_eff  = (X~^T diag(v_eff_j) X~ + diag(1/prior_var_j))^{-1},
//   v_eff_ij = v_ij - v_ij^2 lambda_j^T V_eta_i lambda_j   (Schur complement over M).
// [[Rcpp::export]]
List uq_species_diag_cpp(const arma::mat& X_tilde, const arma::mat& BL,
                         const arma::mat& observed, const arma::mat& V_eta,
                         const arma::vec& sigma_sq, const arma::mat& prior_var,
                         int block = 5000, int n_threads = 1) {
  const arma::uword p = BL.n_rows, d = BL.n_cols;
  const arma::uword k = static_cast<arma::uword>(std::sqrt((double) V_eta.n_cols));
  arma::mat V_cond(p, d), V_eff(p, d);
  const arma::vec cond_prec  = 1.0 / sigma_sq;
  const arma::mat prior_prec = 1.0 / prior_var;

  for (arma::uword j0 = 0; j0 < p; j0 += block) {
    const arma::uword j1 = std::min<arma::uword>(j0 + block, p) - 1;
    const arma::mat BL_b = BL.rows(j0, j1);
    const arma::mat v    = cell_weights(X_tilde, BL_b, observed, j0);
    const arma::mat Q    = V_eta * outer_rows(BL_b.cols(d - k, d - 1)).t();  // lambda_j^T V_eta_i lambda_j
    const arma::mat v_eff = arma::clamp(v - (v % v) % Q, 0.0, arma::datum::inf);

    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (arma::uword jj = 0; jj < BL_b.n_rows; ++jj) {
      const arma::uword j = j0 + jj;
      for (int pass = 0; pass < 2; ++pass) {
        arma::vec w, prec;
        if (pass == 0) { w = v.col(jj);     prec = cond_prec; }
        else           { w = v_eff.col(jj); prec = prior_prec.row(j).t(); }
        arma::mat H = (X_tilde.each_col() % w).t() * X_tilde;
        H.diag() += prec;
        arma::mat L;
        arma::rowvec dg;
        if (arma::chol(L, H, "lower")) dg = arma::sum(arma::square(arma::inv(arma::trimatl(L))), 0);
        else                           dg = arma::diagvec(sym_inv(H)).t();
        if (pass == 0) V_cond.row(j) = dg; else V_eff.row(j) = dg;
      }
    }
  }
  return List::create(Named("V_cond") = V_cond, Named("V_eff") = V_eff);
}

// Data precision of each species after integrating out M, flattened:
//   A_j = X~^T diag(v_eff_j) X~,  column j = vec(A_j).
// [[Rcpp::export]]
arma::mat uq_species_A_cpp(const arma::mat& X_tilde, const arma::mat& BL,
                           const arma::mat& observed, const arma::mat& V_eta,
                           int block = 5000, int n_threads = 1) {
  const arma::uword p = BL.n_rows, d = BL.n_cols;
  const arma::uword k = static_cast<arma::uword>(std::sqrt((double) V_eta.n_cols));
  arma::mat A(d * d, p);
  for (arma::uword j0 = 0; j0 < p; j0 += block) {
    const arma::uword j1 = std::min<arma::uword>(j0 + block, p) - 1;
    const arma::mat BL_b = BL.rows(j0, j1);
    const arma::mat v    = cell_weights(X_tilde, BL_b, observed, j0);
    const arma::mat Q    = V_eta * outer_rows(BL_b.cols(d - k, d - 1)).t();
    const arma::mat v_eff = arma::clamp(v - (v % v) % Q, 0.0, arma::datum::inf);
    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (arma::uword jj = 0; jj < BL_b.n_rows; ++jj)
      A.col(j0 + jj) = arma::vectorise((X_tilde.each_col() % v_eff.col(jj)).t() * X_tilde);
  }
  return A;
}

// Trainable embeddings, block-Jacobi in species (the taxonomy coupling of the
// Xi prior is kept only through its diagonal q_diag_j = 2(tau + gamma [L_norm]_jj)):
//   W_j = (Sigma + A_j^{-1})^{-1},  P_j = q_diag I_H + Gamma' W_j Gamma,
//   M_j = W_j - W_j Gamma P_j^{-1} Gamma' W_j,  U_j = Gamma P_j^{-1} Gamma'.
// [[Rcpp::export]]
List uq_trainable_pass1_cpp(const arma::mat& A, const arma::vec& sig_bar,
                            const arma::mat& Gamma, const arma::vec& q_diag, int n_threads = 1) {
  const arma::uword d = Gamma.n_rows, H = Gamma.n_cols, p = A.n_cols;
  const arma::mat S_inv = arma::diagmat(1.0 / sig_bar);
  arma::mat W(d * d, p), M(d * d, p), U(d * d, p);
  #pragma omp parallel for num_threads(n_threads) schedule(static)
  for (arma::uword j = 0; j < p; ++j) {
    const arma::mat Aj = arma::reshape(A.col(j), d, d);
    const arma::mat Wj = S_inv - S_inv * sym_inv(Aj + S_inv) * S_inv;
    const arma::mat WG = Wj * Gamma;
    const arma::mat Pinv = sym_inv(q_diag(j) * arma::eye(H, H) + Gamma.t() * WG);
    W.col(j) = arma::vectorise(Wj);
    M.col(j) = arma::vectorise(Wj - WG * Pinv * WG.t());
    U.col(j) = arma::vectorise(Gamma * Pinv * Gamma.t());
  }
  return List::create(Named("W") = W, Named("M") = M, Named("U") = U);
}

// D_j = Var(Gamma xi_j) = (I - U W) K (I - W U) + U, made leave-one-out by
// D + D W (I - D W)^{-1} D, and the diagonal of (A_j + (Sigma + D_j)^{-1})^{-1}.
// [[Rcpp::export]]
List uq_trainable_pass2_cpp(const arma::mat& A, const arma::mat& W, const arma::mat& K,
                            const arma::mat& U, const arma::vec& sig_bar, int n_threads = 1) {
  const arma::uword d = sig_bar.n_elem, p = A.n_cols;
  const arma::mat I = arma::eye(d, d);
  arma::mat V(p, d), D(p, d);
  #pragma omp parallel for num_threads(n_threads) schedule(static)
  for (arma::uword j = 0; j < p; ++j) {
    const arma::mat Wj = arma::reshape(W.col(j), d, d), Uj = arma::reshape(U.col(j), d, d);
    arma::mat Dj = (I - Uj * Wj) * arma::reshape(K.col(j), d, d) * (I - Wj * Uj) + Uj;
    Dj += Dj * Wj * safe_solve(I - Dj * Wj, Dj);   // leave-one-out step
    Dj = 0.5 * (Dj + Dj.t());
    const arma::mat Hb = arma::reshape(A.col(j), d, d) + sym_inv(arma::diagmat(sig_bar) + Dj);
    V.row(j) = arma::diagvec(sym_inv(Hb)).t();
    D.row(j) = arma::diagvec(Dj).t();
  }
  return List::create(Named("V") = V, Named("D") = D);
}
