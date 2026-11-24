#include <RcppArmadillo.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <dlfcn.h>

// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

arma::mat taxonomy_signed_laplacian_prod_impl(const arma::mat& X,
                                              const arma::imat& codes) {
  const int p = X.n_rows;
  const int h = X.n_cols;
  const int L = codes.n_cols;
  arma::mat WX(p, h, arma::fill::zeros);
  arma::vec deg(p, arma::fill::zeros);

  for (int ell = 0; ell < L; ++ell) {
    const double coef = static_cast<double>(ell + 1 - (L - 1));
    if (coef == 0.0) continue;

    const int gmax = codes.col(ell).max();
    arma::mat sum_cur(gmax + 1, h, arma::fill::zeros);
    arma::vec n_cur(gmax + 1, arma::fill::zeros);
    for (int i = 0; i < p; ++i) {
      const int g = codes(i, ell);
      sum_cur.row(g) += X.row(i);
      n_cur(g) += 1.0;
    }

    arma::mat sum_next;
    arma::vec n_next;
    if (ell + 1 < L) {
      const int gmax_next = codes.col(ell + 1).max();
      sum_next.zeros(gmax_next + 1, h);
      n_next.zeros(gmax_next + 1);
      for (int i = 0; i < p; ++i) {
        const int g = codes(i, ell + 1);
        sum_next.row(g) += X.row(i);
        n_next(g) += 1.0;
      }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < p; ++i) {
      arma::rowvec exact_sum;
      double exact_n;
      if (ell + 1 < L) {
        exact_sum = sum_cur.row(codes(i, ell)) - sum_next.row(codes(i, ell + 1));
        exact_n = n_cur(codes(i, ell)) - n_next(codes(i, ell + 1));
      } else {
        exact_sum = sum_cur.row(codes(i, ell)) - X.row(i);
        exact_n = n_cur(codes(i, ell)) - 1.0;
      }
      WX.row(i) += coef * exact_sum;
      deg(i) += std::abs(coef) * exact_n;
    }
  }

  arma::mat LX = X;
  LX.each_col() %= deg;
  LX -= WX;
  return LX;
}

arma::vec taxonomy_signed_laplacian_degree_impl(const arma::imat& codes) {
  const int p = codes.n_rows;
  const int L = codes.n_cols;
  arma::vec deg(p, arma::fill::zeros);
  for (int ell = 0; ell < L; ++ell) {
    const double coef = static_cast<double>(ell + 1 - (L - 1));
    if (coef == 0.0) continue;
    const int gmax = codes.col(ell).max();
    arma::vec n_cur(gmax + 1, arma::fill::zeros);
    for (int i = 0; i < p; ++i) n_cur(codes(i, ell)) += 1.0;
    arma::vec n_next;
    if (ell + 1 < L) {
      const int gmax_next = codes.col(ell + 1).max();
      n_next.zeros(gmax_next + 1);
      for (int i = 0; i < p; ++i) n_next(codes(i, ell + 1)) += 1.0;
    }
    for (int i = 0; i < p; ++i) {
      const double exact_n = (ell + 1 < L)
        ? n_cur(codes(i, ell)) - n_next(codes(i, ell + 1))
        : n_cur(codes(i, ell)) - 1.0;
      deg(i) += std::abs(coef) * exact_n;
    }
  }
  return deg;
}

// codes are cumulative taxonomy path factors, one column per taxonomy level.
// [[Rcpp::export]]
arma::mat taxonomy_signed_laplacian_prod_cpp(const arma::mat& X,
                                             const arma::imat& codes) {
  return taxonomy_signed_laplacian_prod_impl(X, codes);
}

// [[Rcpp::export]]
arma::vec taxonomy_signed_laplacian_degree_cpp(const arma::imat& codes) {
  return taxonomy_signed_laplacian_degree_impl(codes);
}

// Normalised signed Laplacian product: L_norm X = D^{-1/2} L D^{-1/2} X.
//
// Original (unscaled) L:
//   L = D - W,  where W_{ij} = sum over taxonomy levels of coef_ell * I[same group at ell]
//   and d_i = sum_ell |coef_ell| * (number of co-members at level ell not in same finer group).
//   Eigenvalues are O(max group size) — can reach 10^3..10^4 for large genera.
//
// Normalised L_norm = D^{-1/2} L D^{-1/2}:
//   (L_norm)_{ij} = L_{ij} / sqrt(d_i * d_j).
//   Diagonal: (L_norm)_{ii} = 1 for every connected species, 0 for isolated.
//   Eigenvalues bounded in [0, 2] regardless of genus sizes.
//
// Application as a matrix-free operator (precomputed inv_sqrt_d overload):
//   L_norm X = D^{-1/2} ( L ( D^{-1/2} X ) )
//   i.e. pre-scale rows of X by 1/sqrt(d_i), apply L, post-scale rows by 1/sqrt(d_i).
//
// Numerical stability: d_i is clamped to eps > 0 so isolated species (d_i = 0)
// contribute nothing rather than producing Inf/NaN.
static arma::mat taxonomy_signed_laplacian_norm_prod_impl(
    const arma::mat& X,
    const arma::imat& codes,
    const arma::vec& inv_sqrt_d) {         // precomputed 1/sqrt(max(d_i, eps))
  arma::mat Xs = X;
  Xs.each_col() %= inv_sqrt_d;            // D^{-1/2} X
  arma::mat result = taxonomy_signed_laplacian_prod_impl(Xs, codes);  // L (D^{-1/2} X)
  result.each_col() %= inv_sqrt_d;        // D^{-1/2} L D^{-1/2} X
  return result;
}

// Convenience overload: computes inv_sqrt_d internally (used in one-shot calls).
static arma::mat taxonomy_signed_laplacian_norm_prod_impl(
    const arma::mat& X, const arma::imat& codes) {
  const arma::uword p = X.n_rows;
  const double eps = 1e-8;
  arma::vec d = taxonomy_signed_laplacian_degree_impl(codes);
  arma::vec inv_sqrt_d(p);
  for (arma::uword i = 0; i < p; ++i)
    inv_sqrt_d(i) = 1.0 / std::sqrt(std::max(d(i), eps));
  return taxonomy_signed_laplacian_norm_prod_impl(X, codes, inv_sqrt_d);
}

// [[Rcpp::export]]
arma::mat taxonomy_signed_laplacian_norm_prod_cpp(const arma::mat& X,
                                                   const arma::imat& codes) {
  return taxonomy_signed_laplacian_norm_prod_impl(X, codes);
}

double xi_objective_impl(const arma::mat& X,
                         const arma::mat& Phi,
                         const arma::imat& codes,
                         const arma::mat& Beta_Lambda,
                         const arma::mat& Gamma,
                         const arma::vec& sigma_sq_vec,
                         const arma::vec& tau_vec,
                         double xi_gamma) {
  arma::mat Mu = X * Gamma.t();
  arma::mat Delta = Beta_Lambda - Mu;
  Delta.each_row() /= sigma_sq_vec.t();
  const double beta_term = -0.5 * arma::accu((Beta_Lambda - Mu) % Delta);
  const arma::mat D = X - Phi;
  // Additive per-component ARD: tau_h replaces scalar lambda_xi per column.
  // -sum_h tau_h ||delta_h||^2  -  gamma_xi tr(Delta' L Delta)
  arma::rowvec col_sq = arma::sum(D % D, 0);
  double prior = -arma::dot(tau_vec, col_sq.t());
  if (xi_gamma != 0.0) {
    arma::mat LD = taxonomy_signed_laplacian_norm_prod_impl(D, codes);
    prior -= xi_gamma * arma::accu(D % LD);
  }
  return beta_term + prior;
}

// [[Rcpp::export]]
double xi_prior_term_taxonomy_cpp(const arma::mat& Xi,
                                  const arma::mat& Phi,
                                  const arma::imat& codes,
                                  const arma::vec& tau_vec,
                                  double xi_gamma) {
  const arma::mat D = Xi - Phi;
  arma::rowvec col_sq = arma::sum(D % D, 0);
  double prior = -arma::dot(tau_vec, col_sq.t());
  if (xi_gamma != 0.0) {
    arma::mat LD = taxonomy_signed_laplacian_norm_prod_impl(D, codes);
    prior -= xi_gamma * arma::accu(D % LD);
  }
  return prior;
}

// Matrix-free CG solve for the additive per-component ARD model:
//   Xi (Gamma' Sigma^{-1} Gamma + 2 diag(tau)) + 2 xi_gamma L_norm Xi
//     = B Sigma^{-1} Gamma + 2 (diag(tau) + xi_gamma L_norm) Phi
// A(Xi) = Xi*C + 2*xi_gamma*L_norm*Xi,  C = Gamma'Sigma^{-1}Gamma + 2*diag(tau).
// The signed taxonomy-path Laplacian is applied as an operator; the dense
// species-by-species L is never formed.
// [[Rcpp::export]]
List update_xi_taxonomy_cg_cpp(const arma::mat& Beta_Lambda,
                               const arma::mat& Gamma,
                               const arma::vec& sigma_sq_vec,
                               const arma::mat& Xi_init,
                               const arma::mat& Phi,
                               const arma::imat& codes,
                               const arma::vec& tau_vec,
                               double xi_gamma = 1.0,
                               int maxit = 50,
                               double tol = 1e-4,
                               int n_threads = 1) {
  // Resolve openblas_set_num_threads once; NULL if not using OpenBLAS.
  typedef void (*set_nt_t)(int);
  static set_nt_t blas_nt = reinterpret_cast<set_nt_t>(
      dlsym(RTLD_DEFAULT, "openblas_set_num_threads"));
  if (!blas_nt)
    blas_nt = reinterpret_cast<set_nt_t>(
        dlsym(RTLD_DEFAULT, "openblas_set_num_threads_"));
#ifdef _OPENMP
  if (n_threads > 0) omp_set_num_threads(n_threads);
#endif
  // Enable BLAS multi-threading for the large p×H² matmuls that follow.
  // We call blas_nt(1) before each taxonomy Laplacian OMP section to prevent
  // thread over-subscription (OMP × BLAS nesting).
  if (blas_nt) blas_nt(n_threads);
  const int h = Xi_init.n_cols;
  arma::mat Gamma_sigma = Gamma;
  Gamma_sigma.each_col() /= sigma_sq_vec;
  // C = Gamma'Sigma^{-1}Gamma + 2*diag(tau): data precision plus per-column ARD.
  arma::mat coef_precision = Gamma.t() * Gamma_sigma + arma::diagmat(2.0 * tau_vec);
  arma::mat B_sigma = Beta_Lambda;
  B_sigma.each_row() /= sigma_sq_vec.t();
  // Precompute D^{-1/2} for the normalised Laplacian; needed for both RHS and CG matvec.
  const arma::uword p_xi = Xi_init.n_rows;
  const double norm_eps = 1e-8;
  arma::vec laplacian_diag_raw = taxonomy_signed_laplacian_degree_impl(codes);
  arma::vec inv_sqrt_d(p_xi), norm_laplacian_diag(p_xi);
  for (arma::uword i = 0; i < p_xi; ++i) {
    inv_sqrt_d(i) = 1.0 / std::sqrt(std::max(laplacian_diag_raw(i), norm_eps));
    norm_laplacian_diag(i) = (laplacian_diag_raw(i) > norm_eps) ? 1.0 : 0.0;
  }
  // rhs = B Sigma^{-1} Gamma + 2 Phi diag(tau) + 2 xi_gamma L_norm Phi
  arma::mat rhs = B_sigma * Gamma + 2.0 * Phi * arma::diagmat(tau_vec);
  if (xi_gamma != 0.0) {
    if (blas_nt) blas_nt(1);
    rhs += 2.0 * xi_gamma *
      taxonomy_signed_laplacian_norm_prod_impl(Phi, codes, inv_sqrt_d);
    if (blas_nt) blas_nt(n_threads);
  }

  arma::vec eigval;
  arma::mat eigvec;
  arma::eig_sym(eigval, eigvec, coef_precision);
  for (arma::uword j = 0; j < eigval.n_elem; ++j)
    if (!std::isfinite(eigval(j)) || eigval(j) < 1e-12) eigval(j) = 1e-12;

  // A(V) = V * coef_precision + 2 * xi_gamma * L_norm * V
  auto matvec_op = [&](const arma::mat& V) -> arma::mat {
    arma::mat out = V * coef_precision;
    if (xi_gamma != 0.0) {
      if (blas_nt) blas_nt(1);
      out += 2.0 * xi_gamma *
        taxonomy_signed_laplacian_norm_prod_impl(V, codes, inv_sqrt_d);
      if (blas_nt) blas_nt(n_threads);
    }
    return out;
  };

  // Preconditioner: diagonal approx d(i,j) = eigval[j] + 2*xi_gamma*L_ii
  auto precond_op = [&](const arma::mat& R) -> arma::mat {
    arma::mat PR = R * eigvec;
    for (arma::uword i = 0; i < PR.n_rows; ++i) {
      for (int j = 0; j < h; ++j) {
        double d = eigval(j) + 2.0 * xi_gamma * norm_laplacian_diag(i);
        if (!std::isfinite(d) || d < 1e-12) d = 1e-12;
        PR(i, j) /= d;
      }
    }
    return PR * eigvec.t();
  };

  arma::mat Xi = Xi_init;
  arma::mat residual = rhs - matvec_op(Xi);
  const double norm_rhs = std::max(std::sqrt(arma::accu(rhs % rhs)), 1e-12);
  double rel = std::sqrt(arma::accu(residual % residual)) / norm_rhs;

  arma::mat precond_residual = precond_op(residual);
  arma::mat direction = precond_residual;
  double rz = arma::accu(residual % precond_residual);
  int iter_done = 0;
  bool converged = rel <= tol;

  for (int it = 0; it < maxit && !converged; ++it) {
    arma::mat matvec_direction = matvec_op(direction);
    const double denom = arma::accu(direction % matvec_direction);
    if (!std::isfinite(denom) || denom <= 0.0) break;
    const double alpha = rz / denom;
    if (!std::isfinite(alpha)) break;
    Xi += alpha * direction;
    residual -= alpha * matvec_direction;
    rel = std::sqrt(arma::accu(residual % residual)) / norm_rhs;
    iter_done = it + 1;
    if (!std::isfinite(rel) || rel <= tol) {
      converged = std::isfinite(rel) && rel <= tol;
      break;
    }
    precond_residual = precond_op(residual);
    const double rz_new = arma::accu(residual % precond_residual);
    if (!std::isfinite(rz_new) || rz_new <= 0.0) break;
    const double beta = rz_new / rz;
    if (!std::isfinite(beta)) break;
    direction = precond_residual + beta * direction;
    rz = rz_new;
  }
  // Restore BLAS to single-thread after CG so caller OMP regions stay clean.
  if (blas_nt) blas_nt(1);

  const double obj = xi_objective_impl(
    Xi, Phi, codes, Beta_Lambda, Gamma, sigma_sq_vec, tau_vec, xi_gamma
  );
  return List::create(
    _["Xi"] = Xi,
    _["objective"] = obj,
    _["iter"] = iter_done,
    _["rel_resid"] = rel,
    _["converged"] = converged
  );
}
