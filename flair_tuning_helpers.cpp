// tuning/flair_tuning_helpers.cpp
// C++ replacements for the July helpers removed from the production cpps, plus
// the blockwise sigma^2 update. Single pass over D = Xi - Phi, matrix-free
// signed-Laplacian operator (never forms the p x p matrix), OpenMP row loops.
#include <RcppArmadillo.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <dlfcn.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

// Cumulative-path codes are nested (each level-(l+1) group lies inside one
// level-l group), so group sums are built once at the finest level and
// aggregated upward through parent maps instead of rescanning all p rows per
// level: one serial p x h scan + tiny per-level aggregations + one OMP output
// pass, versus 2(L-1) serial p x h scans in the naive version.
static arma::mat tax_lap_prod(const arma::mat& X, const arma::imat& codes) {
  const int p = X.n_rows, h = X.n_cols, L = codes.n_cols;
  std::vector<arma::mat> S(L);
  std::vector<arma::vec> n(L);
  std::vector<arma::ivec> parent(L);  // parent[l](g at level l+1) = group at level l
  for (int ell = 0; ell < L; ++ell) {
    const int gmax = codes.col(ell).max();
    S[ell].zeros(gmax + 1, h);
    n[ell].zeros(gmax + 1);
    if (ell + 1 < L) parent[ell].zeros(codes.col(ell + 1).max() + 1);
  }
  for (int i = 0; i < p; ++i) {
    S[L - 1].row(codes(i, L - 1)) += X.row(i);
    n[L - 1](codes(i, L - 1)) += 1.0;
    for (int ell = 0; ell + 1 < L; ++ell) parent[ell](codes(i, ell + 1)) = codes(i, ell);
  }
  for (int ell = L - 2; ell >= 0; --ell)
    for (arma::uword g = 0; g < parent[ell].n_elem; ++g) {
      S[ell].row(parent[ell](g)) += S[ell + 1].row(g);
      n[ell](parent[ell](g)) += n[ell + 1](g);
    }

  arma::mat LX(p, h);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < p; ++i) {
    arma::rowvec acc(h, arma::fill::zeros);
    double d = 0.0;
    for (int ell = 0; ell < L; ++ell) {
      const double coef = static_cast<double>(ell + 1 - (L - 1));
      if (coef == 0.0) continue;
      arma::rowvec es; double en;
      if (ell + 1 < L) {
        es = S[ell].row(codes(i, ell)) - S[ell + 1].row(codes(i, ell + 1));
        en = n[ell](codes(i, ell)) - n[ell + 1](codes(i, ell + 1));
      } else {
        es = S[ell].row(codes(i, ell)) - X.row(i);
        en = n[ell](codes(i, ell)) - 1.0;
      }
      acc += coef * es;
      d += std::abs(coef) * en;
    }
    LX.row(i) = d * X.row(i) - acc;
  }
  return LX;
}

static arma::vec tax_lap_degree(const arma::imat& codes) {
  const int p = codes.n_rows, L = codes.n_cols;
  arma::vec deg(p, arma::fill::zeros);
  for (int ell = 0; ell < L; ++ell) {
    const double coef = static_cast<double>(ell + 1 - (L - 1));
    if (coef == 0.0) continue;
    arma::vec n_cur(codes.col(ell).max() + 1, arma::fill::zeros);
    for (int i = 0; i < p; ++i) n_cur(codes(i, ell)) += 1.0;
    arma::vec n_next;
    if (ell + 1 < L) {
      n_next.zeros(codes.col(ell + 1).max() + 1);
      for (int i = 0; i < p; ++i) n_next(codes(i, ell + 1)) += 1.0;
    }
    for (int i = 0; i < p; ++i)
      deg(i) += std::abs(coef) * ((ell + 1 < L)
        ? n_cur(codes(i, ell)) - n_next(codes(i, ell + 1))
        : n_cur(codes(i, ell)) - 1.0);
  }
  return deg;
}

// L_norm X = D^{-1/2} L (D^{-1/2} X) with precomputed 1/sqrt(deg).
static arma::mat tax_lap_norm_prod_pre(const arma::mat& X, const arma::imat& codes,
                                       const arma::vec& isd) {
  arma::mat Xs = X;
  Xs.each_col() %= isd;
  arma::mat R = tax_lap_prod(Xs, codes);
  R.each_col() %= isd;
  return R;
}

// L_norm X = D^{-1/2} L (D^{-1/2} X); degrees clamped at 1e-8.
static arma::mat tax_lap_norm_prod(const arma::mat& X, const arma::imat& codes) {
  const arma::uword p = X.n_rows;
  arma::vec deg = tax_lap_degree(codes);
  arma::vec isd(p);
  for (arma::uword i = 0; i < p; ++i) isd(i) = 1.0 / std::sqrt(std::max(deg(i), 1e-8));
  return tax_lap_norm_prod_pre(X, codes, isd);
}

// Set the OpenBLAS thread count from R (the production update_Gamma_cpp has no
// n_threads arg; its K x K GEMMs run on whatever BLAS threading is active).
// [[Rcpp::export]]
void set_blas_threads(int n) {
  typedef void (*set_nt_t)(int);
  static set_nt_t blas_nt = reinterpret_cast<set_nt_t>(
      dlsym(RTLD_DEFAULT, "openblas_set_num_threads"));
  if (!blas_nt)
    blas_nt = reinterpret_cast<set_nt_t>(
        dlsym(RTLD_DEFAULT, "openblas_set_num_threads_"));
  if (blas_nt) blas_nt(n);
}

// Same MAP as production update_Gamma_cpp, restructured: the (q+k) per-row
// GEMVs over Xi (each a full h x p pass) become one GEMM, and the h x h
// solves run under OMP with single-threaded BLAS. Same math; differs from
// production only by GEMM-vs-GEMV kernel rounding (~1e-14 rel).
// [[Rcpp::export]]
arma::mat update_Gamma_fast_cpp(const arma::mat& Beta_Lambda,
                                const arma::mat& Xi_mat,
                                const arma::vec& sigma_sq_vec,
                                const arma::vec& kappa_sq_vec,
                                int n_threads = 1) {
  const int qk = Beta_Lambda.n_cols;
  const int h  = Xi_mat.n_rows;
  set_blas_threads(n_threads);
  arma::mat XiXit = Xi_mat * Xi_mat.t();
  arma::mat BLs = Beta_Lambda;
  BLs.each_row() /= sigma_sq_vec.t();
  arma::mat RHS = Xi_mat * BLs;            // h x (q+k)
  set_blas_threads(1);
  arma::mat Gamma(qk, h);
  const arma::mat I_h = arma::eye(h, h);
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(static)
#endif
  for (int r = 0; r < qk; ++r) {
    arma::mat prec_r = XiXit / sigma_sq_vec(r) + I_h / kappa_sq_vec(r);
    arma::vec gr = arma::solve(prec_r, RHS.col(r), arma::solve_opts::fast);
    Gamma.row(r) = gr.t();
  }
  return Gamma;
}

// Fused sufficient statistics for the Xi hyperparameter MAPs:
//   S_vec[h] = ||D_h||^2 / p,  S_gamma = tr(D' L_norm D) / (p h),
// plus the raw sums so the caller can reuse them for the Xi log-prior
// (-sum_h tau_h ||D_h||^2 - gamma tr) without a second Laplacian product.
// [[Rcpp::export]]
List compute_xi_ss_cpp(const arma::mat& Xi_t, const arma::mat& Phi_t,
                       const arma::imat& codes, double xi_gamma) {
  const double p = Xi_t.n_rows, h = Xi_t.n_cols;
  arma::mat D = Xi_t - Phi_t;
  arma::vec sumsq = arma::sum(D % D, 0).t();
  arma::mat LD = tax_lap_norm_prod(D, codes);
  const double tr_DLD = arma::accu(D % LD);
  return List::create(_["S_vec"] = sumsq / p,
                      _["S_gamma"] = tr_DLD / (p * h),
                      _["sumsq"] = sumsq,
                      _["tr_DLD"] = tr_DLD);
}

// InvGamma precision MAP (JSDMs4): theta(S;a,b) = {-(a+1)+sqrt((a+1)^2+4bS)}/(2S),
// b/(a+1) at S=0; clamped to [lb,ub]. Returns a one-element list (callers use [[1]]).
// [[Rcpp::export]]
List ig_map_vec_cpp(const arma::vec& S, double a, double b, double lb, double ub) {
  arma::vec r(S.n_elem);
  const double a1 = a + 1.0;
  for (arma::uword i = 0; i < S.n_elem; ++i)
    r(i) = (S(i) <= 0.0) ? b / a1
                         : (-a1 + std::sqrt(a1 * a1 + 4.0 * b * S(i))) / (2.0 * S(i));
  return List::create(arma::clamp(r, lb, ub));
}

// Half-Cauchy precision MAP (JSDMs4): 2/(S + sqrt(S^2 + 8 S s^2)), ub at S=0.
// [[Rcpp::export]]
List hc_map_tau_cpp(const arma::vec& S, double s, double lb, double ub) {
  arma::vec r(S.n_elem);
  const double s2 = s * s;
  for (arma::uword i = 0; i < S.n_elem; ++i)
    r(i) = (S(i) <= 0.0) ? ub
                         : 2.0 / (S(i) + std::sqrt(S(i) * S(i) + 8.0 * S(i) * s2));
  return List::create(arma::clamp(r, lb, ub));
}

// Eigenbasis-decoupled PCG for the Xi update. Same linear system, same
// preconditioner and same iteration as the production update_xi_taxonomy_cg_cpp,
// but run in the eigenbasis of C = Gamma'Sigma^{-1}Gamma + 2 diag(tau): with
// Z = Xi * E (E orthogonal), V*C becomes a column scaling and the
// preconditioner becomes diagonal, so the three p x H x H GEMMs per CG
// iteration disappear (one rotation in, one out). Orthogonal rotation
// preserves every CG inner product, so iterates match the production solver
// to floating-point rounding.
// [[Rcpp::export]]
List update_xi_taxonomy_cg_fast_cpp(const arma::mat& Beta_Lambda,
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
  typedef void (*set_nt_t)(int);
  static set_nt_t blas_nt = reinterpret_cast<set_nt_t>(
      dlsym(RTLD_DEFAULT, "openblas_set_num_threads"));
  if (!blas_nt)
    blas_nt = reinterpret_cast<set_nt_t>(
        dlsym(RTLD_DEFAULT, "openblas_set_num_threads_"));
#ifdef _OPENMP
  if (n_threads > 0) omp_set_num_threads(n_threads);
#endif
  if (blas_nt) blas_nt(n_threads);

  const int h = Xi_init.n_cols;
  const arma::uword p = Xi_init.n_rows;
  arma::mat Gamma_sigma = Gamma;
  Gamma_sigma.each_col() /= sigma_sq_vec;
  arma::mat coef_precision = Gamma.t() * Gamma_sigma + arma::diagmat(2.0 * tau_vec);
  arma::mat B_sigma = Beta_Lambda;
  B_sigma.each_row() /= sigma_sq_vec.t();

  arma::vec deg = tax_lap_degree(codes);
  arma::vec isd(p), ndiag(p);
  for (arma::uword i = 0; i < p; ++i) {
    isd(i) = 1.0 / std::sqrt(std::max(deg(i), 1e-8));
    ndiag(i) = (deg(i) > 1e-8) ? 1.0 : 0.0;
  }

  arma::mat rhs = B_sigma * Gamma + 2.0 * Phi * arma::diagmat(tau_vec);
  if (xi_gamma != 0.0) {
    if (blas_nt) blas_nt(1);
    rhs += 2.0 * xi_gamma * tax_lap_norm_prod_pre(Phi, codes, isd);
    if (blas_nt) blas_nt(n_threads);
  }

  arma::vec eigval;
  arma::mat eigvec;
  arma::eig_sym(eigval, eigvec, coef_precision);
  for (arma::uword j = 0; j < eigval.n_elem; ++j)
    if (!std::isfinite(eigval(j)) || eigval(j) < 1e-12) eigval(j) = 1e-12;

  // Rotate once: everything below lives in Z = V * eigvec coordinates.
  arma::mat rhs_z = rhs * eigvec;
  arma::mat Z = Xi_init * eigvec;

  auto matvec_op = [&](const arma::mat& V) -> arma::mat {
    arma::mat out = V;
    out.each_row() %= eigval.t();
    if (xi_gamma != 0.0) {
      if (blas_nt) blas_nt(1);
      out += 2.0 * xi_gamma * tax_lap_norm_prod_pre(V, codes, isd);
      if (blas_nt) blas_nt(n_threads);
    }
    return out;
  };

  auto precond_op = [&](const arma::mat& R) -> arma::mat {
    arma::mat PR = R;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (arma::uword i = 0; i < p; ++i) {
      for (int j = 0; j < h; ++j) {
        double d = eigval(j) + 2.0 * xi_gamma * ndiag(i);
        if (!std::isfinite(d) || d < 1e-12) d = 1e-12;
        PR(i, j) /= d;
      }
    }
    return PR;
  };

  arma::mat residual = rhs_z - matvec_op(Z);
  const double norm_rhs = std::max(std::sqrt(arma::accu(rhs_z % rhs_z)), 1e-12);
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
    Z += alpha * direction;
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

  arma::mat Xi = Z * eigvec.t();
  // Same objective as production xi_objective_impl (multi-threaded BLAS here).
  arma::mat Mu = Xi * Gamma.t();
  arma::mat Delta = Beta_Lambda - Mu;
  Delta.each_row() /= sigma_sq_vec.t();
  double obj = -0.5 * arma::accu((Beta_Lambda - Mu) % Delta);
  arma::mat D = Xi - Phi;
  arma::rowvec col_sq = arma::sum(D % D, 0);
  obj -= arma::dot(tau_vec, col_sq.t());
  if (xi_gamma != 0.0) {
    if (blas_nt) blas_nt(1);
    arma::mat LD = tax_lap_norm_prod_pre(D, codes, isd);
    obj -= xi_gamma * arma::accu(D % LD);
  }
  if (blas_nt) blas_nt(1);

  return List::create(
    _["Xi"] = Xi,
    _["objective"] = obj,
    _["iter"] = iter_done,
    _["rel_resid"] = rel,
    _["converged"] = converged
  );
}

// Blockwise sigma^2 MAP: b_r = alpha_r * p (soft floor 2*alpha_r), lb = 0.01.
// [[Rcpp::export]]
arma::vec update_sigma_sq_blockwise_cpp(const arma::mat& Beta_Lambda,
                                        const arma::mat& Gamma,
                                        const arma::mat& Xi_mat,
                                        double a, const arma::vec& alpha,
                                        double ub) {
  arma::mat Mu = (Gamma * Xi_mat).t();
  Mu -= Beta_Lambda;
  arma::vec SS = arma::sum(Mu % Mu, 0).t();
  const double p = Beta_Lambda.n_rows;
  return arma::clamp((alpha * p + SS / 2.0) / (a + p / 2.0 + 1.0), 0.01, ub);
}

// Set the OpenBLAS thread count for R-level GEMMs (no-op if the symbol is
// absent). The heavy C++ entry points set their own count on entry.
// [[Rcpp::export]]
void set_blas_threads_cpp(int n) {
  typedef void (*set_nt_t)(int);
  set_nt_t f = reinterpret_cast<set_nt_t>(
      dlsym(RTLD_DEFAULT, "openblas_set_num_threads"));
  if (!f) f = reinterpret_cast<set_nt_t>(
      dlsym(RTLD_DEFAULT, "openblas_set_num_threads_"));
  if (f && n > 0) f(n);
}
