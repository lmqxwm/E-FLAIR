#include <RcppArmadillo.h>
#include <Rcpp.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <dlfcn.h>   // dlsym — for runtime OpenBLAS thread control
#include <chrono>



// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
bool flair_plus_has_openmp_cpp() {
#ifdef _OPENMP
  return true;
#else
  return false;
#endif
}


inline double softplus(double x) {
  if (x > 35.0) return x;
  if (x < -35.0) return std::exp(x);
  return std::log1p(std::exp(x));
}

double F_logistic(double x){
  if (x >= 0) {
    double z = std::exp(-x);
    return 1.0 / (1.0 + z);
  }
  double z = std::exp(x);
  return z / (1.0 + z);
}

// Fused sigmoid + Bernoulli log-likelihood: computes p = sigmoid(x) and
// ll = y*x - log(1+exp(x)) in a single pass with one exp() call.
// Branches on sign of x to avoid overflow: for x>=0 we fold out exp(-x),
// for x<0 we fold out exp(x).  Both branches reuse the same z and log_denom
// for p and ll, halving exp() cost vs calling F_logistic + softplus separately.
// Use this wherever both p (for residual/weight) and ll (for objective) are needed.
// For ll-only paths use:  y*x - softplus(x)  (softplus has its own >35 early-exit).
inline void bernoulli_logit_prob_ll_stable(double y, double x,
                                           double& p, double& ll) {
  if (x >= 0.0) {
    const double z = std::exp(-x);          // z = exp(-x), avoids overflow
    const double log_denom = std::log1p(z); // log(1 + exp(-x)) = log(1+exp(x)) - x
    p  = 1.0 / (1.0 + z);                  // sigmoid(x) = 1/(1+exp(-x))
    ll = (y != 0.0) ? -log_denom : -x - log_denom; // y=1: x-log(1+exp(x))=-log_denom; y=0: -log(1+exp(x))
  } else {
    const double z = std::exp(x);           // z = exp(x), avoids underflow
    const double log_denom = std::log1p(z); // log(1 + exp(x))
    p  = z / (1.0 + z);                    // sigmoid(x) = exp(x)/(1+exp(x))
    ll = (y != 0.0) ? x - log_denom : -log_denom; // y=1: x-log(1+exp(x)); y=0: -log(1+exp(x))
  }
}

inline double trust_region_scale_from_norm(double norm, double delta) {
  if (!std::isfinite(norm) || norm <= 0.0) return 1.0;
  return std::min(1.0, std::max(1e-12, delta) / norm);
}

inline double predicted_reduction_m_block(
    const arma::mat& Grad_b,
    const arma::mat& H,
    const arma::mat& Step_b) {

  arma::mat HS = Step_b * H;
  return arma::accu(Grad_b % Step_b) - 0.5 * arma::accu(HS % Step_b);
}

double log_post_latent_factor_conditional(
    const arma::vec& y,
    const arma::vec& observed,
    const arma::vec& x,
    const arma::mat& Lambda,
    const arma::mat& Beta,
    const arma::vec& M_i,
    double prior_var) {

  int p = y.n_elem;
  double obj = -0.5 * arma::dot(M_i, M_i) / prior_var;
  for (int j = 0; j < p; ++j) {
    if (observed(j) != 1) continue;
    double eta = arma::dot(Beta.row(j), x) + arma::dot(Lambda.row(j), M_i);
    obj += y(j) * eta - softplus(eta);
  }
  return obj;
}

double log_post_latent_factor_conditional_fullobs(
    const arma::vec& y,
    const arma::vec& x,
    const arma::mat& Lambda,
    const arma::mat& Beta,
    const arma::vec& M_i,
    double prior_var) {

  int p = y.n_elem;
  double obj = -0.5 * arma::dot(M_i, M_i) / prior_var;
  for (int j = 0; j < p; ++j) {
    double eta = arma::dot(Beta.row(j), x) + arma::dot(Lambda.row(j), M_i);
    obj += y(j) * eta - softplus(eta);
  }
  return obj;
}

double log_post_beta_lambda_conditional(
    const arma::vec& y,
    const arma::vec& observed,
    const arma::mat& X_tilde,
    const arma::vec& bl_j,
    const arma::vec& mu_j,
    const arma::vec& sig_vec) {   // (q+k): per-dimension variance

  arma::vec eta  = X_tilde * bl_j;
  arma::vec diff = bl_j - mu_j;
  double obj = -0.5 * arma::dot(diff, diff / sig_vec);
  for (arma::uword i = 0; i < eta.n_elem; ++i) {
    if (observed(i) != 1) continue;
    obj += y(i) * eta(i) - softplus(eta(i));
  }
  return obj;
}


arma::mat grad_regression_coeffs_logit(
    arma::vec y, arma::vec observed, arma::mat M, arma::vec lambda, double prior_var_beta,
    double prior_var_lambda, int q, int k){
  // returns gradient and hessian of logistic regression log-lik + normal prior
  // p * (p+1) mat where first col is grad and rest is hessian
  int n = y.size();
  int p = M.n_cols;
  
  arma::vec scores(n, arma::fill::zeros);
  arma::vec grad_log_pi(p, arma::fill::zeros);
  arma::mat hessian(p, p, arma::fill::zeros);
  
  for(int i=0; i<n; ++i){
    scores(i) = 0;
    if(observed(i)==1){
      double eta_lambda_i = dot(M.row(i), lambda);
      double F_i = F_logistic(eta_lambda_i);
      scores(i) = y(i) - F_i;
      hessian = hessian + F_i* (1 - F_i)*M(i, arma::span(0, p-1)).t()*M(i, arma::span(0,p-1));
    }
  }
  for(int j=0; j<p; ++j){
    grad_log_pi(j) = sum(scores.t()*M(arma::span(0,n-1), j));
  }
  arma::vec tau_1 = 1/prior_var_beta * arma::ones(q);
  arma::vec tau_2 = 1/prior_var_lambda * arma::ones(k);
  arma::vec prior_var_inv = join_cols(tau_1, tau_2);
  
  grad_log_pi(arma::span(0,q-1)) = grad_log_pi(arma::span(0,q-1)) - lambda(arma::span(0,q-1)) / 
    prior_var_beta;
  grad_log_pi(arma::span(q,k+q-1)) = grad_log_pi(arma::span(q,k+q-1)) - lambda(arma::span(q,k+q-1)) / 
    prior_var_lambda;
  
  arma::mat D = arma::diagmat(join_cols(tau_1, tau_2)); 
  hessian = hessian + D; // negative hessian
  arma::mat result(p, p+1);
  result(arma::span(0, p-1), 0) = grad_log_pi;
  result(arma::span(0, p -1), arma::span(1, p)) = -hessian;
  return result;
}


arma::mat optimize_regression_coeffs_logit(
    arma::vec y, arma::vec observed, arma::mat X, arma::mat M, arma::vec beta_init, arma::vec lambda_init, 
    double prior_var_beta, double prior_var_lambda, int max_it, double epsilon, double C_lambda = 10,
    double C_mu = 10, double C_beta = 10, double step_size = 1){
  // find MAP for regression coeffs and loadings given factors via projected newton descent 
  
  int n = y.size();
  int k = M.n_cols;
  int q = X.n_cols;
  arma::mat M_new = join_rows(X, M);
  arma::vec regression_vector_init = join_cols(beta_init, lambda_init);
  arma::vec regression_vector = regression_vector_init;
  arma::vec prior_var_vec = join_cols(
    prior_var_beta * arma::ones(q),
    prior_var_lambda * arma::ones(k)
  );
  arma::vec prior_mean(k + q, arma::fill::zeros);
  arma::mat H(k+q, k+q, arma::fill::zeros);
  for(int i = 1; i<=max_it; ++i){
    arma::mat res = grad_regression_coeffs_logit(
      y, observed, M_new, regression_vector, prior_var_beta, prior_var_lambda, q, k
    );
    
    arma::vec u = res(arma::span(0, k+q-1), 0);
    H = res(arma::span(0, k+q-1), arma::span(1, k+q));
    arma::vec update;
    if (!arma::solve(update, H, -u)) break;
    double norm_update = norm(update, 2);
    
    if(norm_update<epsilon){
      break;
    }
    
    const double obj_old = log_post_beta_lambda_conditional(
      y, observed, M_new, regression_vector, prior_mean, prior_var_vec);
    double step = std::min(1.0, std::max(1e-8, step_size));
    bool accepted = false;
    for(int ls = 0; ls < 25; ++ls){
      arma::vec candidate = regression_vector + step * update;
      if(candidate(0)>C_mu){
        candidate(0) = C_mu;
      }
      else if(candidate(0)<(-1)*C_mu){
        candidate(0) = (-1)*C_mu;
      }
      for(int j=1; j<(q); ++j){
        if(candidate(j)>C_beta){
          candidate(j) = C_beta;
        }
        else if(candidate(j)<(-C_beta)){
          candidate(j) = (-1)*C_beta;
        }
      }
      for(int j=q; j<(k+q); ++j){
        if(candidate(j)>C_lambda){
          candidate(j) = C_lambda;
        }
        else if(candidate(j)<(-C_lambda)){
          candidate(j) = (-1)*C_lambda;
        }
      }
      const double obj_new = log_post_beta_lambda_conditional(
        y, observed, M_new, candidate, prior_mean, prior_var_vec);
      if(obj_new >= obj_old){
        regression_vector = candidate;
        accepted = true;
        break;
      }
      step *= 0.5;
    }
    if(!accepted) break;
    
  }
  arma::mat result((k+q), (k+q+1));
  result.col(0) = regression_vector; // MAP
  
  arma::mat I = arma::diagmat(arma::ones(k+q));
  arma::mat H_inv = inv(H); //inverse hessian
  result(arma::span(0, k+q-1), arma::span(1, k+q)) = H_inv;
  
  return result;
}


arma::mat stabilize_covariance_for_mvnrnd(const arma::mat& Sigma) {
  arma::mat Sigma_sym = 0.5 * (Sigma + Sigma.t());

  arma::vec eigval;
  arma::mat eigvec;
  bool eig_ok = arma::eig_sym(eigval, eigvec, Sigma_sym);
  if (!eig_ok || eigval.has_nan() || eigval.has_inf()) {
    Rcpp::stop("Cannot stabilize covariance matrix for mvnrnd(): eigen decomposition failed.");
  }

  double scale = std::max(1.0, arma::as_scalar(arma::max(arma::abs(eigval))));
  double floor = 1e-10 * scale;
  eigval.transform([floor](double x) { return x < floor ? floor : x; });

  arma::mat Sigma_psd = eigvec * arma::diagmat(eigval) * eigvec.t();
  Sigma_psd = 0.5 * (Sigma_psd + Sigma_psd.t());
  Sigma_psd.diag() += floor;
  return Sigma_psd;
}

// [[Rcpp::export]]. 
arma::mat gen_mvnrnd(arma::vec mu, arma::mat Sigma, int n_MC=100){
  int k = Sigma.n_rows;
  arma::mat Sigma_psd = stabilize_covariance_for_mvnrnd(Sigma);
  arma::mat samples(n_MC, k);
  for(int i=0; i<n_MC; ++i) {
    samples(i, arma::span(0, k-1)) = arma::mvnrnd(mu, Sigma_psd).t();
  }
  return samples;
}


// [[Rcpp::export]]
List map_regression_coeffs_mvlogit(
    arma::mat Y, arma::mat observed_mat, arma::mat X, arma::mat M, arma::mat Beta_init, arma::mat Lambda_init,
    arma::vec prior_var_beta, arma::vec prior_var_lambda, int n_MC=100, int max_it=200, double epsilon=0.001,
    double C_lambda=5, double C_mu=8, double C_beta=5, double step_size = 1){  
  
  int p = Y.n_cols;
  int n = Y.n_rows;
  int k = M.n_cols;
  int q = X.n_cols;
  
  arma::mat inits = join_rows(Beta_init, Lambda_init);
  arma::cube Lambda_samples(n_MC, k, p);
  arma::cube Beta_samples(n_MC, q, p);
  arma::mat Lambda_hat(p, k);
  arma::mat Beta_hat(p, q);
  arma::mat res_j(k+q, k+q);
  arma::cube V_js(k+q, k+q, p);
  arma::cube Lambda_covs(k+q, k+q, p);
  arma::mat V_j(k+q, k+q);
  arma::vec D(p);
  
  double rho_sq = 1;
  //omp_set_dynamic(1);
  //omp_set_num_threads(4);
  //#pragma omp parallel for 
  for(int j=0; j<p; ++j){
    res_j = optimize_regression_coeffs_logit(
      Y.col(j), observed_mat.col(j), X, M, Beta_init.row(j).t(), Lambda_init.row(j).t(), prior_var_beta(j), 
      prior_var_lambda(j),  max_it, epsilon, C_lambda=C_lambda, C_mu=C_mu, C_beta=C_beta, step_size);
    arma::mat V_j = res_j(arma::span(0, k+q-1), arma::span(1, k+q));
    V_j = -0.5*(V_j + V_j.t());
    D(j) = arma::trace(V_j(arma::span(q, k+q-1), arma::span(q, k+q-1)));
    V_js.slice(j) = V_j;
    
    arma::vec regr_vect_mean = res_j(arma::span(0, k+q-1), 0);
    Lambda_hat.row(j) = regr_vect_mean(arma::span(q, (k+q-1))).t();
    Beta_hat.row(j) = regr_vect_mean(arma::span(0, (q-1))).t();
    arma::mat sample_j(n_MC, k + q);
    if (n_MC <= 1) {
      sample_j.row(0) = regr_vect_mean.t();
    } else {
      sample_j = gen_mvnrnd(regr_vect_mean, rho_sq*V_j, n_MC);
    }
    Beta_samples.slice(j) = sample_j(arma::span(0, n_MC-1), arma::span(0, q-1));
    Lambda_samples.slice(j) = sample_j(arma::span(0, n_MC-1), arma::span(q, k+q-1));
  }
  return List::create(Named("Lambda_samples") = Lambda_samples,
                      Named("Beta_samples") = Beta_samples,
                      Named("Lambda_hat") = Lambda_hat,
                      Named("Covariances") = V_js,
                      Named("Beta_hat") = Beta_hat,
                      Named("D") = D);
}

// [[Rcpp::export]]
arma::mat compute_inv_hessian_logit(
    arma::vec y, arma::vec observed, arma::mat X_tilde, arma::vec regr_coeff, 
    double prior_var_beta, double prior_var_lambda, int q){
  
  int n = y.size();
  int q_tilde = X_tilde.n_cols;
  int k = q_tilde - q;
  
  arma::mat hessian(q_tilde, q_tilde, arma::fill::zeros);
  //omp_set_dynamic(1);
  //omp_set_num_threads(4);
  //#pragma omp parallel for 
  for(int i=0; i<n; ++i){
    if(observed(i)==1) {
      double eta_lambda_i = dot(X_tilde.row(i), regr_coeff);
      double F_i = F_logistic(eta_lambda_i);
      hessian = hessian + F_i * (1- F_i) * X_tilde(i, arma::span(0, q_tilde-1)).t() * X_tilde(i, arma::span(0,q_tilde-1));
    }
  }
  arma::vec tau_1 = 1/prior_var_beta * arma::ones(q);
  arma::vec tau_2 = 1/prior_var_lambda * arma::ones(k);
  arma::vec prior_var_inv = join_cols(tau_1, tau_2);
  
  arma::mat D = arma::diagmat(join_cols(tau_1, tau_2)); 
  hessian = hessian + D;
  arma::mat result = inv(-hessian); //inverse hessian
  return result;
  
}


// [[Rcpp::export]]
arma::mat compute_D(
    arma::mat Y, arma::mat observed_mat, arma::mat X, arma::mat M, arma::mat Beta, arma::mat Lambda,
    arma::vec prior_var_beta, arma::vec prior_var_lambda, int n_MC=100){
  
  int p = Y.n_cols;
  int k = M.n_cols;
  int q = X.n_cols;
  
  arma::mat V_j(k+q, k+q);
  arma::vec D(p);
  arma::mat X_tilde = join_rows(X, M);
  arma::mat regr_coeffs = join_rows(Beta, Lambda);
  //omp_set_dynamic(1);
  //omp_set_num_threads(4);
  //#pragma omp parallel for 
  
  for(int j=0; j<p; ++j){
    V_j = compute_inv_hessian_logit(
      Y.col(j), observed_mat.col(j), X_tilde, regr_coeffs.row(j).t(), prior_var_beta(j), 
      prior_var_lambda(j), q);
    V_j = -0.5*(V_j + V_j.t());
    D(j) = arma::trace(V_j(arma::span(q, k+q-1), arma::span(q, k+q-1)));
  }
  return  D;
}


// [[Rcpp::export]]
arma::mat compute_variances(
    arma::mat Y, arma::mat observed_mat, arma::mat X, arma::mat M, arma::mat Beta, arma::mat Lambda,
    arma::vec prior_var_beta, arma::vec prior_var_lambda, int n_MC=100){
  
  int p = Y.n_cols;
  int k = M.n_cols;
  int q = X.n_cols;
  
  arma::mat V_j(k+q, k+q);
  arma::mat D(p, k+q);
  arma::mat X_tilde = join_rows(X, M);
  arma::mat regr_coeffs = join_rows(Beta, Lambda);
  //omp_set_dynamic(1);
  //omp_set_num_threads(4);
  //#pragma omp parallel for 
  
  for(int j=0; j<p; ++j){
    V_j = compute_inv_hessian_logit(
      Y.col(j), observed_mat.col(j), X_tilde, regr_coeffs.row(j).t(), prior_var_beta(j), 
      prior_var_lambda(j), q);
    V_j = -0.5*(V_j + V_j.t());
    D(j, arma::span(0, k+q-1)) = arma::diagvec(V_j).t();
  }
  return D;
}


// [[Rcpp::export]]
List sample_full_conditional_logit(
    arma::mat Y, arma::mat observed_mat, arma::mat X, arma::mat M, arma::mat Beta, arma::mat Lambda,
    arma::vec prior_var_beta, arma::vec prior_var_lambda, int n_MC=100, double rho=1){
  
  
  int p = Y.n_cols;
  int n = Y.n_rows;
  int k = M.n_cols;
  int q = X.n_cols;
  
  arma::mat regr_coeffs = join_rows(Beta, Lambda);
  arma::cube Lambda_samples(n_MC, k, p);
  arma::cube Beta_samples(n_MC, q, p);
  arma::mat Lambda_hat(p, k);
  arma::mat Beta_hat(p, q);
  arma::mat res_j(k+q, k+q);
  arma::cube V_js(k+q, k+q, p);
  arma::cube Lambda_covs(k+q, k+q, p);
  arma::mat V_j(k+q, k+q);
  arma::vec D(p);
  
  double rho_sq = rho*rho;
  
  
  arma::mat X_tilde = join_rows(X, M);
  //omp_set_dynamic(1);
  //omp_set_num_threads(4);
  //#pragma omp parallel for 
  for(int j=0; j<p; ++j){
    
    V_j = compute_inv_hessian_logit(
      Y.col(j), observed_mat.col(j), X_tilde, regr_coeffs.row(j).t(), prior_var_beta(j), 
      prior_var_lambda(j), q);
    Lambda_hat.row(j) = regr_coeffs(j, arma::span(q, (k+q-1)));
    Beta_hat.row(j) = regr_coeffs(j, arma::span(0, (q-1)));
    V_j = -0.5*(V_j + V_j.t());
    D(j) = arma::trace(V_j(arma::span(q, k+q-1), arma::span(q, k+q-1)));
    arma::mat sample_j = gen_mvnrnd(regr_coeffs.row(j).t(), rho_sq*V_j, n_MC);
    V_js.slice(j) = V_j;
    Beta_samples.slice(j) = sample_j(arma::span(0, n_MC-1), arma::span(0, q-1));
    Lambda_samples.slice(j) = sample_j(arma::span(0, n_MC-1), arma::span(q, k+q-1));
    
  }
  return List::create(Named("Lambda_samples") = Lambda_samples,
                      Named("Beta_samples") = Beta_samples,
                      Named("Lambda_hat") = Lambda_hat,
                      Named("Covariances") = V_js,
                      Named("Beta_hat") = Beta_hat,
                      Named("D") = D);
}




arma::mat grad_latent_factors_logit(
    arma::vec y, arma::vec observed, arma::vec x, arma::mat Lambda, arma::mat Beta,
    arma::vec M_i, double prior_var) {
  // returns gradient and hessian of latent logistic regression log-lik + normal prior w.r.t. latent factors
  int p = y.size();
  int k = Lambda.n_cols;
  
  arma::vec scores(p, arma::fill::zeros);
  arma::vec grad_log_pi(k, arma::fill::zeros);
  arma::mat hessian(k, k, arma::fill::zeros);
  
  //omp_set_dynamic(1);
  //omp_set_num_threads(4);
  //#pragma omp parallel for 
  for(int i=0; i<p; ++i){
    scores(i) = 0;
    if(observed(i)==1){
      double eta_lambda_i = dot(Lambda.row(i), M_i);
      double Beta_x_i = dot(Beta.row(i), x);
      double F_i = F_logistic(Beta_x_i + eta_lambda_i);
      scores(i) = y(i) - F_i;
      grad_log_pi = grad_log_pi + scores(i)*Lambda(i, arma::span(0,k-1)).t();
      hessian = hessian + F_i * (1 -F_i) * 
        Lambda(i, arma::span(0, k-1)).t() * Lambda(i, arma::span(0,k-1));
    }
  }
  arma::mat I_k = arma::diagmat(arma::ones(k));
  grad_log_pi = grad_log_pi - M_i/prior_var;
  hessian = hessian + (1/prior_var)*I_k;
  arma::mat result(k, k+1);
  result(arma::span(0, k-1), 0) = grad_log_pi;
  result(arma::span(0, k -1), arma::span(1, k)) = -hessian;
  return result;
}

arma::mat grad_latent_factors_logit_fullobs(
    arma::vec y, arma::vec x, arma::mat Lambda, arma::mat Beta,
    arma::vec M_i, double prior_var) {
  // Full-observation specialization avoids carrying an n x p observed matrix.
  int p = y.size();
  int k = Lambda.n_cols;

  arma::vec grad_log_pi(k, arma::fill::zeros);
  arma::mat hessian(k, k, arma::fill::zeros);

  for(int i=0; i<p; ++i){
    double eta_lambda_i = dot(Lambda.row(i), M_i);
    double Beta_x_i = dot(Beta.row(i), x);
    double F_i = F_logistic(Beta_x_i + eta_lambda_i);
    double score_i = y(i) - F_i;
    grad_log_pi = grad_log_pi + score_i * Lambda(i, arma::span(0, k-1)).t();
    hessian = hessian + F_i * (1 - F_i) *
      Lambda(i, arma::span(0, k-1)).t() * Lambda(i, arma::span(0, k-1));
  }
  arma::mat I_k = arma::diagmat(arma::ones(k));
  grad_log_pi = grad_log_pi - M_i / prior_var;
  hessian = hessian + (1 / prior_var) * I_k;
  arma::mat result(k, k+1);
  result(arma::span(0, k-1), 0) = grad_log_pi;
  result(arma::span(0, k -1), arma::span(1, k)) = -hessian;
  return result;
}


arma::vec append_to_vec(arma::vec vec, double value) {
  vec.resize(vec.n_elem + 1);
  vec(vec.n_elem - 1) = value;
  return vec;
}

arma::vec optimize_latent_factors_logit(
    arma::vec y, arma::vec observed, arma::vec x, arma::mat Lambda, arma::mat Beta, arma::vec M_i_init, 
    double prior_var, int max_it, double epsilon, int n){
  
  int p = y.size();
  int k = Lambda.n_cols;
  int q = Beta.n_cols;
  
  arma::vec M_i = M_i_init;
  arma::mat H(k, k+1, arma::fill::zeros);
  for(int i = 1; i<=max_it; ++i){
    arma::mat res = grad_latent_factors_logit(
      y, observed, x, Lambda, Beta, M_i, prior_var);
    arma::vec u = res(arma::span(0, k-1), 0);
    H = res(arma::span(0, k-1), arma::span(1, k));
    arma::vec update;
    if (!arma::solve(update, H, -u)) break;
    double norm_update = norm(update, 2);
    if(norm_update<epsilon){
      break;
    }
    const double C_m = 2.0 * std::sqrt(std::log(static_cast<double>(k * n)));
    const double obj_old = log_post_latent_factor_conditional(
      y, observed, x, Lambda, Beta, M_i, prior_var);
    double step = 1.0;
    bool accepted = false;
    for(int ls = 0; ls < 25; ++ls){
      arma::vec candidate = M_i + step * update;
      for(int j = 0; j < k; ++j){
        if(candidate(j) >  C_m) candidate(j) =  C_m;
        else if(candidate(j) < -C_m) candidate(j) = -C_m;
      }
      const double obj_new = log_post_latent_factor_conditional(
        y, observed, x, Lambda, Beta, candidate, prior_var);
      if(obj_new >= obj_old){
        M_i = candidate;
        accepted = true;
        break;
      }
      step *= 0.5;
    }
    if(!accepted) break;
  }

  arma::vec result = M_i;
  return result;
}

arma::vec optimize_latent_factors_logit_fullobs(
    arma::vec y, arma::vec x, arma::mat Lambda, arma::mat Beta, arma::vec M_i_init,
    double prior_var, int max_it, double epsilon, int n){

  int k = Lambda.n_cols;
  arma::vec M_i = M_i_init;
  arma::mat H(k, k+1, arma::fill::zeros);
  for(int i = 1; i<=max_it; ++i){
    arma::mat res = grad_latent_factors_logit_fullobs(
      y, x, Lambda, Beta, M_i, prior_var);
    arma::vec u = res(arma::span(0, k-1), 0);
    H = res(arma::span(0, k-1), arma::span(1, k));
    arma::vec update;
    if (!arma::solve(update, H, -u)) break;
    double norm_update = norm(update, 2);
    if(norm_update<epsilon){
      break;
    }
    const double C_m = 2.0 * std::sqrt(std::log(static_cast<double>(k * n)));
    const double obj_old = log_post_latent_factor_conditional_fullobs(
      y, x, Lambda, Beta, M_i, prior_var);
    double step = 1.0;
    bool accepted = false;
    for(int ls = 0; ls < 25; ++ls){
      arma::vec candidate = M_i + step * update;
      for(int j = 0; j < k; ++j){
        if(candidate(j) >  C_m) candidate(j) =  C_m;
        else if(candidate(j) < -C_m) candidate(j) = -C_m;
      }
      const double obj_new = log_post_latent_factor_conditional_fullobs(
        y, x, Lambda, Beta, candidate, prior_var);
      if(obj_new >= obj_old){
        M_i = candidate;
        accepted = true;
        break;
      }
      step *= 0.5;
    }
    if(!accepted) break;
  }

  return M_i;
}


// Shared-Hessian MAP for M — mirrors map_beta_lambda_plus_blocks_cpp.
//
// Algorithm (per Newton iteration):
//   PHASE 1  multi-threaded BLAS:
//     eta   = B_x + M * Lambda^T             (n x p)
//     F     = sigmoid(eta),  W = F(1-F)
//     w_bar = colMeans(W)                     (p-vec, shared weight)
//     H_sh  = Lambda^T diag(w_bar) Lambda + I/prior_var   (k x k, ONE Hessian)
//     Grad  = (Y-F)*Lambda - M/prior_var      (n x k)
//     Dir   = solve(H_sh, Grad^T)^T           (n x k, same Hessian for all sites)
//     convergence check: max row-norm(Dir) < epsilon
//
//   PHASE 2  OMP parallel over m_block-site blocks (single-threaded BLAS inside):
//     For each block b:
//       line-search:  eta_c = B_x[b,:] + M_c * Lambda^T   (uses cached B_x)
//                    accept if obj(M_c) >= obj(M_b)
//
// Memory: B_x (n x p) allocated once per call; eta + W + resid each n x p.
//   Total extra ~4x n*p*8 bytes (e.g. 4 x 4.7 GB for n=2299, p=255K).
// [[Rcpp::export]]
List map_latent_factors_shared_hessian_fullobs(
    const arma::mat& Y,          // n x p
    const arma::mat& X,          // n x q
    const arma::mat& Lambda,     // p x k
    const arma::mat& Beta,       // p x q
    const arma::mat& M_init,     // n x k
    double prior_var = 1.0,
    double epsilon   = 1e-3,
    int    n_threads = 1,
    int    m_block   = 1000,     // unused (kept for API compat)
    double step_init = 1.0)      // warm-start delta (carried across R-level outer iters)
{
  typedef void (*set_nt_t)(int);
  // Try C name first, then Fortran-convention suffix (some containers/FlexiBLAS).
  set_nt_t blas_nt = reinterpret_cast<set_nt_t>(
      dlsym(RTLD_DEFAULT, "openblas_set_num_threads"));
  if (!blas_nt)
    blas_nt = reinterpret_cast<set_nt_t>(
        dlsym(RTLD_DEFAULT, "openblas_set_num_threads_"));

  const arma::uword n   = Y.n_rows;
  const arma::uword p   = Y.n_cols;
  const arma::uword k   = M_init.n_cols;
  const double C_m      = 2.0 * std::sqrt(std::log(static_cast<double>(k) * n + 1.0));

  auto sp_fn = [](double x) -> double {
    if (x >  35.0) return x;
    if (x < -35.0) return std::exp(x);
    return std::log1p(std::exp(x));
  };

  // Precompute B_x = X * Beta^T  (n x p) — fixed, reused in every LS step.
  if (blas_nt && n_threads > 0) blas_nt(n_threads);
  arma::mat B_x = X * Beta.t();   // n x p
  arma::mat M   = M_init;

  // Warm-start delta: persists across inner iterations; caller passes step_out
  // back as step_init next outer iteration.
  double delta = std::max(1e-8, step_init);
  const double delta_max = std::max(delta, C_m);

  for (int it = 0; it < 1; ++it) {

    // ---- PHASE 1: multi-threaded BLAS for matmuls, OMP for element-wise ------
    if (blas_nt && n_threads > 0) blas_nt(n_threads);
    arma::mat eta = B_x + M * Lambda.t();   // n x p, BLAS

    // Sigmoid + residual: parallelise with OMP.
    // Replace W(n,p) with per-thread row accumulators (p-vec each) to avoid
    // allocating an n×p weight matrix and the subsequent sequential arma::mean.
    if (blas_nt) blas_nt(1);
    const int nt = std::max(1, n_threads);
    std::vector<arma::vec> w_loc(nt, arma::vec(p, arma::fill::zeros));
    arma::mat resid(n, p);
#ifdef _OPENMP
    if (n_threads > 0) omp_set_num_threads(n_threads);
#pragma omp parallel for schedule(static)
#endif
    for (arma::sword i = 0; i < static_cast<arma::sword>(n); ++i) {
      int tid = 0;
#ifdef _OPENMP
      tid = omp_get_thread_num();
#endif
      for (arma::uword j = 0; j < p; ++j) {
        const double fi  = 1.0 / (1.0 + std::exp(-eta(i, j)));
        w_loc[tid](j) += fi * (1.0 - fi);
        resid(i, j)    = Y(i, j) - fi;
      }
    }
    // Reduce thread-local w accumulators → w_bar (sequential over nt, not n*p).
    arma::vec w_sum(p, arma::fill::zeros);
    for (int t = 0; t < nt; ++t) w_sum += w_loc[t];

    // Shared Hessian (k x k): back to multi-threaded BLAS for the DGEMM.
    if (blas_nt && n_threads > 0) blas_nt(n_threads);
    arma::vec w_bar = arma::clamp(w_sum / static_cast<double>(n), 1e-8, arma::datum::inf);
    arma::mat LW    = Lambda.each_col() % arma::sqrt(w_bar);        // p x k
    arma::mat H_sh  = LW.t() * LW + arma::eye(k, k) / prior_var;   // k x k

    arma::mat Grad  = resid * Lambda - M / prior_var;               // n x k, BLAS
    arma::mat Dir_t;
    if (!arma::solve(Dir_t, H_sh, Grad.t(), arma::solve_opts::fast)) break;
    arma::mat Dir = Dir_t.t();  // n x k

    // Convergence
    double max_norm = arma::max(arma::sqrt(arma::sum(Dir % Dir, 1)));
    if (max_norm < epsilon) break;

    // ---- PHASE 2: TRUE TRUST REGION (rho ratio + warm-start delta) ----------
    // Maximisation framework: actual_red = obj_new - obj_old (>0 = better).
    // pred_red = Grad^T Step - 0.5 * sum_i step_i^T H_sh step_i  (>0 for descent).
    // rho = actual / pred.  Accept if rho > eta_accept.  Expand/shrink delta by rho.
    arma::mat SP_eta = eta; SP_eta.transform(sp_fn);
    const double obj_old = arma::accu(Y % eta - SP_eta)
                         - 0.5 * arma::accu(M % M) / prior_var;

    bool accepted = false;
    double try_delta = delta;
    for (int tr = 0; tr < 8; ++tr) {
      const double scale = trust_region_scale_from_norm(max_norm, try_delta);
      const arma::mat Step = arma::clamp(M + scale * Dir, -C_m, C_m) - M;

      // Predicted reduction from shared quadratic model (sum over sites):
      // pred = Grad^T Step - 0.5 * trace(Step H_sh Step^T)  per-row, summed
      const double pred = arma::accu(Grad % Step)
                        - 0.5 * arma::accu((Step * H_sh) % Step);

      const arma::mat M_c = M + Step;
      if (blas_nt && n_threads > 0) blas_nt(n_threads);
      arma::mat eta_c = B_x + M_c * Lambda.t();
      arma::mat SP_c = eta_c; SP_c.transform(sp_fn);
      const double obj_new = arma::accu(Y % eta_c - SP_c)
                           - 0.5 * arma::accu(M_c % M_c) / prior_var;

      const double actual = obj_new - obj_old;
      const double rho    = (pred > 1e-14) ? (actual / pred) : -1.0;

      if (rho > 0.1) {
        M = M_c;
        // Expand radius based on quality of model
        if (rho > 0.75)
          delta = std::min(delta_max, try_delta * 2.0);
        else
          delta = try_delta;   // model adequate, keep radius
        accepted = true;
        break;
      }
      // Poor model: shrink aggressively
      try_delta *= (rho > 0.25) ? 0.5 : 0.25;
    }
    if (!accepted) {
      delta = std::max(1e-8, delta * 0.25);  // single moderate shrink, not cascade-collapse
      break;
    }
  }
  if (blas_nt && n_threads > 0) blas_nt(n_threads);  // restore for caller
  return List::create(_["M"] = M, _["step_out"] = delta);
}

// Block-preconditioned shared-Hessian MAP for M, full-observation case.
//
// Species blocks accumulate the all-site gradient and the k x k shared Hessian;
// one Newton direction is solved from the sums, then a global trust-region
// acceptance step verifies the full all-species M objective.  The old all-species Lhat is computed during
// the gradient/Hessian pass and reused through trust-region acceptance.
// [[Rcpp::export]]
List map_latent_factors_shared_hessian_species_blocks_fullobs(
    const arma::mat& Y,          // n x p
    const arma::mat& X,          // n x q
    const arma::mat& Lambda,     // p x k
    const arma::mat& Beta,       // p x q
    const arma::mat& M_init,     // n x k
    double prior_var = 1.0,
    double epsilon   = 1e-3,
    int    n_threads = 1,
    int    species_block = 1000,
    double step_init = 1.0,
    int    max_ls = 20,
    Rcpp::Nullable<Rcpp::IntegerVector> species_order_r = R_NilValue,
    double ll_init = NA_REAL)
{
  typedef void (*set_nt_t)(int);
  set_nt_t blas_nt = reinterpret_cast<set_nt_t>(
      dlsym(RTLD_DEFAULT, "openblas_set_num_threads"));
  if (!blas_nt)
    blas_nt = reinterpret_cast<set_nt_t>(
        dlsym(RTLD_DEFAULT, "openblas_set_num_threads_"));

  const arma::uword n   = Y.n_rows;
  const arma::uword p   = Y.n_cols;
  const arma::uword k   = M_init.n_cols;
  const arma::uword sb  = static_cast<arma::uword>(std::max(1, species_block));
  const double C_m      = 2.0 * std::sqrt(std::log(static_cast<double>(k) * n + 1.0));
  arma::uvec sord;
  bool use_sord = false;
  if (species_order_r.isNotNull()) {
    sord = Rcpp::as<arma::uvec>(Rcpp::IntegerVector(species_order_r));
    use_sord = (sord.n_elem == p);
  }

  auto total0 = std::chrono::steady_clock::now();
  double subset_seconds = 0.0;
  double xbeta_seconds = 0.0;
  double eta_seconds = 0.0;
  double sigmoid_seconds = 0.0;
  double hessian_seconds = 0.0;
  double grad_seconds = 0.0;
  double solve_seconds = 0.0;
  double old_lhat_seconds = 0.0;
  double ls_eta_seconds = 0.0;
  double ls_lhat_seconds = 0.0;
  double current_ll = NA_REAL;
  double last_ll_old = NA_REAL;
  double last_ll_new = NA_REAL;
  double last_lhat_old = NA_REAL;
  double last_lhat_new = NA_REAL;
  double last_pred = NA_REAL;
  double last_actual = NA_REAL;
  double last_rho = NA_REAL;
  double last_try_delta = NA_REAL;
  double last_max_dir_norm = NA_REAL;
  double last_step_norm = NA_REAL;
  double accepted_step = 0.0;
  int total_ls = 0;
  arma::mat M = M_init;

  // Warm-start delta across inner iterations; caller passes step_out back
  // as step_init on the next R-level outer iteration.
  double delta = std::max(1e-8, step_init);
  const double delta_max = std::max(delta, C_m);

  for (int it = 0; it < 1; ++it) {
    arma::mat Grad_accum(n, k, arma::fill::zeros);   // for TR pred reduction
    arma::mat H_sum(k, k, arma::fill::zeros);        // sum of block Hessians
    double ll_old = 0.0;
    const bool use_ll_init = std::isfinite(ll_init);
    double obj_old = -0.5 * arma::accu(M % M) / prior_var;
    const int n_blocks = static_cast<int>((p + sb - 1) / sb);
    const int nt = std::max(1, n_threads);
    std::vector<arma::mat> Grad_loc(nt, arma::mat(n, k, arma::fill::zeros));
    std::vector<arma::mat> H_loc(nt, arma::mat(k, k, arma::fill::zeros));

    if (blas_nt) blas_nt(1);
#ifdef _OPENMP
    if (n_threads > 0) omp_set_num_threads(n_threads);
    omp_set_nested(0);
    omp_set_max_active_levels(1);
#pragma omp parallel for schedule(static) reduction(+:ll_old,subset_seconds,xbeta_seconds,eta_seconds,sigmoid_seconds,old_lhat_seconds,hessian_seconds,grad_seconds,solve_seconds)
#endif
    for (arma::sword block_id = 0; block_id < static_cast<arma::sword>(n_blocks); ++block_id) {
      const arma::uword s = static_cast<arma::uword>(block_id) * sb;
      const arma::uword e = std::min(s + sb, p);
      int tid = 0;
#ifdef _OPENMP
      tid = omp_get_thread_num();
#endif
      auto t0 = std::chrono::steady_clock::now();
      const arma::uvec idx = use_sord
        ? (sord.subvec(s, e - 1) - 1)
        : arma::regspace<arma::uvec>(s, e - 1);

      arma::mat Y_b      = Y.cols(idx);
      arma::mat Lambda_b = Lambda.rows(idx);
      arma::mat Beta_b   = Beta.rows(idx);
      auto t1 = std::chrono::steady_clock::now();
      subset_seconds += std::chrono::duration<double>(t1 - t0).count();

      t0 = std::chrono::steady_clock::now();
      arma::mat XB_b = X * Beta_b.t();
      t1 = std::chrono::steady_clock::now();
      xbeta_seconds += std::chrono::duration<double>(t1 - t0).count();

      t0 = std::chrono::steady_clock::now();
      arma::mat eta = XB_b + M * Lambda_b.t();
      t1 = std::chrono::steady_clock::now();
      eta_seconds += std::chrono::duration<double>(t1 - t0).count();

      t0 = std::chrono::steady_clock::now();
      arma::mat resid(n, e - s);
      arma::vec w_sum(e - s, arma::fill::zeros);
      double ll_block = 0.0;
      for (arma::uword j = 0; j < e - s; ++j) {
        double w_j = 0.0;
        for (arma::uword i = 0; i < n; ++i) {
          const double zij = eta(i, j);
          const double yij = Y_b(i, j);
          double pv = 0.0;
          if (use_ll_init) {
            pv = F_logistic(zij);
          } else {
            double ll_ij = 0.0;
            bernoulli_logit_prob_ll_stable(yij, zij, pv, ll_ij);
            ll_block += ll_ij;
          }
          w_j += pv * (1.0 - pv);
          resid(i, j) = yij - pv;
        }
        w_sum(j) = w_j;
      }
      ll_old += ll_block;
      t1 = std::chrono::steady_clock::now();
      sigmoid_seconds += std::chrono::duration<double>(t1 - t0).count();
      old_lhat_seconds += std::chrono::duration<double>(t1 - t0).count();

      t0 = std::chrono::steady_clock::now();
      arma::vec w_bar = arma::clamp(w_sum / static_cast<double>(n), 1e-8, arma::datum::inf);
      arma::mat LW = Lambda_b.each_col() % arma::sqrt(w_bar);
      arma::mat H_b = LW.t() * LW + arma::eye(k, k) / prior_var / static_cast<double>(n_blocks);
      t1 = std::chrono::steady_clock::now();
      hessian_seconds += std::chrono::duration<double>(t1 - t0).count();

      t0 = std::chrono::steady_clock::now();
      arma::mat Grad_b = resid * Lambda_b - M / prior_var / static_cast<double>(n_blocks);
      t1 = std::chrono::steady_clock::now();
      grad_seconds += std::chrono::duration<double>(t1 - t0).count();

      Grad_loc[tid] += Grad_b;
      H_loc[tid]    += H_b;
    }
    for (int t = 0; t < nt; ++t) {
      Grad_accum += Grad_loc[t];
      H_sum      += H_loc[t];
    }
    obj_old += ll_old;
    if (use_ll_init) ll_old = ll_init;
    obj_old = ll_old - 0.5 * arma::accu(M % M) / prior_var;
    current_ll = ll_old;
    last_ll_old = ll_old;
    last_lhat_old = obj_old;

    if (n_blocks == 0) break;
    // Full shared-Hessian Newton direction: summed gradient against summed Hessian.
    auto ts = std::chrono::steady_clock::now();
    arma::mat Dir;
    if (!arma::solve(Dir, H_sum, Grad_accum.t(), arma::solve_opts::fast)) break;
    Dir = Dir.t();
    solve_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();

    const double max_norm = arma::max(arma::sqrt(arma::sum(Dir % Dir, 1)));
    last_max_dir_norm = max_norm;
    if (max_norm < epsilon) break;

    // ---- TRUE TRUST REGION: rho ratio + warm-start delta -------------------
    bool accepted = false;
    double try_delta = delta;
    for (int tr = 0; tr < max_ls; ++tr) {
      ++total_ls;
      const double scale = trust_region_scale_from_norm(max_norm, try_delta);
      arma::mat Step = arma::clamp(M + scale * Dir, -C_m, C_m) - M;
      const double step_norm = arma::max(arma::sqrt(arma::sum(Step % Step, 1)));
      last_try_delta = try_delta;
      last_step_norm = step_norm;

      // Predicted reduction from the full quadratic model
      const double pred = arma::accu(Grad_accum % Step)
                        - 0.5 * arma::accu((Step * H_sum) % Step);

      arma::mat M_c = M + Step;
      double obj_new = -0.5 * arma::accu(M_c % M_c) / prior_var;
      double ll_new = 0.0;

      if (blas_nt) blas_nt(1);
#ifdef _OPENMP
      if (n_threads > 0) omp_set_num_threads(n_threads);
#pragma omp parallel for schedule(static) reduction(+:ll_new,subset_seconds,ls_eta_seconds,ls_lhat_seconds)
#endif
      for (arma::sword block_id = 0; block_id < static_cast<arma::sword>(n_blocks); ++block_id) {
        const arma::uword s = static_cast<arma::uword>(block_id) * sb;
        const arma::uword e = std::min(s + sb, p);
        auto t0 = std::chrono::steady_clock::now();
        const arma::uvec idx = use_sord
          ? (sord.subvec(s, e - 1) - 1)
          : arma::regspace<arma::uvec>(s, e - 1);

        arma::mat Y_b      = Y.cols(idx);
        arma::mat Lambda_b = Lambda.rows(idx);
        arma::mat Beta_b   = Beta.rows(idx);
        auto t1 = std::chrono::steady_clock::now();
        subset_seconds += std::chrono::duration<double>(t1 - t0).count();

        t0 = std::chrono::steady_clock::now();
        arma::mat XB_b = X * Beta_b.t();
        arma::mat eta_c = XB_b + M_c * Lambda_b.t();
        t1 = std::chrono::steady_clock::now();
        ls_eta_seconds += std::chrono::duration<double>(t1 - t0).count();

        t0 = std::chrono::steady_clock::now();
        double ll_block = 0.0;
        for (arma::uword j = 0; j < e - s; ++j) {
          for (arma::uword i = 0; i < n; ++i) {
            const double zc = eta_c(i, j);
            ll_block += Y_b(i, j) * zc - softplus(zc);
          }
        }
        ll_new += ll_block;
        t1 = std::chrono::steady_clock::now();
        ls_lhat_seconds += std::chrono::duration<double>(t1 - t0).count();
      }
      obj_new += ll_new;

      const double actual = obj_new - obj_old;
      const double rho    = (pred > 1e-14) ? (actual / pred) : -1.0;

      last_ll_new    = ll_new;
      last_lhat_new  = obj_new;
      last_pred      = pred;
      last_actual    = actual;
      last_rho       = rho;
      if (rho > 0.1) {
        M = M_c;
        current_ll = ll_new;
        delta = (rho > 0.75) ? std::min(delta_max, try_delta * 2.0) : try_delta;
        accepted = true;
        accepted_step = step_norm;
        break;
      }
      try_delta *= (rho > 0.25) ? 0.5 : 0.25;
    }
    if (!accepted) {
      delta = std::max(1e-8, delta * 0.25);  // single moderate shrink, not cascade-collapse
      break;
    }
  }

  if (blas_nt && n_threads > 0) blas_nt(n_threads);
  auto total1 = std::chrono::steady_clock::now();
  const double total_seconds = std::chrono::duration<double>(total1 - total0).count();
  return List::create(
    _["M"] = M,
    _["lhat_time"] = old_lhat_seconds + ls_lhat_seconds,
    _["n_species"] = static_cast<int>(p),
    _["accepted"] = accepted_step > 0.0,
    _["step"] = accepted_step,
    _["ll"] = current_ll,
    _["ll_old"] = last_ll_old,
    _["ll_new"] = last_ll_new,
    _["lhat_old"] = last_lhat_old,
    _["lhat_new"] = last_lhat_new,
    _["pred"] = last_pred,
    _["actual"] = last_actual,
    _["rho"] = last_rho,
    _["try_delta"] = last_try_delta,
    _["max_dir_norm"] = last_max_dir_norm,
    _["step_norm"] = last_step_norm,
    _["ls_trials"] = total_ls,
    _["step_out"] = delta,
    _["timing"] = NumericVector::create(
      _["subset"] = subset_seconds,
      _["XB"] = xbeta_seconds,
      _["eta"] = eta_seconds,
      _["sigmoid"] = sigmoid_seconds,
      _["hessian"] = hessian_seconds,
      _["grad"] = grad_seconds,
      _["solve"] = solve_seconds,
      _["old_lhat"] = old_lhat_seconds,
      _["ls_eta"] = ls_eta_seconds,
      _["ls_lhat"] = ls_lhat_seconds,
      _["total"] = total_seconds
    )
  );
}

// Block-preconditioned shared-Hessian MAP for M — masked / partially-observed case.
// Same structure as map_latent_factors_shared_hessian_species_blocks_fullobs but
// observed_mat (n x p, 1.0 = observed) gates all likelihood contributions.
// [[Rcpp::export]]
List map_latent_factors_shared_hessian_species_blocks(
    const arma::mat& Y,
    const arma::mat& observed_mat,   // n x p, 1.0 = observed
    const arma::mat& X,
    const arma::mat& Lambda,         // p x k
    const arma::mat& Beta,           // p x q
    const arma::mat& M_init,         // n x k
    double prior_var = 1.0,
    double epsilon   = 1e-3,
    int    n_threads = 1,
    int    species_block = 1000,
    double step_init = 1.0,
    int    max_ls = 20,
    Rcpp::Nullable<Rcpp::IntegerVector> species_order_r = R_NilValue,
    double ll_init = NA_REAL)
{
  typedef void (*set_nt_t)(int);
  set_nt_t blas_nt = reinterpret_cast<set_nt_t>(
      dlsym(RTLD_DEFAULT, "openblas_set_num_threads"));
  if (!blas_nt)
    blas_nt = reinterpret_cast<set_nt_t>(
        dlsym(RTLD_DEFAULT, "openblas_set_num_threads_"));

  const arma::uword n   = Y.n_rows;
  const arma::uword p   = Y.n_cols;
  const arma::uword k   = M_init.n_cols;
  const arma::uword sb  = static_cast<arma::uword>(std::max(1, species_block));
  const double C_m      = 2.0 * std::sqrt(std::log(static_cast<double>(k) * n + 1.0));
  arma::uvec sord;
  bool use_sord = false;
  if (species_order_r.isNotNull()) {
    sord = Rcpp::as<arma::uvec>(Rcpp::IntegerVector(species_order_r));
    use_sord = (sord.n_elem == p);
  }

  auto total0 = std::chrono::steady_clock::now();
  double subset_seconds = 0.0;
  double xbeta_seconds = 0.0;
  double eta_seconds = 0.0;
  double sigmoid_seconds = 0.0;
  double hessian_seconds = 0.0;
  double grad_seconds = 0.0;
  double solve_seconds = 0.0;
  double old_lhat_seconds = 0.0;
  double ls_eta_seconds = 0.0;
  double ls_lhat_seconds = 0.0;
  double current_ll = NA_REAL;
  double last_ll_old = NA_REAL;
  double last_ll_new = NA_REAL;
  double last_lhat_old = NA_REAL;
  double last_lhat_new = NA_REAL;
  double last_pred = NA_REAL;
  double last_actual = NA_REAL;
  double last_rho = NA_REAL;
  double last_try_delta = NA_REAL;
  double last_max_dir_norm = NA_REAL;
  double last_step_norm = NA_REAL;
  double accepted_step = 0.0;
  int total_ls = 0;
  arma::mat M = M_init;

  // Warm-start delta across inner iterations; caller passes step_out back
  // as step_init on the next R-level outer iteration.
  double delta = std::max(1e-8, step_init);
  const double delta_max = std::max(delta, C_m);

  for (int it = 0; it < 1; ++it) {
    arma::mat Grad_accum(n, k, arma::fill::zeros);   // for TR pred reduction
    arma::mat H_sum(k, k, arma::fill::zeros);        // sum of block Hessians
    double ll_old = 0.0;
    const bool use_ll_init = std::isfinite(ll_init);
    double obj_old = -0.5 * arma::accu(M % M) / prior_var;
    const int n_blocks = static_cast<int>((p + sb - 1) / sb);
    const int nt = std::max(1, n_threads);
    std::vector<arma::mat> Grad_loc(nt, arma::mat(n, k, arma::fill::zeros));
    std::vector<arma::mat> H_loc(nt, arma::mat(k, k, arma::fill::zeros));

    if (blas_nt) blas_nt(1);
#ifdef _OPENMP
    if (n_threads > 0) omp_set_num_threads(n_threads);
    omp_set_nested(0);
    omp_set_max_active_levels(1);
#pragma omp parallel for schedule(static) reduction(+:ll_old,subset_seconds,xbeta_seconds,eta_seconds,sigmoid_seconds,old_lhat_seconds,hessian_seconds,grad_seconds,solve_seconds)
#endif
    for (arma::sword block_id = 0; block_id < static_cast<arma::sword>(n_blocks); ++block_id) {
      const arma::uword s = static_cast<arma::uword>(block_id) * sb;
      const arma::uword e = std::min(s + sb, p);
      int tid = 0;
#ifdef _OPENMP
      tid = omp_get_thread_num();
#endif
      auto t0 = std::chrono::steady_clock::now();
      const arma::uvec idx = use_sord
        ? (sord.subvec(s, e - 1) - 1)
        : arma::regspace<arma::uvec>(s, e - 1);

      arma::mat Y_b      = Y.cols(idx);
      arma::mat Obs_b    = observed_mat.cols(idx);
      arma::mat Lambda_b = Lambda.rows(idx);
      arma::mat Beta_b   = Beta.rows(idx);
      auto t1 = std::chrono::steady_clock::now();
      subset_seconds += std::chrono::duration<double>(t1 - t0).count();

      t0 = std::chrono::steady_clock::now();
      arma::mat XB_b = X * Beta_b.t();
      t1 = std::chrono::steady_clock::now();
      xbeta_seconds += std::chrono::duration<double>(t1 - t0).count();

      t0 = std::chrono::steady_clock::now();
      arma::mat eta = XB_b + M * Lambda_b.t();
      t1 = std::chrono::steady_clock::now();
      eta_seconds += std::chrono::duration<double>(t1 - t0).count();

      t0 = std::chrono::steady_clock::now();
      arma::mat resid(n, e - s);
      arma::vec w_sum(e - s, arma::fill::zeros);
      double ll_block = 0.0;
      for (arma::uword j = 0; j < e - s; ++j) {
        double w_j = 0.0;
        for (arma::uword i = 0; i < n; ++i) {
          const double obs_ij = Obs_b(i, j);
          const double zij = eta(i, j);
          const double yij = Y_b(i, j);
          double pv = 0.0;
          if (use_ll_init) {
            pv = F_logistic(zij);
          } else {
            double ll_ij = 0.0;
            bernoulli_logit_prob_ll_stable(yij, zij, pv, ll_ij);
            ll_block += ll_ij * obs_ij;
          }
          w_j += pv * (1.0 - pv) * obs_ij;
          resid(i, j) = (yij - pv) * obs_ij;
        }
        w_sum(j) = w_j;
      }
      ll_old += ll_block;
      t1 = std::chrono::steady_clock::now();
      sigmoid_seconds += std::chrono::duration<double>(t1 - t0).count();
      old_lhat_seconds += std::chrono::duration<double>(t1 - t0).count();

      t0 = std::chrono::steady_clock::now();
      arma::vec w_bar = arma::clamp(w_sum / static_cast<double>(n), 1e-8, arma::datum::inf);
      arma::mat LW = Lambda_b.each_col() % arma::sqrt(w_bar);
      arma::mat H_b = LW.t() * LW + arma::eye(k, k) / prior_var / static_cast<double>(n_blocks);
      t1 = std::chrono::steady_clock::now();
      hessian_seconds += std::chrono::duration<double>(t1 - t0).count();

      t0 = std::chrono::steady_clock::now();
      arma::mat Grad_b = resid * Lambda_b - M / prior_var / static_cast<double>(n_blocks);
      t1 = std::chrono::steady_clock::now();
      grad_seconds += std::chrono::duration<double>(t1 - t0).count();

      Grad_loc[tid] += Grad_b;
      H_loc[tid]    += H_b;
    }
    for (int t = 0; t < nt; ++t) {
      Grad_accum += Grad_loc[t];
      H_sum      += H_loc[t];
    }
    obj_old += ll_old;
    if (use_ll_init) ll_old = ll_init;
    obj_old = ll_old - 0.5 * arma::accu(M % M) / prior_var;
    current_ll = ll_old;
    last_ll_old = ll_old;
    last_lhat_old = obj_old;

    if (n_blocks == 0) break;
    // Full shared-Hessian Newton direction: summed gradient against summed Hessian.
    auto ts = std::chrono::steady_clock::now();
    arma::mat Dir;
    if (!arma::solve(Dir, H_sum, Grad_accum.t(), arma::solve_opts::fast)) break;
    Dir = Dir.t();
    solve_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();

    const double max_norm = arma::max(arma::sqrt(arma::sum(Dir % Dir, 1)));
    last_max_dir_norm = max_norm;
    if (max_norm < epsilon) break;

    // ---- TRUE TRUST REGION: rho ratio + warm-start delta (partial-obs) -----
    bool accepted = false;
    double try_delta = delta;
    for (int tr = 0; tr < max_ls; ++tr) {
      ++total_ls;
      const double scale = trust_region_scale_from_norm(max_norm, try_delta);
      arma::mat Step = arma::clamp(M + scale * Dir, -C_m, C_m) - M;
      const double step_norm = arma::max(arma::sqrt(arma::sum(Step % Step, 1)));
      last_try_delta = try_delta;
      last_step_norm = step_norm;

      const double pred = arma::accu(Grad_accum % Step)
                        - 0.5 * arma::accu((Step * H_sum) % Step);

      arma::mat M_c = M + Step;
      double obj_new = -0.5 * arma::accu(M_c % M_c) / prior_var;
      double ll_new = 0.0;

      if (blas_nt) blas_nt(1);
#ifdef _OPENMP
      if (n_threads > 0) omp_set_num_threads(n_threads);
#pragma omp parallel for schedule(static) reduction(+:ll_new,subset_seconds,ls_eta_seconds,ls_lhat_seconds)
#endif
      for (arma::sword block_id = 0; block_id < static_cast<arma::sword>(n_blocks); ++block_id) {
        const arma::uword s = static_cast<arma::uword>(block_id) * sb;
        const arma::uword e = std::min(s + sb, p);
        auto t0 = std::chrono::steady_clock::now();
        const arma::uvec idx = use_sord
          ? (sord.subvec(s, e - 1) - 1)
          : arma::regspace<arma::uvec>(s, e - 1);

        arma::mat Y_b      = Y.cols(idx);
        arma::mat Obs_b    = observed_mat.cols(idx);
        arma::mat Lambda_b = Lambda.rows(idx);
        arma::mat Beta_b   = Beta.rows(idx);
        auto t1 = std::chrono::steady_clock::now();
        subset_seconds += std::chrono::duration<double>(t1 - t0).count();

        t0 = std::chrono::steady_clock::now();
        arma::mat XB_b = X * Beta_b.t();
        arma::mat eta_c = XB_b + M_c * Lambda_b.t();
        t1 = std::chrono::steady_clock::now();
        ls_eta_seconds += std::chrono::duration<double>(t1 - t0).count();

        t0 = std::chrono::steady_clock::now();
        double ll_block = 0.0;
        for (arma::uword j = 0; j < e - s; ++j) {
          for (arma::uword i = 0; i < n; ++i) {
            const double zc = eta_c(i, j);
            ll_block += (Y_b(i, j) * zc - softplus(zc)) * Obs_b(i, j);
          }
        }
        ll_new += ll_block;
        t1 = std::chrono::steady_clock::now();
        ls_lhat_seconds += std::chrono::duration<double>(t1 - t0).count();
      }
      obj_new += ll_new;

      const double actual = obj_new - obj_old;
      const double rho    = (pred > 1e-14) ? (actual / pred) : -1.0;

      last_ll_new   = ll_new;
      last_lhat_new = obj_new;
      last_pred     = pred;
      last_actual   = actual;
      last_rho      = rho;
      if (rho > 0.1) {
        M = M_c;
        current_ll = ll_new;
        delta = (rho > 0.75) ? std::min(delta_max, try_delta * 2.0) : try_delta;
        accepted = true;
        accepted_step = step_norm;
        break;
      }
      try_delta *= (rho > 0.25) ? 0.5 : 0.25;
    }
    if (!accepted) {
      delta = std::max(1e-8, delta * 0.25);  // single moderate shrink, not cascade-collapse
      break;
    }
  }

  if (blas_nt && n_threads > 0) blas_nt(n_threads);
  auto total1 = std::chrono::steady_clock::now();
  const double total_seconds = std::chrono::duration<double>(total1 - total0).count();
  return List::create(
    _["M"] = M,
    _["lhat_time"] = old_lhat_seconds + ls_lhat_seconds,
    _["n_species"] = static_cast<int>(p),
    _["accepted"] = accepted_step > 0.0,
    _["step"] = accepted_step,
    _["ll"] = current_ll,
    _["ll_old"] = last_ll_old,
    _["ll_new"] = last_ll_new,
    _["lhat_old"] = last_lhat_old,
    _["lhat_new"] = last_lhat_new,
    _["pred"] = last_pred,
    _["actual"] = last_actual,
    _["rho"] = last_rho,
    _["try_delta"] = last_try_delta,
    _["max_dir_norm"] = last_max_dir_norm,
    _["step_norm"] = last_step_norm,
    _["ls_trials"] = total_ls,
    _["step_out"] = delta,
    _["timing"] = NumericVector::create(
      _["subset"] = subset_seconds,
      _["XB"] = xbeta_seconds,
      _["eta"] = eta_seconds,
      _["sigmoid"] = sigmoid_seconds,
      _["hessian"] = hessian_seconds,
      _["grad"] = grad_seconds,
      _["solve"] = solve_seconds,
      _["old_lhat"] = old_lhat_seconds,
      _["ls_eta"] = ls_eta_seconds,
      _["ls_lhat"] = ls_lhat_seconds,
      _["total"] = total_seconds
    )
  );
}

// Shared-Hessian MAP for M — masked / partially-observed case.
// Same structure as map_latent_factors_shared_hessian_fullobs;
// observed_mat (n x p, 1.0 = observed) gates likelihood contributions.
// [[Rcpp::export]]
List map_latent_factors_shared_hessian(
    const arma::mat& Y,
    const arma::mat& observed_mat,   // n x p, 1.0 = observed
    const arma::mat& X,
    const arma::mat& Lambda,
    const arma::mat& Beta,
    const arma::mat& M_init,
    double prior_var = 1.0,
    double epsilon   = 1e-3,
    int    n_threads = 1,
    int    m_block   = 1000,         // unused (kept for API compat)
    double step_init = 1.0)          // warm-start delta
{
  typedef void (*set_nt_t)(int);
  set_nt_t blas_nt = reinterpret_cast<set_nt_t>(
      dlsym(RTLD_DEFAULT, "openblas_set_num_threads"));
  if (!blas_nt)
    blas_nt = reinterpret_cast<set_nt_t>(
        dlsym(RTLD_DEFAULT, "openblas_set_num_threads_"));

  const arma::uword n   = Y.n_rows;
  const arma::uword p   = Y.n_cols;
  const arma::uword k   = M_init.n_cols;
  const double C_m      = 2.0 * std::sqrt(std::log(static_cast<double>(k) * n + 1.0));

  auto sp_fn = [](double x) -> double {
    if (x >  35.0) return x;
    if (x < -35.0) return std::exp(x);
    return std::log1p(std::exp(x));
  };

  if (blas_nt && n_threads > 0) blas_nt(n_threads);
  arma::mat B_x = X * Beta.t();
  arma::mat M   = M_init;

  double delta = std::max(1e-8, step_init);
  const double delta_max = std::max(delta, C_m);

  for (int it = 0; it < 1; ++it) {

    // PHASE 1: BLAS for matmuls, OMP for element-wise (sigmoid is not BLAS).
    if (blas_nt && n_threads > 0) blas_nt(n_threads);
    arma::mat eta = B_x + M * Lambda.t();   // n x p, BLAS

    if (blas_nt) blas_nt(1);
    const int nt = std::max(1, n_threads);
    std::vector<arma::vec> w_loc(nt, arma::vec(p, arma::fill::zeros));
    arma::mat resid(n, p);
#ifdef _OPENMP
    if (n_threads > 0) omp_set_num_threads(n_threads);
#pragma omp parallel for schedule(static)
#endif
    for (arma::sword i = 0; i < static_cast<arma::sword>(n); ++i) {
      int tid = 0;
#ifdef _OPENMP
      tid = omp_get_thread_num();
#endif
      for (arma::uword j = 0; j < p; ++j) {
        const double obs = observed_mat(i, j);
        const double fi  = 1.0 / (1.0 + std::exp(-eta(i, j)));
        w_loc[tid](j) += fi * (1.0 - fi) * obs;
        resid(i, j)    = (Y(i, j) - fi) * obs;
      }
    }
    arma::vec w_sum(p, arma::fill::zeros);
    for (int t = 0; t < nt; ++t) w_sum += w_loc[t];

    if (blas_nt && n_threads > 0) blas_nt(n_threads);
    arma::vec w_bar = arma::clamp(w_sum / static_cast<double>(n), 1e-8, arma::datum::inf);
    arma::mat LW    = Lambda.each_col() % arma::sqrt(w_bar);
    arma::mat H_sh  = LW.t() * LW + arma::eye(k, k) / prior_var;

    arma::mat Grad  = resid * Lambda - M / prior_var;   // BLAS
    arma::mat Dir_t;
    if (!arma::solve(Dir_t, H_sh, Grad.t(), arma::solve_opts::fast)) break;
    arma::mat Dir = Dir_t.t();

    double max_norm = arma::max(arma::sqrt(arma::sum(Dir % Dir, 1)));
    if (max_norm < epsilon) break;

    // ---- PHASE 2: TRUE TRUST REGION, partial-obs (rho ratio + warm-start) ---
    arma::mat SP_eta = eta; SP_eta.transform(sp_fn);
    const double obj_old = arma::accu((Y % eta - SP_eta) % observed_mat)
                         - 0.5 * arma::accu(M % M) / prior_var;

    bool accepted = false;
    double try_delta = delta;
    for (int tr = 0; tr < 8; ++tr) {
      const double scale = trust_region_scale_from_norm(max_norm, try_delta);
      const arma::mat Step = arma::clamp(M + scale * Dir, -C_m, C_m) - M;

      // Predicted reduction: g^T s - 0.5 s^T H_sh s  (summed over sites)
      const double pred = arma::accu(Grad % Step)
                        - 0.5 * arma::accu((Step * H_sh) % Step);

      const arma::mat M_c = M + Step;
      if (blas_nt && n_threads > 0) blas_nt(n_threads);
      arma::mat eta_c = B_x + M_c * Lambda.t();
      arma::mat SP_c = eta_c; SP_c.transform(sp_fn);
      const double obj_new = arma::accu((Y % eta_c - SP_c) % observed_mat)
                           - 0.5 * arma::accu(M_c % M_c) / prior_var;

      const double actual = obj_new - obj_old;
      const double rho    = (pred > 1e-14) ? (actual / pred) : -1.0;

      if (rho > 0.1) {
        M = M_c;
        delta = (rho > 0.75) ? std::min(delta_max, try_delta * 2.0) : try_delta;
        accepted = true;
        break;
      }
      try_delta *= (rho > 0.25) ? 0.5 : 0.25;
    }
    if (!accepted) {
      delta = std::max(1e-8, delta * 0.25);  // single moderate shrink, not cascade-collapse
      break;
    }
  }
  if (blas_nt && n_threads > 0) blas_nt(n_threads);
  return List::create(_["M"] = M, _["step_out"] = delta);
}

// [[Rcpp::export]]
arma::mat map_latent_factors_mvlogit(
    arma::mat Y, arma::mat observed_mat, arma::mat X,
    arma::mat Lambda, arma::mat Beta, arma::mat M_init,
    double prior_var=1, int max_it=200, double epsilon=0.001,
    int n_threads=1) {

  const int n = Y.n_rows;
  const int k = M_init.n_cols;
  arma::mat Eta_hat(n, k);

  // Force single-threaded BLAS inside OMP region to avoid nested threading.
  {
    typedef void (*set_nt_t)(int);
    set_nt_t fn = reinterpret_cast<set_nt_t>(dlsym(RTLD_DEFAULT, "openblas_set_num_threads"));
    if (fn) fn(1);
  }
#ifdef _OPENMP
  if (n_threads > 0) omp_set_num_threads(n_threads);
  omp_set_nested(0);
  omp_set_max_active_levels(1);
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < n; ++i) {
    // res_i declared inside loop — each thread has its own copy on the stack.
    arma::vec res_i = optimize_latent_factors_logit(
      Y.row(i).t(), observed_mat.row(i).t(), X.row(i).t(),
      Lambda, Beta, M_init.row(i).t(),
      prior_var, max_it, epsilon, n);
    for (int l = 0; l < k; ++l) Eta_hat(i, l) = res_i(l);
  }
  return Eta_hat;
}

// [[Rcpp::export]]
arma::mat map_latent_factors_mvlogit_fullobs(
    arma::mat Y, arma::mat X,
    arma::mat Lambda, arma::mat Beta, arma::mat M_init,
    double prior_var=1, int max_it=200, double epsilon=0.001,
    int n_threads=1) {

  const int n = Y.n_rows;
  const int k = M_init.n_cols;
  arma::mat Eta_hat(n, k);

  {
    typedef void (*set_nt_t)(int);
    set_nt_t fn = reinterpret_cast<set_nt_t>(dlsym(RTLD_DEFAULT, "openblas_set_num_threads"));
    if (fn) fn(1);
  }
#ifdef _OPENMP
  if (n_threads > 0) omp_set_num_threads(n_threads);
  omp_set_nested(0);
  omp_set_max_active_levels(1);
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < n; ++i) {
    arma::vec res_i = optimize_latent_factors_logit_fullobs(
      Y.row(i).t(), X.row(i).t(),
      Lambda, Beta, M_init.row(i).t(),
      prior_var, max_it, epsilon, n);
    for (int l = 0; l < k; ++l) Eta_hat(i, l) = res_i(l);
  }
  return Eta_hat;
}




// [[Rcpp::export]]
arma::mat compute_B_hessian(arma::mat Lambda_hat, arma::vec W_bar, double sigma=1.702){
  int p = Lambda_hat.n_rows;
  arma::mat B_uv(p, p);
  
  double sigma_sq = pow(sigma,2);
  
  for(int u=0; u<p; ++u){
    double sigma_sq_u = sigma_sq + 1/W_bar[u];
    for(int v=u; v<p; ++v){
      double sigma_sq_v = sigma_sq + 1/W_bar[v];
      
      B_uv(u,v) = sqrt(1+ (
        dot( pow(Lambda_hat.row(u),2), pow(Lambda_hat.row(v),2) )  + 
          pow( dot( Lambda_hat.row(u), Lambda_hat.row(v) ), 2 ) 
      ) /  (sigma_sq_v * dot(Lambda_hat.row(u), Lambda_hat.row(u))  + sigma_sq_u * dot( Lambda_hat.row(v), Lambda_hat.row(v))  ) 
      );
      B_uv(v, u) = B_uv(u,v);
    }
    B_uv(u,u) = sqrt(1+ (dot(Lambda_hat.row(u),Lambda_hat.row(u)))/(2*sigma_sq_u));
  }
  double sigma_sq_u = sigma_sq + 1/W_bar[p-1];
  B_uv(p-1,p-1) = sqrt(1+ (dot(Lambda_hat.row(p-1),Lambda_hat.row(p-1)))/(2*sigma_sq_u));
  return B_uv;
}


// [[Rcpp::export]]
List sample_Lambda_outer(arma::cube Lambda_samples, arma::mat Lambda_hat, double rho){
  
  int n_MC = Lambda_samples.n_rows;
  int k = Lambda_samples.n_cols;
  int p = Lambda_samples.n_slices;
  
  arma::mat Lambda(p, k);
  arma::mat Lambda_cc(p, k);
  arma::mat Lambda_outer(p, p);
  arma::mat Lambda_outer_cc(p, p);
  arma::cube Lambda_outer_samples (p, p, n_MC);
  arma::cube Lambda_outer_samples_cc(p, p, n_MC);
  
  for(int s=0; s<n_MC; ++s) {
    Lambda = arma::mat(Lambda_samples.row(s)).t();
    Lambda_cc = Lambda_hat + rho*(Lambda - Lambda_hat);
    Lambda_outer = Lambda * Lambda.t();
    Lambda_outer_cc = Lambda_cc * Lambda_cc.t();
    Lambda_outer_samples.slice(s) = Lambda_outer;
    Lambda_outer_samples_cc.slice(s) = Lambda_outer_cc;
  }
  
  return List::create(Named("Lambda_outer_samples") = Lambda_outer_samples,
                      Named("Lambda_outer_samples_cc") = Lambda_outer_samples_cc);
}

// Closed-form MAP for Gamma, solved independently per row (dimension r).
// Posterior for Gamma_r: precision = Xi Xi' / sigma^2_r + I_h / kappa^2_r
//                        mean      = Prec_r^{-1} (Xi bl_r) / sigma^2_r
// Xi_mat : h x p   Beta_Lambda : p x (q+k)   sigma_sq_vec,kappa_sq_vec : (q+k)
// [[Rcpp::export]]
arma::mat update_Gamma_cpp(
    const arma::mat& Beta_Lambda,
    const arma::mat& Xi_mat,
    const arma::vec& sigma_sq_vec,   // (q+k): per-dimension residual variance
    const arma::vec& kappa_sq_vec) { // (q+k): per-row Gamma prior variance

  const int qk = Beta_Lambda.n_cols;
  const int h  = Xi_mat.n_rows;
  arma::mat XiXit = Xi_mat * Xi_mat.t();   // h x h, shared across rows
  arma::mat Gamma(qk, h);

  for (int r = 0; r < qk; ++r) {
    const double sig2 = sigma_sq_vec(r);
    const double kap2 = kappa_sq_vec(r);
    arma::mat prec_r  = XiXit / sig2 + arma::eye(h, h) / kap2;
    arma::vec rhs     = Xi_mat * Beta_Lambda.col(r) / sig2;
    arma::vec gr      = arma::solve(prec_r, rhs, arma::solve_opts::fast);
    Gamma.row(r)      = gr.t();
  }
  return Gamma;   // (q+k) x h
}


// Closed-form InvGamma posterior MAP for per-dimension variances.
// MAP: sigma^2_r = clamp( (b + colSS_r/2) / (a + p/2 + 1), lb, ub )
// colSS_r = sum_j delta_{jr}^2  (summed over all p species for dimension r)
// Beta_Lambda : p x (q+k)   Gamma : (q+k) x h   Xi_mat : h x p
// [[Rcpp::export]]
arma::vec update_sigma_sq_cpp(
    const arma::mat& Beta_Lambda,
    const arma::mat& Gamma,
    const arma::mat& Xi_mat,
    double a, double b,
    double lb, double ub) {

  arma::mat Mu    = (Gamma * Xi_mat).t();              // p x (q+k)
  arma::mat Delta = Beta_Lambda - Mu;
  int p           = Beta_Lambda.n_rows;
  arma::vec SS    = arma::sum(Delta % Delta, 0).t();   // (q+k): colSums
  arma::vec sigma_sq = (b + SS / 2.0) / (a + p / 2.0 + 1.0);
  return arma::clamp(sigma_sq, lb, ub);
}


// Closed-form InvGamma posterior MAP for per-row Gamma variances.
// MAP: kappa^2_r = clamp( (d + SS_r/2) / (c + h/2 + 1), lb, ub ),  SS_r = ||Gamma_r||^2 + tr_V_r.
// With tr_V_r = tr Var(Gamma_r | -) (gamma_post_trace_cpp) this is the EM update;
// without it, the conditional mode, which uses the shrunken Gamma_r only.
// Gamma : (q+k) x h
// [[Rcpp::export]]
arma::vec update_kappa_sq_cpp(
    const arma::mat& Gamma,
    double c_hyp, double d_hyp,
    double lb, double ub,
    Rcpp::NumericVector tr_V = Rcpp::NumericVector::create()) {

  int h = Gamma.n_cols;
  arma::vec SS = arma::sum(Gamma % Gamma, 1);   // (q+k): rowSums
  if (tr_V.size()) SS += Rcpp::as<arma::vec>(tr_V);
  arma::vec kappa_sq = (d_hyp + SS / 2.0) / (c_hyp + h / 2.0 + 1.0);
  return arma::clamp(kappa_sq, lb, ub);
}


// MAP for kappa^2 under Half-Cauchy(0, hc_scale) prior on kappa.
// Solves (h+3)*u^2 - (SS-(h+1)*s^2)*u - s^2*SS = 0; positive root clamped to [lb,ub].
// [[Rcpp::export]]
arma::vec update_kappa_sq_halfcauchy_cpp(
    const arma::mat& Gamma, double hc_scale, double lb, double ub,
    Rcpp::NumericVector tr_V = Rcpp::NumericVector::create()) {
  int h = Gamma.n_cols;
  double s2 = hc_scale * hc_scale, d = 2.0 * (h + 3.0);
  arma::vec SS = arma::sum(Gamma % Gamma, 1);
  if (tr_V.size()) SS += Rcpp::as<arma::vec>(tr_V);
  arma::vec b  = SS - (h + 1.0) * s2;
  return arma::clamp((b + arma::sqrt(b % b + (2.0 * d * s2) * SS)) / d, lb, ub);
}


// tr Var(Gamma_r | B, sigma^2, kappa^2) over the first n_free coordinates:
//   V_r = (EXX / sigma_r^2 + diag(1/kappa_r^2 on n_free, mu_prec on the rest))^{-1},
// EXX = E(Xi Xi') (= Xi Xi' for fixed embeddings).
// [[Rcpp::export]]
arma::vec gamma_post_trace_cpp(const arma::mat& EXX, const arma::vec& sigma_sq,
                               const arma::vec& kappa_sq, int n_free, double mu_prec) {
  const arma::uword H = EXX.n_rows, d = sigma_sq.n_elem;
  arma::vec out(d), rid(H);
  for (arma::uword r = 0; r < d; ++r) {
    rid.fill(mu_prec); rid.head(n_free).fill(1.0 / kappa_sq(r));
    const arma::vec v = arma::diagvec(arma::inv_sympd(EXX / sigma_sq(r) + arma::diagmat(rid)));
    out(r) = arma::accu(v.head(n_free));
  }
  return out;
}


// Newton-Raphson MAP for all p species in FLAIR+.
// Structured prior: beta_lambda_j ~ N(mu_j, diag(sigma_sq_vec))  (shared across j)
// X_tilde : n x (q+k)   BL_init/Mu : p x (q+k)   sigma_sq_vec : (q+k)
// [[Rcpp::export]]
arma::mat map_beta_lambda_plus(
    const arma::mat& Y,
    const arma::mat& observed_mat,
    const arma::mat& X_tilde,
    const arma::mat& BL_init,
    const arma::mat& Mu,
    const arma::vec& sigma_sq_vec,   // (q+k): per-dimension, shared across species
    double epsilon,
    double C_clip) {

  int p = Y.n_cols;
  arma::mat BL = BL_init;
  // Precompute diagonal prior precision matrix (same for every species j)
  arma::mat prior_prec = arma::diagmat(1.0 / sigma_sq_vec);

  for (int j = 0; j < p; ++j) {
    arma::vec bl_j  = BL.row(j).t();
    arma::vec mu_j  = Mu.row(j).t();
    arma::vec y_j   = Y.col(j);
    arma::vec obs_j = observed_mat.col(j);

    for (int it = 0; it < 1; ++it) {
      arma::vec pv    = 1.0 / (1.0 + arma::exp(-(X_tilde * bl_j)));
      arma::vec wt    = pv % (1.0 - pv) % obs_j;
      arma::vec resid = (y_j - pv) % obs_j;

      arma::vec grad = X_tilde.t() * resid - (bl_j - mu_j) / sigma_sq_vec;

      // neg_H = X_tilde' diag(wt) X_tilde + diag(1/sigma^2)  (always pos-def)
      arma::mat Xw    = X_tilde.each_col() % arma::sqrt(wt);
      arma::mat neg_H = Xw.t() * Xw + prior_prec;

      arma::vec upd;
      if (!arma::solve(upd, neg_H, grad, arma::solve_opts::fast)) break;
      if (arma::norm(upd, 2) < epsilon) break;

      double obj_old = log_post_beta_lambda_conditional(
        y_j, obs_j, X_tilde, bl_j, mu_j, sigma_sq_vec);
      double step = 1.0;
      bool accepted = false;
      for (int ls = 0; ls < 25; ++ls) {
        arma::vec cand = arma::clamp(bl_j + step * upd, -C_clip, C_clip);
        double obj_new = log_post_beta_lambda_conditional(
          y_j, obs_j, X_tilde, cand, mu_j, sigma_sq_vec);
        if (obj_new >= obj_old) {
          bl_j = cand;
          accepted = true;
          break;
        }
        step *= 0.5;
      }
      if (!accepted) break;
    }
    BL.row(j) = bl_j.t();
  }
  return BL;
}

inline double block_ll_from_Z_cpp(
    const arma::mat& Y_blk,
    const arma::mat& obs_blk,
    const arma::mat& Z,
    bool full_obs) {

  double ll = 0.0;
  const arma::uword n = Z.n_rows;
  const arma::uword b = Z.n_cols;
  for (arma::uword j = 0; j < b; ++j) {
    for (arma::uword i = 0; i < n; ++i) {
      if (!full_obs && obs_blk(i, j) != 1.0) continue;
      const double zij = Z(i, j);
      ll += Y_blk(i, j) * zij - softplus(zij);
    }
  }
  return ll;
}

inline double block_prior_term_cpp(
    const arma::mat& BL_blk,
    const arma::mat& Mu_blk,
    const arma::vec& sigma_sq_vec) {   // (q+k): per-dimension, shared across species

  arma::mat Delta  = BL_blk - Mu_blk;    // b x d
  const int p_blk  = BL_blk.n_rows;
  arma::mat Delta2 = Delta % Delta;
  Delta2.each_row() /= sigma_sq_vec.t(); // divide each species row by per-dim variances
  return -0.5 * arma::accu(Delta2) -
    0.5 * p_blk * arma::sum(arma::log(sigma_sq_vec));
}

// Exact Bernoulli-logistic data term over all species.
// [[Rcpp::export]]
double compute_data_term_cpp(
    const arma::mat& Y,
    const arma::mat& observed_mat,
    bool full_obs,
    const arma::mat& X_tilde,
    const arma::mat& Beta_Lambda) {

  typedef void (*set_nt_t)(int);
  set_nt_t blas_nt = reinterpret_cast<set_nt_t>(
      dlsym(RTLD_DEFAULT, "openblas_set_num_threads"));
  if (!blas_nt)
    blas_nt = reinterpret_cast<set_nt_t>(
        dlsym(RTLD_DEFAULT, "openblas_set_num_threads_"));
  if (blas_nt) blas_nt(1);

  double ll = 0.0;
  const arma::uword n = Y.n_rows;
  const arma::uword p = Y.n_cols;
  const arma::uword block_size = 1024;
  const arma::uword n_blocks = (p + block_size - 1) / block_size;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:ll)
#endif
  for (arma::sword block_id = 0; block_id < static_cast<arma::sword>(n_blocks); ++block_id) {
    const arma::uword start = static_cast<arma::uword>(block_id) * block_size;
    const arma::uword end = std::min(start + block_size, p);
    arma::mat Z_blk = X_tilde * Beta_Lambda.rows(start, end - 1).t();

    for (arma::uword j = 0; j < end - start; ++j) {
      const arma::uword col = start + j;
      for (arma::uword i = 0; i < n; ++i) {
        if (!full_obs && observed_mat(i, col) != 1.0) continue;
        const double zij = Z_blk(i, j);
        ll += Y(i, col) * zij - softplus(zij);
      }
    }
  }
  return ll;
}

// Exact Beta_Lambda prior term given Mu = t(Gamma %*% Xi).
// sigma_sq_vec : (q+k) per-dimension (shared across all p species)
// [[Rcpp::export]]
double compute_beta_prior_term_cpp(
    const arma::mat& Beta_Lambda,   // p x (q+k)
    const arma::mat& Mu,            // p x (q+k)
    const arma::vec& sigma_sq_vec)  // (q+k): per-dimension variance
{
  arma::mat Delta  = Beta_Lambda - Mu;
  const int p      = Beta_Lambda.n_rows;
  arma::mat Delta2 = Delta % Delta;
  Delta2.each_row() /= sigma_sq_vec.t();
  return -0.5 * arma::accu(Delta2) -
    0.5 * p * arma::sum(arma::log(sigma_sq_vec));
}

// Exact full log-posterior for FLAIR+.
// sigma_sq_vec : (q+k) per-dimension   kappa_sq_vec : (q+k) per-row of Gamma
// [[Rcpp::export]]
double compute_log_post_plus_cpp(
    const arma::mat& Y,
    const arma::mat& observed_mat,
    bool full_obs,
    const arma::mat& X_tilde,
    const arma::mat& Beta_Lambda,
    const arma::mat& M,
    const arma::mat& Gamma,
    const arma::mat& Xi_mat,
    const arma::vec& sigma_sq_vec,   // (q+k): per-dimension residual variance
    double a,
    double b,
    const arma::vec& kappa_sq_vec,   // (q+k): per-row Gamma prior variance
    double c_hyp,
    double d_hyp) {

  const double ll    = compute_data_term_cpp(Y, observed_mat, full_obs, X_tilde, Beta_Lambda);
  const arma::mat Mu = (Gamma * Xi_mat).t();
  const double lp_bl = compute_beta_prior_term_cpp(Beta_Lambda, Mu, sigma_sq_vec);
  const double lp_M  = -0.5 * arma::accu(M % M);

  // Gamma prior: per-row kappa^2_r
  arma::vec gamma_ss = arma::sum(Gamma % Gamma, 1);   // (q+k): rowSums
  const double lp_G  = -0.5 * arma::dot(gamma_ss, 1.0 / kappa_sq_vec) -
    0.5 * static_cast<double>(Gamma.n_cols) * arma::accu(arma::log(kappa_sq_vec));

  // sigma^2_r InvGamma prior over (q+k) dimensions
  const double lp_sig = arma::accu(
    a * std::log(b) - std::lgamma(a) -
      (a + 1.0) * arma::log(sigma_sq_vec) - b / sigma_sq_vec
  );
  // kappa^2_r InvGamma prior over (q+k) rows of Gamma
  const double lp_kap = arma::accu(
    c_hyp * std::log(d_hyp) - std::lgamma(c_hyp) -
      (c_hyp + 1.0) * arma::log(kappa_sq_vec) - d_hyp / kappa_sq_vec
  );
  return ll + lp_bl + lp_M + lp_G + lp_sig + lp_kap;
}

// Shared-Hessian batched FLAIR+ update over species blocks.
// Returns updated Beta_Lambda plus exact block deltas for ll and beta prior.
// [[Rcpp::export]]
List map_beta_lambda_plus_blocks_cpp(
    const arma::mat& Y,
    const arma::mat& observed_mat,
    bool   full_obs,
    const arma::mat& X_tilde,
    const arma::mat& BL_init,
    const arma::mat& Mu,
    const arma::vec& sigma_sq_vec,
    const arma::uvec& ord,
    int    batch_size,
    double C_clip,
    double step_cap,
    int    n_threads) {

  const arma::uword p = Y.n_cols;
  const arma::uword d = X_tilde.n_cols;
  const arma::uword n_batches = (p + batch_size - 1) / batch_size;
  arma::mat BL_new = BL_init;
  arma::vec delta_ll(n_batches, arma::fill::zeros);
  arma::vec delta_prior(n_batches, arma::fill::zeros);

#ifdef _OPENMP
  if (n_threads > 0) omp_set_num_threads(n_threads);
  omp_set_nested(0);
  omp_set_max_active_levels(1);
#endif
  // openblas_set_num_threads(1): force OpenBLAS to single-thread mode so it
  // does not spawn pthreads inside our OMP parallel region (which triggers the
  // "Detect OpenMP Loop" warning and can deadlock pthread builds).
  // dlsym is used so the code links cleanly with any BLAS (MKL, ATLAS, etc.).
  {
    typedef void (*set_nt_t)(int);
    set_nt_t fn = reinterpret_cast<set_nt_t>(dlsym(RTLD_DEFAULT, "openblas_set_num_threads"));
    if (fn) fn(1);
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (arma::sword batch_id = 0; batch_id < static_cast<arma::sword>(n_batches); ++batch_id) {
    const arma::uword start = static_cast<arma::uword>(batch_id) * batch_size;
    const arma::uword end = std::min(start + static_cast<arma::uword>(batch_size), p);
    const arma::uvec idx = ord.subvec(start, end - 1) - 1;

    arma::mat Y_blk  = Y.cols(idx);
    arma::mat BL_blk = BL_init.rows(idx);
    arma::mat Mu_blk = Mu.rows(idx);
    // sigma_sq_vec is (q+k): same for every species — no per-batch subsetting needed
    arma::mat obs_blk;
    if (!full_obs)
      obs_blk = observed_mat.cols(idx);

    arma::mat eta = X_tilde * BL_blk.t();
    arma::mat pv  = 1.0 / (1.0 + arma::exp(-eta));
    arma::mat wt;
    arma::mat resid;
    if (full_obs) {
      wt    = pv % (1.0 - pv);
      resid = Y_blk - pv;
    } else {
      wt    = (pv % (1.0 - pv)) % obs_blk;
      resid = (Y_blk - pv) % obs_blk;
    }

    arma::vec w_bar    = arma::clamp(arma::mean(wt, 1), 1e-8, arma::datum::inf);
    arma::mat Xw       = X_tilde.each_col() % arma::sqrt(w_bar);
    arma::mat H_shared = Xw.t() * Xw + arma::diagmat(1.0 / sigma_sq_vec);
    arma::mat Delta    = BL_blk - Mu_blk;
    Delta.each_row()  /= sigma_sq_vec.t();   // per-dimension division across all species in block
    arma::mat grad_t   = X_tilde.t() * resid - Delta.t();

    arma::mat dir_t;
    bool solved = arma::solve(dir_t, H_shared, grad_t, arma::solve_opts::fast);
    if (!solved) {
      BL_new.rows(idx) = BL_blk;
      continue;
    }
    arma::mat dir_blk = dir_t.t();
    arma::vec row_norm = arma::sqrt(arma::sum(dir_blk % dir_blk, 1));
    arma::vec scale = arma::clamp(step_cap / arma::clamp(row_norm, 1e-8, arma::datum::inf), 0.0, 1.0);
    dir_blk.each_col() %= scale;

    const double ll_old    = block_ll_from_Z_cpp(Y_blk, obs_blk, eta, full_obs);
    const double prior_old = block_prior_term_cpp(BL_blk, Mu_blk, sigma_sq_vec);
    const double obj_old   = ll_old + prior_old;

    double step = 1.0;
    bool accepted = false;
    arma::mat cand   = BL_blk;
    arma::mat Z_cand = eta;
    double ll_new    = ll_old;
    double prior_new = prior_old;
    for (int ls = 0; ls < 20; ++ls) {
      cand    = arma::clamp(BL_blk + step * dir_blk, -C_clip, C_clip);
      Z_cand  = X_tilde * cand.t();
      ll_new  = block_ll_from_Z_cpp(Y_blk, obs_blk, Z_cand, full_obs);
      prior_new = block_prior_term_cpp(cand, Mu_blk, sigma_sq_vec);
      if (ll_new + prior_new >= obj_old) {
        accepted = true;
        break;
      }
      step *= 0.5;
    }

    if (accepted) {
      BL_new.rows(idx) = cand;
      delta_ll(batch_id) = ll_new - ll_old;
      delta_prior(batch_id) = prior_new - prior_old;
    } else {
      BL_new.rows(idx) = BL_blk;
    }
  }

  return List::create(
    _["Beta_Lambda"] = BL_new,
    _["delta_ll"] = delta_ll,
    _["delta_prior"] = delta_prior
  );
}


// Linear congruential traversal of ord_base (1-based species indices).
//
// Visits every element exactly once via:
//   result[i] = ord_base[(start + i * stride) % n]
// with gcd(stride, n) == 1 (guaranteed by the search below).
//
// Why LCS instead of a full shuffle:
//   - stride ≈ n/2 means consecutive elements within each batch are spaced
//     ~n/2 apart in prevalence rank, so every batch mixes common and rare
//     species (fast mixing).
//   - Ergodic: different (start, stride) pairs across outer iterations ensure
//     every species visits every batch position over multiple calls.
//   - O(n) traversal, O(log n) per GCD check, typically 1-3 retries to find
//     a coprime stride — far cheaper than the O(n) search in the old R code.
//   - All arithmetic in uint64_t: no overflow for n up to ~1.8 × 10^19.
//
// When randomize == false, ord_base is returned unchanged (iter 1 keeps
// the prevalence-descending order set by the caller).
// [[Rcpp::export]]
arma::uvec build_batch_order_cpp(const arma::uvec& ord_base, bool randomize) {
  if (!randomize) return ord_base;

  const uint64_t n = static_cast<uint64_t>(ord_base.n_elem);
  if (n <= 1) return ord_base;

  // Euclidean GCD in 64-bit.
  auto gcd64 = [](uint64_t a, uint64_t b) -> uint64_t {
    while (b) { uint64_t t = b; b = a % b; a = t; }
    return a;
  };

  GetRNGstate();
  // Random start in [0, n).
  const uint64_t start = static_cast<uint64_t>(R::unif_rand() * static_cast<double>(n)) % n;
  // Candidate stride centred near n/2 for maximum within-batch mixing.
  // Add random jitter in [-n/8, +n/8] so successive iterations differ.
  const uint64_t centre = n / 2;
  const uint64_t jitter_range = std::max(uint64_t(1), n / 8);
  uint64_t stride = centre + static_cast<uint64_t>(R::unif_rand() * static_cast<double>(jitter_range));
  if (stride < 1) stride = 1;
  // Walk forward until coprime (expected 2-3 steps for typical n).
  while (gcd64(stride, n) != 1) { ++stride; if (stride >= n) stride = 1; }
  PutRNGstate();

  arma::uvec result(ord_base.n_elem);
  uint64_t pos = start;
  for (uint64_t i = 0; i < n; ++i) {
    result[static_cast<arma::uword>(i)] = ord_base[static_cast<arma::uword>(pos)];
    pos = (pos + stride) % n;
  }
  return result;
}


// Regress M_train ~ X_train (OLS), predict M for test sites from X_test.
// Uses only training data, no test outcomes involved.
// [[Rcpp::export]]
arma::mat impute_M_test_cpp(const arma::mat& X_train,
                            const arma::mat& M_train,
                            const arma::mat& X_test) {
  arma::mat B = arma::solve(X_train.t() * X_train, X_train.t() * M_train);
  return X_test * B;
}

// Frobenius norm of the M-step gradient at the current M.
// Useful for diagnosing the Lin et al. (2008) gradient-norm Δ₀ suggestion.
// Obs: pass a 0×0 matrix for full-observation case.
// [[Rcpp::export]]
double m_grad_norm_cpp(const arma::mat& Y,
                       const arma::mat& Obs,
                       const arma::mat& X,
                       const arma::mat& Lambda,
                       const arma::mat& Beta,
                       const arma::mat& M,
                       double prior_var) {
  arma::mat r = X * Beta.t() + M * Lambda.t();   // n × p; reused as resid
  for (arma::uword i = 0; i < r.n_elem; ++i)
    r(i) = Y(i) - 1.0 / (1.0 + std::exp(-r(i)));
  if (Obs.n_elem > 0) r %= Obs;
  const arma::mat G = r * Lambda - M / prior_var; // n × k
  return std::sqrt(arma::accu(G % G));
}

// [[Rcpp::export]]
arma::cube correct_Beta_samples(arma::cube Beta_samples, arma::mat Beta_hat, double rho){
  
  int n_MC = Beta_samples.n_rows;
  int q = Beta_samples.n_cols;
  int p = Beta_samples.n_slices;
  
  arma::mat Beta(p, q);
  arma::mat Beta_cc(p, q);
  arma::cube Beta_samples_cc(p, q, n_MC);
  
  for(int s=0; s<n_MC; ++s) {
    Beta = arma::mat(Beta_samples.row(s)).t();
    Beta_cc = Beta_hat + rho*(Beta - Beta_hat);
    Beta_samples_cc.slice(s) = Beta_cc;
  }
  
  return Beta_samples_cc;
}


// Per-species conditional posterior covariance diagonal for
// b̃_j = (beta_j, lambda_j) in the FLAIR+ logistic model.
//
// Accepts precomputed mu_mat (n x p sigmoid probabilities) so the caller can
// reuse the same matrix already computed for the M-posterior loop.
//
// For each j:
//   w_j  = observed[:,j] * mu_mat[:,j] * (1 - mu_mat[:,j])
//   H_j  = X_tilde^T diag(w_j) X_tilde + diag(sigma_sq_inv)
//   diag(H_j^{-1})_r = ||column r of chol(H_j)^{-1}||^2
//
// Returns: V_diag  (p x d)
// [[Rcpp::export]]
arma::mat compute_V_diag_all_cpp(
    const arma::mat& mu_mat,        // n x p: precomputed plogis(X_tilde %*% t(BL))
    const arma::mat& observed,      // n x p: observation mask (0/1, double)
    const arma::mat& X_tilde,       // n x d: [X, M]
    const arma::vec& sigma_sq_inv   // d: 1/sigma_sq
) {
  const int p = static_cast<int>(mu_mat.n_cols);
  const int d = static_cast<int>(X_tilde.n_cols);
  arma::mat V_diag(p, d);
  const arma::mat prec_prior = arma::diagmat(sigma_sq_inv);  // d x d, reused each j

  for (int j = 0; j < p; ++j) {
    const arma::vec mu_j = mu_mat.col(j);
    arma::vec w_j  = observed.col(j) % mu_j % (1.0 - mu_j);
    arma::mat H_j  = X_tilde.t() * arma::diagmat(w_j) * X_tilde + prec_prior;
    // diag(H^{-1})_r = ||col r of L^{-1}||^2  where H = L L^T (Cholesky)
    arma::mat L;
    if (arma::chol(L, H_j, "lower")) {
      arma::mat L_inv = arma::inv(arma::trimatl(L));
      V_diag.row(j)   = arma::sum(L_inv % L_inv, 0);   // column sums of squares
    } else {
      V_diag.row(j) = arma::diagvec(arma::inv_sympd(H_j)).t();
    }
  }
  return V_diag;
}

// Per-species observed information A_j = X' diag(w_j) X via single-precision
// BLAS syrk: the n x d^2 product dominates the E-step at large p, and
// float + syrk is ~4x a double gemm. A_j only enters prior-regularized
// inverses, where float accumulation error (~1e-5 relative) is negligible.
extern "C" void ssyrk_(const char*, const char*, const int*, const int*,
                       const float*, const float*, const int*,
                       const float*, float*, const int*);
static arma::mat trait_obs_info_f(const arma::fmat& Xf, const arma::mat& X_tilde,
                                  const arma::rowvec& bl, const arma::mat& observed,
                                  bool full_obs, int j) {
  arma::vec mu = 1.0 / (1.0 + arma::exp(-(X_tilde * bl.t())));
  arma::vec w  = mu % (1.0 - mu);
  if (!full_obs) w %= observed.col(j);
  arma::fmat Xw = Xf.each_col() % arma::conv_to<arma::fvec>::from(arma::sqrt(w));
  const int n = Xw.n_rows, d = Xw.n_cols;
  arma::fmat A(d, d);
  const float one = 1.0f, zero = 0.0f;
  ssyrk_("U", "T", &d, &n, &one, Xw.memptr(), &n, &zero, A.memptr(), &d);
  return arma::conv_to<arma::mat>::from(arma::symmatu(A));
}

// Same as compute_V_diag_all_cpp but computes mu_j per species (no n x p matrix).
// [[Rcpp::export]]
arma::mat compute_V_diag_from_BL_cpp(
    const arma::mat& X_tilde,
    const arma::mat& Beta_Lambda,
    const arma::mat& observed,
    bool full_obs,
    const arma::vec& sigma_sq_inv,
    int n_threads = 1
) {
  const int p = static_cast<int>(Beta_Lambda.n_rows);
  const int d = static_cast<int>(X_tilde.n_cols);
  arma::mat V_diag(p, d);
  const arma::mat prec_prior = arma::diagmat(sigma_sq_inv);
  const arma::fmat Xf = arma::conv_to<arma::fmat>::from(X_tilde);
  #pragma omp parallel for num_threads(n_threads) schedule(static)
  for (int j = 0; j < p; ++j) {
    arma::mat H_j = trait_obs_info_f(Xf, X_tilde, Beta_Lambda.row(j), observed, full_obs, j) + prec_prior;
    V_diag.row(j) = arma::diagvec(arma::inv_sympd(H_j)).t();
  }
  return V_diag;
}

// E-step of the trait layer with b~_j and xi_j integrated out jointly, block-Jacobi
// in species (Xi prior precision kept through its diagonal q_diag_j):
//   V_j = (A_j + S)^{-1},  W_j = S - S V_j S,  P_j = q_diag_j I_H + Gamma' W_j Gamma,
//   Var(xi_j) = P_j^{-1},  Var(b~_j - Gamma xi_j) = V_j + (I - V_j S) Gamma P_j^{-1} Gamma' (I - S V_j).
// Returns V_diag (p x d), S_xi = sum_j Var(xi_j) (H x H), U_diag = sum_j of the
// diagonal of the second term (d), the embedding part of E||b~_j - Gamma xi_j||^2, and
// C = sum_j Cov(b~_j, xi_j) = sum_j V_j S Gamma P_j^{-1} (d x H).
// [[Rcpp::export]]
Rcpp::List trait_estep_cpp(
    const arma::mat& X_tilde,
    const arma::mat& Beta_Lambda,
    const arma::mat& observed,
    bool full_obs,
    const arma::vec& sigma_sq_inv,
    const arma::mat& Gamma,
    const arma::vec& q_diag,
    int n_threads = 1
) {
  const int p = static_cast<int>(Beta_Lambda.n_rows);
  const arma::uword d = X_tilde.n_cols, H = Gamma.n_cols;
  const arma::mat S = arma::diagmat(sigma_sq_inv);
  const arma::mat SG = S * Gamma;
  const arma::fmat Xf = arma::conv_to<arma::fmat>::from(X_tilde);
  arma::mat V_diag(p, d), S_xi(H, H, arma::fill::zeros);
  arma::vec U_diag(d, arma::fill::zeros);
  arma::mat C(d, H, arma::fill::zeros);
  const arma::mat I_d = arma::eye(d, d);
  #pragma omp parallel num_threads(n_threads)
  {
    arma::mat S_loc(H, H, arma::fill::zeros);
    arma::vec U_loc(d, arma::fill::zeros);
    arma::mat C_loc(d, H, arma::fill::zeros);
    #pragma omp for schedule(static)
    for (int j = 0; j < p; ++j) {
      const arma::mat V = arma::inv_sympd(
        trait_obs_info_f(Xf, X_tilde, Beta_Lambda.row(j), observed, full_obs, j) + S);
      const arma::mat WG = SG - S * V * SG;
      const arma::mat GWG = Gamma.t() * WG;
      const arma::mat Pinv = arma::inv_sympd(q_diag(j) * arma::eye(H, H) + 0.5 * (GWG + GWG.t()));
      const arma::mat IVS = I_d - V * S;
      V_diag.row(j) = arma::diagvec(V).t();
      S_loc += Pinv;
      U_loc += arma::sum((IVS * Gamma * Pinv * Gamma.t()) % IVS, 1);
      C_loc += V * SG * Pinv;
    }
    #pragma omp critical
    { S_xi += S_loc; U_diag += U_loc; C += C_loc; }
  }
  return Rcpp::List::create(Rcpp::Named("V_diag") = V_diag, Rcpp::Named("S_xi") = S_xi,
                            Rcpp::Named("U_diag") = U_diag, Rcpp::Named("C") = C);
}
