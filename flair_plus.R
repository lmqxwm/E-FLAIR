################################################################################
# flair_plus.R — FLAIR++ application fitter (minibatch, trust-region M updates)
#
#   * fits a standard backbone subset first, then expands to all species
#   * trainable taxonomy embedding Xi (eigenbasis-decoupled PCG update)
#   * EM sigma^2 update with the per-species posterior variance V_diag
#   * closed-form (tau_xi, xi_gamma) MAP, ridge Gamma, community mean mu
################################################################################

library(parallel)
library(parallelly)
library(MASS)
library(matrixStats)
library(rsvd)
library(Rcpp)

FLAIR_CODE_DIR <- Sys.getenv("FLAIR_CODE_DIR",
  unset = tryCatch(dirname(sys.frame(1)$ofile), error = function(e) getwd()))
sourceCpp(file.path(FLAIR_CODE_DIR, "helper_functions_trustregion.cpp"))
sourceCpp(file.path(FLAIR_CODE_DIR, "flair_plus_minibatch_tr_pretrain.cpp"))
sourceCpp(file.path(FLAIR_CODE_DIR, "flair_tuning_helpers.cpp"))

.xi_cg_fn <- function(...) update_xi_taxonomy_cg_fast_cpp(...)
.gamma_threads <- max(1L, suppressWarnings(as.integer(
  Sys.getenv("SLURM_CPUS_PER_TASK", unset = parallelly::availableCores()))))
.gamma_fn <- function(BL, Xi, s2, k2) update_Gamma_fast_cpp(BL, Xi, s2, k2, .gamma_threads)

safe_logit <- function(p, eps = 1e-6) {
  p <- pmax(pmin(p, 1 - eps), eps)
  log(p / (1 - p))
}


make_taxonomy_path_codes <- function(taxonomy, species,
                                     tax_levels = c("kingdom", "phylum", "class", "order",
                                                    "family", "subfamily", "tribe", "genus")) {
  tax <- taxonomy[match(species, taxonomy$OTU), tax_levels, drop = FALSE]
  paths <- matrix("", nrow(tax), length(tax_levels))
  acc <- rep("root", nrow(tax))
  for (j in seq_along(tax_levels)) {
    val <- as.character(tax[[tax_levels[j]]])
    val[is.na(val) | !nzchar(val)] <- "missing"
    acc <- paste(acc, val, sep = "/")
    paths[, j] <- acc
  }
  codes <- apply(paths, 2L, function(x) as.integer(factor(x)))
  storage.mode(codes) <- "integer"
  codes
}

# Xi_t / Phi_t are optional pre-transposed (p×H) forms of Xi_mat / Phi_mat.
# Passing them avoids a large matrix copy when the caller already holds the
# transposed layout (e.g. cached Phi_t_all, or Xi_t returned from CG).
update_xi_step <- function(Beta_Lambda, Gamma, sigma_sq_vec, Xi_mat, Phi_mat,
                           taxonomy_codes, tau_vec, xi_gamma,
                           xi_maxit, xi_tol, cores,
                           Xi_t = NULL, Phi_t = NULL) {
  if (is.null(Xi_t))  Xi_t  <- t(Xi_mat)
  if (is.null(Phi_t)) Phi_t <- t(Phi_mat)
  res <- .xi_cg_fn(
    Beta_Lambda, Gamma, sigma_sq_vec,
    Xi_t, Phi_t, taxonomy_codes,
    tau_vec, xi_gamma, xi_maxit, xi_tol, cores
  )
  Xi_new <- t(res$Xi)
  dimnames(Xi_new) <- dimnames(Xi_mat)
  list(Xi_mat = Xi_new, Xi_t = res$Xi,   # Xi_t reused by caller; avoids one t()
       objective = res$objective, iter = res$iter,
       rel_resid = if (!is.null(res$rel_resid)) res$rel_resid else NA_real_,
       converged = if (!is.null(res$converged)) isTRUE(res$converged) else NA)
}

xi_prior_term_cpp <- function(Xi_mat, Phi_mat, taxonomy_codes,
                              tau_vec, xi_gamma,
                              Xi_t = NULL, Phi_t = NULL) {
  if (is.null(Xi_t))  Xi_t  <- t(Xi_mat)
  if (is.null(Phi_t)) Phi_t <- t(Phi_mat)
  xi_prior_term_taxonomy_cpp(Xi_t, Phi_t, taxonomy_codes,
                             tau_vec, xi_gamma)
}

# Spectral quadrature of L_norm: sum_i f(lambda_i) ~= sum_k w_k f(theta_k).
# Exact eigenvalues for small p; stochastic Lanczos quadrature with Rademacher
# probes otherwise (the p x p matrix is never formed).
xi_laplacian_spectrum <- function(codes, n_probe = 30L, n_lanczos = 50L,
                                  exact_max_p = 2000L) {
  p <- nrow(codes)
  if (p <= exact_max_p)
    return(list(theta = eigen(taxonomy_signed_laplacian_norm_prod_cpp(diag(p), codes),
                              symmetric = TRUE, only.values = TRUE)$values,
                w = rep(1, p)))
  m <- min(n_lanczos, p)
  V <- matrix(sample(c(-1, 1), p * n_probe, replace = TRUE), p, n_probe) / sqrt(p)
  V_prev <- matrix(0, p, n_probe)
  a <- b <- matrix(0, m, n_probe)
  for (s in seq_len(m)) {
    W <- taxonomy_signed_laplacian_norm_prod_cpp(V, codes)
    a[s, ] <- colSums(W * V)
    W <- W - sweep(V, 2, a[s, ], `*`) -
      if (s > 1) sweep(V_prev, 2, b[s - 1, ], `*`) else 0
    b[s, ] <- sqrt(colSums(W^2))
    V_prev <- V
    V <- sweep(W, 2, pmax(b[s, ], 1e-300), `/`)
  }
  theta <- w <- NULL
  for (c in seq_len(n_probe)) {
    mc <- which(b[, c] < 1e-10)[1]                 # Lanczos breakdown: exact subspace
    mc <- if (is.na(mc)) m else mc
    Tm <- diag(a[seq_len(mc), c], mc)
    if (mc > 1) Tm[cbind(1:(mc - 1), 2:mc)] <- Tm[cbind(2:mc, 1:(mc - 1))] <- b[seq_len(mc - 1), c]
    e <- eigen(Tm, symmetric = TRUE)
    theta <- c(theta, e$values)
    w <- c(w, p * e$vectors[1, ]^2 / n_probe)
  }
  list(theta = theta, w = w)
}

# (H/2) log det(tau I + gamma L_norm), the normaliser of the Xi prior
# N(Phi, {2(tau I + gamma L_norm)}^{-1}) per embedding column, up to a constant.
# With a = eigenvalues of Gamma' Sigma_b^{-1} Gamma, also subtract
# (1/2) log det{2(tau I + gamma L_norm) (x) I_H + I_p (x) Gamma' Sigma_b^{-1} Gamma},
# which integrates Xi out; without it the joint mode has tau, gamma -> infinity.
xi_logdet_term <- function(tau_xi, xi_gamma, spec, H, a = NULL) {
  q_ev <- tau_xi + xi_gamma * pmax(spec$theta, 0)
  out  <- 0.5 * H * sum(spec$w * log(q_ev))
  if (!is.null(a)) out <- out - 0.5 * sum(spec$w * rowSums(log(outer(2 * q_ev, a, `+`))))
  out
}

xi_post_eig <- function(Gamma, sigma_sq_vec)
  pmax(eigen(crossprod(Gamma, Gamma / as.numeric(sigma_sq_vec)),
             symmetric = TRUE, only.values = TRUE)$values, 0)

# EM update of sigma_b^2 with b~_j integrated out: E||b_r - Gamma_r xi||^2 adds
# Var(b~_jr) (V_diag) and the Gamma uncertainty tr(V_Gamma_r Xi Xi').
update_sigma_xi_em <- function(Beta_Lambda, Gamma, Xi_mat, V_diag, sigma_sq_vec,
                               kappa_sq_vec, a, b, lb = 0.01, ub = 100,
                               mu_prior_var = NULL, Mu = NULL, S_V = NULL) {
  # S_V = sum_j Var(xi_j) (H x H) from the trait E-step: with a trained embedding
  # the squared residual also carries Gamma_r S_V Gamma_r', and E[Xi Xi'] = Xi Xi' + S_V
  XX   <- tcrossprod(Xi_mat) + if (is.null(S_V)) 0 else S_V
  sig  <- as.numeric(sigma_sq_vec)
  H    <- nrow(XX)
  rdg  <- function(kap) if (is.null(mu_prior_var)) rep(1 / kap, H) else
    c(rep(1 / kap, H - 1L), 1 / mu_prior_var)
  tr_G <- vapply(seq_along(sig), function(r)
    sum(diag(solve(XX / sig[r] + diag(rdg(kappa_sq_vec[r])), XX))), 0)
  if (is.null(Mu)) Mu <- t(Gamma %*% Xi_mat)   # callers in the fit loop pass the current Mu
  SS <- colSums((Beta_Lambda - Mu)^2) + colSums(V_diag) + tr_G +
    if (is.null(S_V)) 0 else rowSums((Gamma %*% S_V) * Gamma)
  pmin(pmax((b + SS / 2) / (a + nrow(Beta_Lambda) / 2 + 1), lb), ub)
}

# Community mean mu: with trait_intercept the last embedding coordinate is the
# constant 1, so the last column of Gamma is mu. That column has the prior
# N(0, mu_prior_var) instead of the ridge N(0, kappa_r^2), and the constant
# coordinate is excluded from the Xi update, its E-step, the log-determinant
# and the kappa update. The returned functions shadow the global ones inside
# the wrapper.
trait_mean_ops <- function(mu_prior_var) {
  g <- list(kcpp = update_kappa_sq_cpp, khc = update_kappa_sq_halfcauchy_cpp,
            xstep = update_xi_step, hyp = update_xi_hyperparams_R, peig = xi_post_eig,
            ldet = xi_logdet_term, sig = update_sigma_xi_em)
  ridge <- function(H, kap) c(rep(1 / kap, H - 1L), 1 / mu_prior_var)
  ops <- list(
    update_Gamma_cpp = function(Beta_Lambda, Xi_mat, sigma_sq_vec, kappa_sq_vec) {
      H <- nrow(Xi_mat); sig <- as.numeric(sigma_sq_vec); XX <- tcrossprod(Xi_mat)
      XB <- Xi_mat %*% Beta_Lambda   # one GEMM instead of d matvec passes over Xi
      t(vapply(seq_len(ncol(Beta_Lambda)), function(r)
        solve(XX / sig[r] + diag(ridge(H, kappa_sq_vec[r])), XB[, r] / sig[r]), numeric(H)))
    },
    update_sigma_xi_em = function(...) g$sig(..., mu_prior_var = mu_prior_var),
    update_kappa_sq_cpp = function(Gamma, c_hyp, d_hyp, lb, ub) g$kcpp(Gamma[, -ncol(Gamma), drop = FALSE], c_hyp, d_hyp, lb, ub),
    update_kappa_sq_halfcauchy_cpp = function(Gamma, hc_scale, lb, ub) g$khc(Gamma[, -ncol(Gamma), drop = FALSE], hc_scale, lb, ub),
    update_xi_step = function(Beta_Lambda, Gamma, sigma_sq_vec, Xi_mat, Phi_mat, taxonomy_codes, tau_vec, xi_gamma,
                              xi_maxit, xi_tol, cores, Xi_t = NULL, Phi_t = NULL) {
      H <- nrow(Xi_mat)
      r <- g$xstep(Beta_Lambda - matrix(Gamma[, H], nrow(Beta_Lambda), ncol(Beta_Lambda), byrow = TRUE),
                   Gamma[, -H, drop = FALSE], sigma_sq_vec, Xi_mat[-H, , drop = FALSE], Phi_mat[-H, , drop = FALSE],
                   taxonomy_codes, tau_vec[-H], xi_gamma, xi_maxit, xi_tol, cores)
      r$Xi_mat <- rbind(r$Xi_mat, 1); r$Xi_t <- cbind(r$Xi_t, 1); r
    },
    update_xi_hyperparams_R = function(Xi_mat, Phi_mat, taxonomy_codes, spec, xi_e, xi_f, xi_g, xi_h,
                                       tau_xi, xi_gamma, hyper_xi_prior = "ig", xi_s = 1.0, xi_s_gamma = xi_s,
                                       lb = 0.01, ub = 100, Xi_t = NULL, Phi_t = NULL, a = NULL) {
      H <- nrow(Xi_mat)
      g$hyp(Xi_mat[-H, , drop = FALSE], Phi_mat[-H, , drop = FALSE], taxonomy_codes, spec, xi_e, xi_f, xi_g, xi_h,
            tau_xi, xi_gamma, hyper_xi_prior, xi_s, xi_s_gamma, lb, ub, a = a)
    },
    xi_post_eig = function(Gamma, sigma_sq_vec) g$peig(Gamma[, -ncol(Gamma), drop = FALSE], sigma_sq_vec),
    xi_logdet_term = function(tau_xi, xi_gamma, spec, H, a = NULL) g$ldet(tau_xi, xi_gamma, spec, H - 1L, a))
  ops
}

# EM update of (tau_xi, xi_gamma): the squared deviations are replaced by their
# posterior expectation, adding S_trace = sum_j tr Var(xi_j) from the trait E-step
# (diag L_norm = 1, so it enters both terms), with the prior log-determinant only.
# Xi_t, Phi_t: p x H without the constant (community-mean) column.
update_xi_hyperparams_em <- function(Xi_t, Phi_t, taxonomy_codes, spec, S_trace,
                                     xi_e, xi_f, xi_g, xi_h, tau_xi, xi_gamma,
                                     hyper_xi_prior = "ig", xi_s = 1.0, xi_s_gamma = xi_s,
                                     lb = 0.01, ub = 100) {
  D <- Xi_t - Phi_t
  S_xi    <- sum(D^2) + S_trace
  S_gamma <- sum(D * taxonomy_signed_laplacian_norm_prod_cpp(D, taxonomy_codes)) + S_trace
  neg_obj <- function(u) {
    tg <- exp(u)
    tg[1] * S_xi + tg[2] * S_gamma - xi_logdet_term(tg[1], tg[2], spec, ncol(D)) -
      compute_xi_hyperparam_prior(tg[1], tg[2], hyper_xi_prior, xi_e, xi_f, xi_g, xi_h, xi_s, xi_s_gamma)
  }
  u <- optim(log(pmin(pmax(c(tau_xi, xi_gamma), lb), ub)), neg_obj,
             method = "L-BFGS-B", lower = log(lb), upper = log(ub))$par
  list(tau_xi = exp(u[1]), xi_gamma = exp(u[2]))
}

# Closed-form MAP of (tau_xi, xi_gamma): per-species-normalised IG MAP from the
# prior sufficient statistics (the exact Xi-marginal MAP drives both to the
# upper bound on sparse data and freezes Xi at Phi; the closed form keeps them
# small and lets the embedding train).
update_xi_hyperparams_R <- function(Xi_mat, Phi_mat, taxonomy_codes, spec,
                                    xi_e, xi_f, xi_g, xi_h,
                                    tau_xi, xi_gamma,
                                    hyper_xi_prior = "ig",
                                    xi_s = 1.0, xi_s_gamma = xi_s,
                                    lb = 0.01, ub = 100,
                                    Xi_t = NULL, Phi_t = NULL, a = NULL) {
  if (is.null(Xi_t))  Xi_t  <- t(Xi_mat)
  if (is.null(Phi_t)) Phi_t <- t(Phi_mat)
  ss <- compute_xi_ss_cpp(Xi_t, Phi_t, taxonomy_codes, xi_gamma)
  S_xi <- sum(ss$S_vec)
  tau_new <- switch(hyper_xi_prior,
    ig    = ig_map_vec_cpp(c(S_xi), xi_e, xi_f, lb, ub)[[1]],
    gamma = max(lb, min(ub, 1.0 / (xi_f + S_xi))),
    hc    = hc_map_tau_cpp(c(S_xi), xi_s, lb, ub)[[1]])
  gam_new <- switch(hyper_xi_prior,
    ig    = ig_map_vec_cpp(c(ss$S_gamma), xi_g, xi_h, lb, ub)[[1]],
    gamma = max(lb, min(ub, 1.0 / (xi_h + ss$S_gamma))),
    hc    = hc_map_tau_cpp(c(ss$S_gamma), xi_s_gamma, lb, ub)[[1]])
  list(tau_xi = tau_new, xi_gamma = gam_new, S_xi = S_xi, S_vec = ss$S_vec, S_gamma = ss$S_gamma)
}

compute_xi_hyperparam_prior <- function(tau_xi, xi_gamma,
                                        hyper_xi_prior = "ig",
                                        xi_e = 0.1, xi_f = 0.5,
                                        xi_g = 0.1, xi_h = 2.0,
                                        xi_s = 1.0, xi_s_gamma = xi_s) {
  # tau_xi is a single shared scalar (not per-dim ARD), so each prior contributes once.
  switch(hyper_xi_prior,
    ig = {
      (-(xi_e + 1) * log(tau_xi) - xi_f / tau_xi) +
        (-(xi_g + 1) * log(xi_gamma) - xi_h / xi_gamma)
    },
    gamma = {
      (log(tau_xi) - xi_f * tau_xi) + (log(xi_gamma) - xi_h * xi_gamma)
    },
    hc = {
      # log p(tau) = -log(2s^2 tau + 1) - 1/2 log(tau)  [Jacobian of sigma=1/sqrt(2tau)]
      (-log(2 * xi_s^2 * tau_xi + 1) - 0.5 * log(tau_xi)) +
        (-log(2 * xi_s_gamma^2 * xi_gamma + 1) - 0.5 * log(xi_gamma))
    }
  )
}

compute_svd_fp <- function(Y, k, randomized_svd = FALSE) {
  n <- nrow(Y); p <- ncol(Y); k <- min(k, n, p)
  if (n > p) {
    YtY <- crossprod(Y)
    s <- if (randomized_svd) rsvd(YtY, k = k, p = 10, q = 2) else svd(YtY, nu = k, nv = k)
    V <- s$u[, seq_len(k), drop = FALSE]
    d <- pmax(s$d[seq_len(k)], 0)
    D <- diag(sqrt(d), nrow = k)
    U <- Y %*% V %*% diag(ifelse(d > 0, d^(-0.5), 0), nrow = k)
  } else {
    s <- if (randomized_svd) rsvd(Y, k = k, p = 10, q = 2) else svd(Y, nu = k, nv = k)
    U <- s$u[, seq_len(k), drop = FALSE]
    D <- diag(s$d[seq_len(k)], nrow = k)
    V <- s$v[, seq_len(k), drop = FALSE]
  }
  list(U = U, D = D, V = V)
}

compute_M_prior_term <- function(M, prior_var = 1) {
  prior_var <- max(1e-12, as.numeric(prior_var)[1])
  -0.5 * sum(M^2) / prior_var
}

compute_Gamma_prior_term <- function(Gamma, kappa_sq_vec) {
  h <- ncol(Gamma)
  -0.5 * sum(rowSums(Gamma^2) / kappa_sq_vec) -
    0.5 * h * sum(log(kappa_sq_vec))
}

compute_sigma_prior_term <- function(sigma_sq_vec, a, b) {
  sum(a * log(b) - lgamma(a) - (a + 1) * log(sigma_sq_vec) - b / sigma_sq_vec)
}

compute_kappa_prior_term <- function(kappa_sq_vec, kappa_prior, c_hyp, d_hyp, kappa_hc_scale) {
  if (kappa_prior == "ig")
    sum(c_hyp * log(d_hyp) - lgamma(c_hyp) - (c_hyp + 1) * log(kappa_sq_vec) - d_hyp / kappa_sq_vec)
  else
    -0.5 * sum(log(kappa_sq_vec)) - sum(log(1 + kappa_sq_vec / kappa_hc_scale^2))
}

print_runtime_diagnostics <- function(requested_cores, effective_workers, BL_species_batch_size) {
  soft <- tryCatch(extSoftVersion(), error = function(e) NULL)
  blas_path <- if (!is.null(soft) && "BLAS" %in% names(soft)) soft[["BLAS"]] else NA_character_
  lapack_path <- if (!is.null(soft) && "LAPACK" %in% names(soft)) soft[["LAPACK"]] else NA_character_
  env_or_na <- function(x) {
    val <- Sys.getenv(x, unset = "")
    if (identical(val, "")) "<unset>" else val
  }
  cat(sprintf(
    paste0(
      "  Runtime diag | BLAS=%s | LAPACK=%s | C++ OpenMP=%s | ",
      "requested cores=%d | effective workers=%d | BL_species_batch=%d\n"
    ),
    ifelse(is.na(blas_path), "<unknown>", blas_path),
    ifelse(is.na(lapack_path), "<unknown>", lapack_path),
    if (isTRUE(flair_plus_has_openmp_cpp())) "TRUE" else "FALSE",
    requested_cores, effective_workers, BL_species_batch_size
  ))
  cat(sprintf(
    paste0(
      "  Thread env   | OMP_NUM_THREADS=%s | OPENBLAS_NUM_THREADS=%s | ",
      "MKL_NUM_THREADS=%s | VECLIB_MAXIMUM_THREADS=%s\n"
    ),
    env_or_na("OMP_NUM_THREADS"),
    env_or_na("OPENBLAS_NUM_THREADS"),
    env_or_na("MKL_NUM_THREADS"),
    env_or_na("VECLIB_MAXIMUM_THREADS")
  ))
  # Note: OPENBLAS_NUM_THREADS env var is intentionally set to 1 by callers to
  # prevent uncontrolled threading in R-level BLAS calls; the C++ shared-Hessian
  # code overrides it at runtime via openblas_set_num_threads() (dlsym), so the
  # env var does not limit parallelism inside map_latent_factors_shared_hessian*.
  omp_n <- suppressWarnings(as.integer(Sys.getenv("OMP_NUM_THREADS", "")))
  if (!is.na(omp_n) && omp_n < requested_cores)
    message(sprintf(
      "  [WARN] OMP_NUM_THREADS=%d < requested cores=%d.\n  ",
      omp_n, requested_cores),
      "Increase OMP_NUM_THREADS (or SLURM --cpus-per-task) so the C++ OMP ",
      "parallel loops can use all allocated cores.")
}

get_init_plus <- function(Y, X, k, Xi_mat,
                          randomized_svd = FALSE,
                          eps      = 1e-3,
                          sigma_lb = 0.01,
                          sigma_ub = 100,
                          kappa_lb = 0.01,
                          kappa_ub = 100,
                          gamma_ridge = 1e-3) {
  n <- nrow(Y); q <- ncol(X)
  if (!all(is.finite(Xi_mat)))
    stop("Xi_mat contains non-finite values before FLAIR+ initialization.")
  s_Y <- compute_svd_fp(Y, k + q, randomized_svd)
  Y_hat <- pmax(pmin(s_Y$U %*% s_Y$D %*% t(s_Y$V), 1 - eps), eps)
  Z_hat <- safe_logit(Y_hat)
  Beta_hat <- t(solve(crossprod(X), crossprod(X, Z_hat)))
  Z_c <- Z_hat - X %*% t(Beta_hat)
  s_Z <- compute_svd_fp(Z_c, k, randomized_svd)
  M_hat <- s_Z$U * sqrt(n)
  Lambda_hat <- s_Z$V %*% s_Z$D / sqrt(n)
  Beta_Lambda <- cbind(Beta_hat, Lambda_hat)
  XiXit <- Xi_mat %*% t(Xi_mat)
  ridge <- gamma_ridge * max(1, mean(diag(XiXit)))
  Gamma <- t(Beta_Lambda) %*% t(Xi_mat) %*%
    solve(XiXit + diag(ridge, nrow(XiXit)))
  Mu <- t(Gamma %*% Xi_mat)
  Delta    <- Beta_Lambda - Mu
  # Use the same InvGamma-MAP formula as update_sigma_sq_cpp so initialization is
  # consistent with subsequent updates and doesn't collapse to sigma_lb = 0.01 when
  # a large-scale embedding (e.g. PCA) lets Gamma nearly perfectly fit BL at init.
  p_bl     <- nrow(Beta_Lambda)
  SS       <- colSums(Delta^2)
  sigma_sq <- pmax(pmin((2 + SS / 2) / (3 + p_bl / 2 + 1), sigma_ub), sigma_lb)  # (q+k)
  kappa_sq <- pmax(pmin(rowMeans(Gamma^2), kappa_ub), kappa_lb)   # (q+k): per-row of Gamma
  if (!all(is.finite(Beta_Lambda)) || !all(is.finite(M_hat)) ||
      !all(is.finite(Gamma)) || !all(is.finite(sigma_sq)) ||
      !all(is.finite(kappa_sq))) {
    stop("FLAIR+ initialization produced non-finite values; check embedding scale/rank.")
  }
  list(Beta_Lambda = Beta_Lambda, M = M_hat, Gamma = Gamma,
       sigma_sq = sigma_sq, kappa_sq = kappa_sq,
       a = 3, b = 2, c_hyp = 3, d_hyp = 4,
       kappa_prior = "ig", kappa_hc_scale = 1.0)
}

post_process_plus <- function(X, M, Beta_Lambda, q) {
  n <- nrow(X); k <- ncol(M)
  H_x <- solve(crossprod(X), t(X))
  M_res <- M - X %*% (H_x %*% M)
  s_M <- svd(M_res, nu = k, nv = k)
  Beta <- Beta_Lambda[, seq_len(q), drop = FALSE]
  Lambda <- Beta_Lambda[, q + seq_len(k), drop = FALSE]
  Beta_new <- Beta + Lambda %*% (t(M) %*% t(H_x))
  M_new <- s_M$u[, seq_len(k), drop = FALSE] * sqrt(n)
  VD <- s_M$v[, seq_len(k), drop = FALSE] %*% diag(s_M$d[seq_len(k)], nrow = k) / sqrt(n)
  Lambda_new <- Lambda %*% VD
  # b~_j -> T b~_j, the same linear map for every species (and for the prior mean Gamma xi_j)
  T_pp <- rbind(cbind(diag(q), H_x %*% M), cbind(matrix(0, k, q), t(VD)))
  list(Beta_Lambda = cbind(Beta_new, Lambda_new), M = M_new, T = T_pp)
}

realign_plus_prior <- function(Beta_Lambda, M, Gamma, Xi_mat,
                               sigma_sq_vec, kappa_sq_vec,
                               a, b, c_hyp, d_hyp,
                               kappa_prior = "ig", kappa_hc_scale = 1.0,
                               kappa_ub = 100) {
  Gamma <- update_Gamma_cpp(Beta_Lambda, Xi_mat, sigma_sq_vec, kappa_sq_vec)
  sigma_sq_vec <- update_sigma_sq_cpp(Beta_Lambda, Gamma, Xi_mat, a, b, 0.01, 100)
  kappa_sq_vec <- if (kappa_prior == "ig")
    update_kappa_sq_cpp(Gamma, c_hyp, d_hyp, 0.01, kappa_ub) else
    update_kappa_sq_halfcauchy_cpp(Gamma, kappa_hc_scale, 0.01, kappa_ub)
  list(Beta_Lambda = Beta_Lambda, M = M, Gamma = Gamma,
       sigma_sq = sigma_sq_vec, kappa_sq = kappa_sq_vec)
}

as_numeric_binary <- function(A) {
  A_num <- as.matrix(A)
  storage.mode(A_num) <- "double"
  A_num
}

is_full_observed <- function(obs) {
  all(is.na(obs)) || all(obs == 1L | obs == 1)
}

make_x_tilde <- function(X, M) {
  q <- ncol(X); k <- ncol(M)
  X_tilde <- matrix(0, nrow(X), q + k)
  X_tilde[, seq_len(q)] <- X
  X_tilde[, q + seq_len(k)] <- M
  X_tilde
}

update_x_tilde_M <- function(X_tilde, X, M) {
  q <- ncol(X)
  X_tilde[, q + seq_len(ncol(M))] <- M
  X_tilde
}

project_M_species_blocks <- function(Y, X, Lambda, Beta, observed = NA,
                                     M_init = NULL, prior_var = NULL,
                                     epsilon = 1e-3, cores = 1L,
                                     species_block = 1000L,
                                     step_init = 0.25,
                                     max_ls = 8L,
                                     max_it = 100L,
                                     verbose = TRUE) {
  Y_num <- as_numeric_binary(Y)
  if (is.null(M_init)) {
    M_init <- matrix(0, nrow(Y_num), ncol(Lambda))
  }
  if (is.null(prior_var)) prior_var <- 1 / ncol(Lambda)
  prior_var <- max(1e-12, as.numeric(prior_var)[1])
  full_obs <- is_full_observed(observed)
  max_it <- max(1L, as.integer(max_it))
  step_cur <- max(1e-8, as.numeric(step_init))
  last_res <- NULL
  if (full_obs) {
    if (verbose) {
      cat(sprintf("  project_M species-block: full obs, species=%d, block=%d\n",
                  ncol(Y_num), as.integer(species_block)))
    }
    for (it in seq_len(max_it)) {
      last_res <- map_latent_factors_shared_hessian_species_blocks_fullobs(
        Y_num, X, Lambda, Beta, M_init,
        prior_var = prior_var, epsilon = epsilon,
        n_threads = cores, species_block = species_block,
        step_init = step_cur, max_ls = max_ls
      )
      M_init <- last_res$M
      step_cur <- last_res$step_out
      if (verbose) {
        cat(sprintf("    project_M iter %d | step=%.3g ls_trials=%d\n",
                    it, last_res$step, last_res$ls_trials))
        cat(sprintf("      project_M TR | %s\n", format_m_tr_diag(last_res)))
        cat(sprintf("      project_M coords | %s\n",
                    format_latent_coord_diag(M_init, Lambda)))
        cat(sprintf("      project_M similarity | %s\n",
                    format_latent_similarity_diag(M_init, Lambda)))
      }
      if (!isTRUE(last_res$accepted) || last_res$step < epsilon) break
    }
  } else {
    obs_num <- as_numeric_binary(observed)
    keep <- colSums(obs_num == 1) > 0
    dropped <- sum(!keep)
    if (verbose) {
      cat(sprintf("  project_M species-block: observed species=%d/%d, dropped zero-observed=%d, block=%d\n",
                  sum(keep), length(keep), dropped, as.integer(species_block)))
    }
    if (!any(keep)) {
      return(list(M = M_init, step_out = step_init, ll = NA_real_,
                  n_species = 0L, accepted = FALSE, step = 0,
                  ls_trials = 0L, timing = numeric(0)))
    }
    Y_keep <- Y_num[, keep, drop = FALSE]
    obs_keep <- obs_num[, keep, drop = FALSE]
    Lambda_keep <- Lambda[keep, , drop = FALSE]
    Beta_keep <- Beta[keep, , drop = FALSE]
    for (it in seq_len(max_it)) {
      last_res <- map_latent_factors_shared_hessian_species_blocks(
        Y_keep, obs_keep, X, Lambda_keep, Beta_keep, M_init,
        prior_var = prior_var, epsilon = epsilon,
        n_threads = cores, species_block = species_block,
        step_init = step_cur, max_ls = max_ls
      )
      M_init <- last_res$M
      step_cur <- last_res$step_out
      if (verbose) {
        cat(sprintf("    project_M iter %d | step=%.3g ls_trials=%d\n",
                    it, last_res$step, last_res$ls_trials))
        cat(sprintf("      project_M TR | %s\n", format_m_tr_diag(last_res)))
        cat(sprintf("      project_M coords | %s\n",
                    format_latent_coord_diag(M_init, Lambda_keep)))
        cat(sprintf("      project_M similarity | %s\n",
                    format_latent_similarity_diag(M_init, Lambda_keep)))
      }
      if (!isTRUE(last_res$accepted) || last_res$step < epsilon) break
    }
  }
  if (is.null(last_res)) {
    last_res <- list(M = M_init, step_out = step_cur, ll = NA_real_,
                     n_species = ncol(Y_num), accepted = FALSE, step = 0,
                     ls_trials = 0L, timing = numeric(0))
  }
  if (verbose && length(last_res$timing)) {
    timing <- as.numeric(last_res$timing)
    names(timing) <- names(last_res$timing)
    cat(sprintf("    project_M timing | %s\n", format_m_timing(timing)))
  }
  last_res
}

format_m_timing <- function(timing) {
  if (!length(timing)) return("")
  sprintf(
    "cpu-s: subset=%.3fs  XB=%.3fs  eta=%.3fs  sigmoid+old_lhat=%.3fs  hessian=%.3fs  grad=%.3fs  solve=%.3fs  ls_eta=%.3fs  ls_lhat=%.3fs | wall total_cpp=%.3fs",
    timing[["subset"]], timing[["XB"]], timing[["eta"]], timing[["sigmoid"]],
    timing[["hessian"]], timing[["grad"]], timing[["solve"]],
    timing[["ls_eta"]], timing[["ls_lhat"]], timing[["total"]]
  )
}

format_m_tr_diag <- function(res) {
  get_num <- function(name) {
    val <- res[[name]]
    if (is.null(val) || !length(val)) NA_real_ else as.numeric(val)[1]
  }
  sprintf(
    "accepted=%s  rho=%.3g  actual=%.3g  pred=%.3g  step_norm=%.3g  try_delta=%.3g  step_out=%.3g  max_dir=%.3g  ll_old=%.3g  ll_new=%.3g",
    if (isTRUE(res$accepted)) "TRUE" else "FALSE",
    get_num("rho"), get_num("actual"), get_num("pred"),
    get_num("step_norm"), get_num("try_delta"), get_num("step_out"),
    get_num("max_dir_norm"), get_num("ll_old"), get_num("ll_new")
  )
}

format_latent_coord_diag <- function(M, Lambda, max_coords = 15L) {
  if (is.null(M) || is.null(Lambda) || !ncol(M) || !ncol(Lambda)) return("")
  k_show <- min(ncol(M), ncol(Lambda), as.integer(max_coords))
  m_sd <- apply(M[, seq_len(k_show), drop = FALSE], 2L, sd)
  m_abs <- colMeans(abs(M[, seq_len(k_show), drop = FALSE]))
  lam_norm <- sqrt(colSums(Lambda[, seq_len(k_show), drop = FALSE]^2))
  prod_sdM_lam <- m_sd * lam_norm
  paste(sprintf(
    "k%d:sdM=%.3g|absM=%.3g|lam=%.3g|prod_sdM_lam=%.3g",
    seq_len(k_show), m_sd, m_abs, lam_norm, prod_sdM_lam
  ), collapse = "  ")
}

format_latent_similarity_diag <- function(M, Lambda, max_pairs = 5L) {
  if (is.null(M) || is.null(Lambda) || ncol(M) < 2L || ncol(Lambda) < 2L)
    return("k<2")
  k <- min(ncol(M), ncol(Lambda))
  M_use <- M[, seq_len(k), drop = FALSE]
  L_use <- Lambda[, seq_len(k), drop = FALSE]

  safe_cor <- function(A) {
    s <- apply(A, 2L, sd)
    keep <- is.finite(s) & s > 0
    out <- diag(1, ncol(A))
    if (sum(keep) >= 2L) {
      out[keep, keep] <- suppressWarnings(cor(A[, keep, drop = FALSE]))
    }
    out[!is.finite(out)] <- 0
    out
  }
  safe_cos <- function(A) {
    nr <- sqrt(colSums(A^2))
    denom <- tcrossprod(nr)
    out <- crossprod(A) / pmax(denom, 1e-12)
    out[!is.finite(out)] <- 0
    diag(out) <- 1
    out
  }
  top_pairs <- function(S, label) {
    S <- as.matrix(S)
    S[lower.tri(S, diag = TRUE)] <- NA_real_
    vals <- abs(S)
    ok <- which(is.finite(vals), arr.ind = TRUE)
    if (!nrow(ok)) return(sprintf("%s=NA", label))
    ord <- order(vals[ok], decreasing = TRUE)
    ok <- ok[ord[seq_len(min(length(ord), as.integer(max_pairs)))], , drop = FALSE]
    paste0(
      label, "_max=", sprintf("%.3f", vals[ok[1, 1], ok[1, 2]]),
      " [",
      paste(sprintf("k%d-k%d:%.3f", ok[, 1], ok[, 2], S[ok]), collapse = ", "),
      "]"
    )
  }

  eta_cor <- safe_cor(M_use)
  eta_cos <- safe_cos(M_use)
  lambda_cos <- safe_cos(L_use)
  prod_eta_lambda_cos <- eta_cos * lambda_cos

  paste(
    top_pairs(eta_cor, "eta_cor"),
    top_pairs(prod_eta_lambda_cos, "prod_eta_lambda_cos"),
    sep = "  "
  )
}

# build_batch_order is handled by build_batch_order_cpp() in helper_functions_trustregion.cpp.
# The R implementation used 32-bit integer arithmetic that overflowed when n > ~46K
# (e.g. n = 204K ultra-rare species), producing NA indices that crashed Armadillo.

FLAIR_plus_minibatch_pretrain_wrapper <- function(
    Y, X, Xi_mat, k,
    Y_all = NULL, Xi_all = NULL,
    Phi_mat = Xi_mat, Phi_all = Xi_all,
    taxonomy_codes = NULL, taxonomy_codes_all = NULL,
    xi_trainable = TRUE,
    trait_intercept = TRUE,    # append a constant, untrained embedding row: prior mean mu + Gamma xi_j
    mu_prior_var = 100,        # prior variance of the community mean mu (last column of Gamma)
    sigma_em_ub = 100,         # upper clamp of the EM sigma^2 update (caps V_diag inflation)
    vdiag_freq = 10L,          # recompute V_diag every this many stage-2 iterations
    xi_update_freq = 1L,       # update Xi every this many stage-2 iterations
    xi_late_freeze = 0L,       # freeze Xi after this stage-2 iteration (0 = never)
    xi_lambda = 0.1,
    xi_gamma = 0.1,
    xi_e = 0.1,
    xi_f = 0.5,
    xi_g = 0.1,
    xi_h = 2.0,
    hyper_xi_prior = c("ig", "gamma", "hc"),
    xi_s = NULL,
    xi_s_gamma = NULL,
    xi_maxit = 50L,
    xi_tol = 1e-4,
    observed = NA, observed_all = NA,
    iter_max_pretrain = 10,
    iter_max_train = 50,
    tol = 1e-4,
    loss_tol = 1e-4,
    C_clip = 10,
    BL_species_batch_size = 1000L,
    cores = NULL,
    step_cap = NULL,
    randomize_batches = FALSE,
    do_post_process = TRUE,
    randomized_svd = FALSE,
    eps = 1e-3,
    debug_blocks = FALSE,
    timing_diagnostics = TRUE,
    m_block = 300L,
    M_species_batch_size = 1000L,
    m_step_size = 0.25,
    m_lhat_max_ls = 8L,
    m_prior_var = NULL,
    stage1_m_update = c("backbone", "species_minibatch"),
    stage2_m_update = c("all", "backbone", "species_minibatch"),
    init_override = NULL,
    xi_freeze_iters = 0L,
    kappa_prior = c("ig", "halfcauchy"),
    kappa_hc_scale = 1.0,
    kappa_c = NULL,
    kappa_d = NULL,
    kappa_ub = 100,
    sigma_a = NULL,
    sigma_b_pretrain = NULL,
    sigma_b_train = NULL,
    xi_hyper_em = FALSE) {     # EM update of (tau_xi, xi_gamma) instead of the closed form

  kappa_prior <- match.arg(kappa_prior)

  stage1_m_update <- match.arg(stage1_m_update)
  stage2_m_update <- match.arg(stage2_m_update)
  xi_trainable <- isTRUE(xi_trainable)

  # Row-standardize Xi/Phi: each trait dimension h -> mean 0, sd 1 over species.
  # Stats are computed from Xi_mat (backbone) and applied to all four matrices.
  {
    xi_row_means <- rowMeans(Xi_mat)
    xi_row_sds   <- apply(Xi_mat, 1L, sd)
    xi_row_sds[xi_row_sds < 1e-10] <- 1
    Xi_mat  <- sweep(sweep(Xi_mat,  1L, xi_row_means, "-"), 1L, xi_row_sds, "/")
    Phi_mat <- sweep(sweep(Phi_mat, 1L, xi_row_means, "-"), 1L, xi_row_sds, "/")
    if (!is.null(Xi_all)) {
      Xi_all  <- sweep(sweep(Xi_all,  1L, xi_row_means, "-"), 1L, xi_row_sds, "/")
      Phi_all <- sweep(sweep(Phi_all, 1L, xi_row_means, "-"), 1L, xi_row_sds, "/")
    }
  }
  if (trait_intercept) {
    list2env(trait_mean_ops(mu_prior_var), environment())
    .gamma_fn <- update_Gamma_cpp   # mu column needs the intercept ridge, not the fast path
    Xi_mat <- rbind(Xi_mat, 1); Phi_mat <- rbind(Phi_mat, 1)
    if (!is.null(Xi_all)) { Xi_all <- rbind(Xi_all, 1); Phi_all <- rbind(Phi_all, 1) }
  }

  # Stage 1: pretrain on the user-supplied subset.
  Y_sub_num <- as_numeric_binary(Y)
  full_obs_sub <- is_full_observed(observed)
  obs_sub_num <- if (full_obs_sub) NULL else as_numeric_binary(observed)

  init <- if (!is.null(init_override)) {
    stopifnot(is.list(init_override))
    init_override
  } else {
    get_init_plus(Y_sub_num, X, k, Xi_mat,
                  randomized_svd = randomized_svd, eps = eps)
  }

  # Safety clamp: if randomised SVD (or a degenerate seed) produced very large
  # Lambda column norms the M-step Hessian scales as ||Lambda||^2 and the
  # backbone diverges.  We scale both Lambda[:,j] and M[:,j] by the SAME factor
  # sf = lam_cap / ||Lambda[:,j]|| so that the capped Lambda norms equal lam_cap
  # exactly.  Scaling M by the same sf (rather than 1/sf) shrinks predictions
  # by sf^2, pulling them out of sigmoid saturation so the backbone can
  # self-correct.  lp_M improves proportionally (M smaller → less prior penalty).
  {
    q_init      <- ncol(X)
    Lam_init    <- init$Beta_Lambda[, q_init + seq_len(k), drop = FALSE]
    lam_norms   <- sqrt(colSums(Lam_init^2))
    lam_cap     <- 5 * sqrt(nrow(init$M))   # 5·√n_sub ≈ 190 for n=1437
    needs_clip  <- lam_norms > lam_cap
    if (any(needs_clip)) {
      sf <- ifelse(needs_clip, lam_cap / lam_norms, 1)  # exact cap, not √-ratio
      init$Beta_Lambda[, q_init + seq_len(k)] <- Lam_init %*% diag(sf, nrow = k)
      init$M <- init$M %*% diag(sf, nrow = k)
      cat(sprintf(
        "  [init] Lambda norm cap (cap=%.0f): norms %s → %s\n",
        lam_cap,
        paste(round(lam_norms, 1), collapse = "/"),
        paste(round(lam_norms * sf, 1), collapse = "/")))
    }
  }

  Beta_Lambda  <- init$Beta_Lambda
  M            <- init$M
  Gamma        <- init$Gamma
  sigma_sq_vec <- init$sigma_sq   # (q+k): per-dimension
  kappa_sq_vec <- init$kappa_sq   # (q+k): per-row of Gamma
  a <- init$a; b <- init$b; c_hyp <- init$c_hyp; d_hyp <- init$d_hyp
  if (!is.null(kappa_c)) c_hyp <- as.numeric(kappa_c)[1]
  if (!is.null(kappa_d)) d_hyp <- as.numeric(kappa_d)[1]
  if (!is.null(sigma_a)) a <- as.numeric(sigma_a)[1]
  if (!is.null(sigma_b_pretrain)) b <- as.numeric(sigma_b_pretrain)[1]
  if (!is.finite(a) || a <= 0 || !is.finite(b) || b <= 0)
    stop("sigma_a and sigma_b_pretrain must be positive finite values.")
  sigma_sq_vec <- update_sigma_sq_cpp(
    Beta_Lambda, Gamma, Xi_mat, a, b, 0.01, 100
  )

  q <- ncol(X)
  if (is.null(cores)) {
    slurm_n <- suppressWarnings(as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "")))
    cores <- if (!is.na(slurm_n) && slurm_n >= 1L)
      max(1L, slurm_n - 2L)
    else
      max(1L, as.integer(parallelly::availableCores()) - 2L)
  }
  cores <- max(1L, as.integer(cores))
  iter_max_pretrain <- max(0L, as.integer(iter_max_pretrain))
  iter_max_train <- max(1L, as.integer(iter_max_train))
  xi_freeze_iters <- max(0L, as.integer(xi_freeze_iters))
  BL_species_batch_size <- max(1L, as.integer(BL_species_batch_size))
  M_species_batch_size <- max(1L, as.integer(M_species_batch_size))
  m_step_size <- max(1e-8, as.numeric(m_step_size))
  m_lhat_max_ls <- max(1L, as.integer(m_lhat_max_ls))
  if (is.null(m_prior_var)) m_prior_var <- 1 / k
  m_prior_var <- max(1e-12, as.numeric(m_prior_var)[1])
  worker_cores <- min(cores, BL_species_batch_size)
  has_openmp <- isTRUE(flair_plus_has_openmp_cpp())
  if (is.null(step_cap))
    step_cap <- sqrt(q + k)
  tau_xi <- as.numeric(xi_lambda)[1]
  if (xi_trainable) {
    if (!is.null(taxonomy_codes)) storage.mode(taxonomy_codes) <- "integer"
    if (!is.null(taxonomy_codes_all)) storage.mode(taxonomy_codes_all) <- "integer"
    xi_lambda <- as.numeric(xi_lambda)[1]
    tau_xi    <- xi_lambda                     # single shared scalar τ_ξ (not per-dim ARD)
    tau_vec   <- rep(tau_xi, nrow(Xi_mat))     # H-vector for C++ calls; always rep(tau_xi, H)
    xi_gamma <- as.numeric(xi_gamma)[1]
    xi_e <- as.numeric(xi_e)[1]
    xi_f <- as.numeric(xi_f)[1]
    xi_g <- as.numeric(xi_g)[1]
    xi_h <- as.numeric(xi_h)[1]
    hyper_xi_prior <- match.arg(hyper_xi_prior)
    if (hyper_xi_prior == "hc") {
      # Default s = sqrt(mean col-variance of Phi_mat) so HC prior on sigma_h is
      # scaled to the embedding's own magnitude.  tau=1/(s*sqrt(2S)) when S<<s^2,
      # so s controls the floor: larger s = looser prior = lower tau floor.
      # NOTE: at init S_h~0, tau→ub regardless of s. HC is most effective once
      # Xi has moved; consider hyper_xi_prior="gamma" if Xi doesn't escape Phi.
      if (is.null(xi_s))
        xi_s <- sqrt(max(mean(colMeans(t(Phi_mat)^2)), 1e-6))
      if (is.null(xi_s_gamma)) xi_s_gamma <- xi_s
    } else {
      if (is.null(xi_s))       xi_s       <- 1.0
      if (is.null(xi_s_gamma)) xi_s_gamma <- 1.0
    }
    xi_s <- as.numeric(xi_s)[1]
    xi_s_gamma <- as.numeric(xi_s_gamma)[1]
    xi_maxit <- max(1L, as.integer(xi_maxit))
    xi_tol <- as.numeric(xi_tol)[1]
  }

  X_tilde <- make_x_tilde(X, M)
  Mu <- t(Gamma %*% Xi_mat)
  ll_cur <- compute_data_term_cpp(
    Y_sub_num,
    if (is.null(obs_sub_num)) matrix(0, 0, 0) else obs_sub_num,
    is.null(obs_sub_num),
    X_tilde,
    Beta_Lambda
  )
  lp_bl_cur  <- compute_beta_prior_term_cpp(Beta_Lambda, Mu, sigma_sq_vec)
  lp_M_cur   <- compute_M_prior_term(M, m_prior_var)
  lp_G_cur   <- compute_Gamma_prior_term(Gamma, kappa_sq_vec)
  lp_sig_cur <- compute_sigma_prior_term(sigma_sq_vec, a, b)
  lp_kap_cur <- compute_kappa_prior_term(kappa_sq_vec, kappa_prior, c_hyp, d_hyp, kappa_hc_scale)
  lp_xi_cur <- 0
  lp_old <- ll_cur + lp_bl_cur + lp_M_cur + lp_G_cur + lp_sig_cur + lp_kap_cur + lp_xi_cur
  ord_base <- order(colSums(Y), decreasing = TRUE)
  idx_batches <- split(ord_base, ceiling(seq_along(ord_base) / BL_species_batch_size))
  effective_workers <- if (has_openmp) min(worker_cores, length(idx_batches)) else 1L
  cat(sprintf("FLAIR+ minibatch backbone | n=%d  p=%d  q=%d  k=%d  h=%d  BL_species_batch=%d\n",
              nrow(Y_sub_num), ncol(Y_sub_num), q, k, nrow(Xi_mat), BL_species_batch_size))
  cat(sprintf("  Sigma_sq prior: IG(a=%.4g, b=%.4g)\n", a, b))
  cat(sprintf("  M prior: eta_i ~ N(0, %.4g I_k); E||eta_i||^2 = %.4g\n",
              m_prior_var, k * m_prior_var))
  if (xi_trainable) {
    cat("  Xi update: fixed during backbone pretrain; lambda/gamma start in all-species fit.\n")
    cat(sprintf("  Xi hyperparam prior: %s", hyper_xi_prior))
    if (hyper_xi_prior == "ig")
      cat(sprintf("  (tau~IG(e=%.3g,f=%.3g) mode=%.4g;  gamma_xi~IG(g=%.3g,h=%.3g) mode=%.4g)\n",
                  xi_e, xi_f, xi_f/(xi_e+1), xi_g, xi_h, xi_h/(xi_g+1)))
    else if (hyper_xi_prior == "gamma")
      cat(sprintf("  (tau_xi~Gamma(2,f=%.3g) mode=%.4g;  gamma_xi~Gamma(2,h=%.3g) mode=%.4g)\n",
                  xi_f, 1/xi_f, xi_h, 1/xi_h))
    else
      cat(sprintf("  (HC(0,s=%.3g) on sigma_h=1/sqrt(2*tau_h);  HC(0,s_gamma=%.3g) on sigma_gamma)\n",
                  xi_s, xi_s_gamma))
  }
  cat(sprintf("  Stage1 M update = %s%s\n",
              stage1_m_update,
              if (identical(stage1_m_update, "species_minibatch"))
                sprintf(" columns (backbone species blocks, block=%d)",
                        M_species_batch_size)
              else " columns"))
  print_runtime_diagnostics(cores, effective_workers, BL_species_batch_size)
  cat(sprintf("  Block workers used = %d%s\n",
              effective_workers,
              if (effective_workers > 1L) "" else " (sequential)"))
  cat(sprintf("  Init  | log-post = %12.2f\n", lp_old))
  bb_delta <- 1.0
  bb_m_step_size <- m_step_size
  bb_n_diverge <- 0L   # consecutive iterations where log-post strictly worsens

  if (iter_max_pretrain == 0L && !is.null(Y_all) && !is.null(Xi_all)) {
    cat("  Stage1 pretrain skipped; using SVD backbone initialization for Stage2.\n")
  }
  for (iter in seq_len(iter_max_pretrain)) {
    iter_t0 <- proc.time()[3]

    t_mu <- 0
    ord_iter <- build_batch_order_cpp(ord_base, randomize_batches && iter > 1L)

    Beta   <- Beta_Lambda[, seq_len(q),      drop = FALSE]
    Lambda <- Beta_Lambda[, q + seq_len(k),  drop = FALSE]

    if (iter == 1L)
      cat(sprintf("  [ref] ||grad_M(M0)|| = %.4g  (Lin 2008 suggested Δ₀; current Δ₀ = %.4g)\n",
                  m_grad_norm_cpp(Y_sub_num,
                                  if (full_obs_sub) matrix(0, 0, 0) else obs_sub_num,
                                  X, Lambda, Beta, M, m_prior_var),
                  m_step_size))
    t_m0 <- proc.time()[3]
    bb_m_diag <- NULL
    if (identical(stage1_m_update, "species_minibatch")) {
      bb_res <- if (full_obs_sub) {
        map_latent_factors_shared_hessian_species_blocks_fullobs(
          Y_sub_num, X, Lambda, Beta, M,
        prior_var = m_prior_var, epsilon = tol,
          n_threads = cores, species_block = M_species_batch_size,
          step_init = bb_m_step_size, max_ls = m_lhat_max_ls,
          species_order_r = as.integer(ord_iter),
          ll_init = ll_cur
        )
      } else {
        map_latent_factors_shared_hessian_species_blocks(
          Y_sub_num, obs_sub_num, X, Lambda, Beta, M,
        prior_var = m_prior_var, epsilon = tol,
          n_threads = cores, species_block = M_species_batch_size,
          step_init = bb_m_step_size, max_ls = m_lhat_max_ls,
          species_order_r = as.integer(ord_iter),
          ll_init = ll_cur
        )
      }
      M <- bb_res$M
      bb_m_step_size <- bb_res$step_out
      bb_m_diag <- bb_res
    } else {
      bb_res <- if (full_obs_sub) {
        map_latent_factors_shared_hessian_fullobs(
          Y_sub_num, X, Lambda, Beta, M,
          prior_var = m_prior_var, epsilon = tol,
          n_threads = cores, m_block = m_block, step_init = bb_delta
        )
      } else {
        map_latent_factors_shared_hessian(
          Y_sub_num, obs_sub_num, X, Lambda, Beta, M,
          prior_var = m_prior_var, epsilon = tol,
          n_threads = cores, m_block = m_block, step_init = bb_delta
        )
      }
      M        <- bb_res$M
      bb_delta <- bb_res$step_out
    }
    t_m <- proc.time()[3] - t_m0
    X_tilde <- update_x_tilde_M(X_tilde, X, M)
    t_data0 <- proc.time()[3]
    ll_cur <- compute_data_term_cpp(
      Y_sub_num,
      if (is.null(obs_sub_num)) matrix(0, 0, 0) else obs_sub_num,
      is.null(obs_sub_num),
      X_tilde,
      Beta_Lambda
    )
    t_data <- proc.time()[3] - t_data0
    lp_M_cur <- compute_M_prior_term(M, m_prior_var)
    cat(sprintf("    [backbone M]  ll_data=%12.2f  lp_M=%12.2f  subtotal=%12.2f\n",
                ll_cur, lp_M_cur, ll_cur + lp_M_cur + lp_bl_cur))

    t_bl0 <- proc.time()[3]
    blk_res <- map_beta_lambda_plus_blocks_cpp(
      Y_sub_num,
      if (is.null(obs_sub_num)) matrix(0, 0, 0) else obs_sub_num,
      is.null(obs_sub_num),
      X_tilde,
      Beta_Lambda,
      Mu,
      sigma_sq_vec,
      as.integer(ord_iter),
      BL_species_batch_size,
      C_clip,
      step_cap,
      effective_workers
    )
    t_bl <- proc.time()[3] - t_bl0

    t_merge0 <- proc.time()[3]
    Beta_Lambda <- blk_res$Beta_Lambda
    ll_cur <- ll_cur + sum(blk_res$delta_ll)
    lp_bl_cur <- lp_bl_cur + sum(blk_res$delta_prior)
    t_merge <- proc.time()[3] - t_merge0
    cat(sprintf("    [backbone BL] ll_data=%12.2f  lp_BL=%12.2f  subtotal=%12.2f\n",
                ll_cur, lp_bl_cur, ll_cur + lp_M_cur + lp_bl_cur))

    t_g0 <- proc.time()[3]
    Gamma <- .gamma_fn(Beta_Lambda, Xi_mat, sigma_sq_vec, kappa_sq_vec)
    Mu <- t(Gamma %*% Xi_mat)
    t_g <- proc.time()[3] - t_g0

    t_xi <- 0

    t_s0 <- proc.time()[3]
    sigma_sq_vec <- update_sigma_sq_cpp(Beta_Lambda, Gamma, Xi_mat, a, b, 0.01, 100)
    lp_bl_cur  <- compute_beta_prior_term_cpp(Beta_Lambda, Mu, sigma_sq_vec)
    lp_sig_cur <- compute_sigma_prior_term(sigma_sq_vec, a, b)
    kappa_sq_vec <- if (kappa_prior == "ig")
      update_kappa_sq_cpp(Gamma, c_hyp, d_hyp, 0.01, kappa_ub) else
      update_kappa_sq_halfcauchy_cpp(Gamma, kappa_hc_scale, 0.01, kappa_ub)
    lp_G_cur   <- compute_Gamma_prior_term(Gamma, kappa_sq_vec)
    lp_kap_cur <- compute_kappa_prior_term(kappa_sq_vec, kappa_prior, c_hyp, d_hyp, kappa_hc_scale)
    t_s <- proc.time()[3] - t_s0

    t_lp0 <- proc.time()[3]
    lp_new <- ll_cur + lp_bl_cur + lp_M_cur + lp_G_cur + lp_sig_cur + lp_kap_cur + lp_xi_cur
    t_lp <- proc.time()[3] - t_lp0

    rel_ch <- if (is.finite(lp_new) && is.finite(lp_old)) {
      (lp_new - lp_old) / max(abs(lp_old), 1)
    } else {
      NA_real_
    }
    cat(sprintf("  Iter %2d | log-post = %12.2f | rel delta = %+.3e\n",
                iter, lp_new, rel_ch))
    if (timing_diagnostics) {
      cat(sprintf("    step time | Mu=%.2fs  M=%.2fs  ll_cur=%.2fs  BetaLam=%.2fs  merge=%.2fs  Gamma=%.2fs  Xi=%.2fs  sigma^2+kappa^2=%.2fs  logpost=%.2fs  total=%.2fs\n",
                  t_mu, t_m, t_data, t_bl, t_merge, t_g, t_xi, t_s, t_lp, proc.time()[3] - iter_t0))
      kappa_ref <- if (kappa_prior == "ig") d_hyp / (c_hyp + 1) else kappa_hc_scale
      cat(sprintf("      kappa | min=%.4g q25=%.4g med=%.4g q75=%.4g max=%.4g  (%s ref=%.4g)\n",
                  min(kappa_sq_vec), quantile(kappa_sq_vec, 0.25), median(kappa_sq_vec),
                  quantile(kappa_sq_vec, 0.75), max(kappa_sq_vec),
                  if (kappa_prior == "ig") "prior_mode" else "hc_scale^2", kappa_ref))
      if (identical(stage1_m_update, "species_minibatch") && !is.null(bb_m_diag)) {
        bb_timing <- as.numeric(bb_m_diag$timing)
        names(bb_timing) <- names(bb_m_diag$timing)
        cat(sprintf("      M minibatch | blocks=%d  species=%d  Lhat_cpu=%.3fs  last_step=%.3g  ls_trials=%d\n",
                    ceiling(ncol(Y_sub_num) / M_species_batch_size),
                    bb_m_diag$n_species, bb_m_diag$lhat_time,
                    bb_m_diag$step, bb_m_diag$ls_trials))
        cat(sprintf("      M TR diag    | %s\n", format_m_tr_diag(bb_m_diag)))
        cat(sprintf("      M coord diag | %s\n",
                    format_latent_coord_diag(M, Beta_Lambda[, q + seq_len(k), drop = FALSE])))
        cat(sprintf("      M similarity | %s\n",
                    format_latent_similarity_diag(M, Beta_Lambda[, q + seq_len(k), drop = FALSE])))
        cat(sprintf("      M timing    | %s\n", format_m_timing(bb_timing)))
      }
    }
    if (!is.finite(lp_new)) {
      stop("Non-finite backbone log-posterior encountered in FLAIR_plus_minibatch_wrapper().")
    }
    if (is.finite(rel_ch) && rel_ch >= 0 && rel_ch < loss_tol) {
      cat("  Converged.\n")
      break
    }
    # Divergence guard: stop early if log-post worsens 2+ consecutive iterations.
    if (is.finite(rel_ch) && rel_ch < -0.02) {
      bb_n_diverge <- bb_n_diverge + 1L
      if (bb_n_diverge >= 2L) {
        cat("  Backbone log-posterior declined 2 consecutive iterations; stopping early.\n")
        break
      }
    } else {
      bb_n_diverge <- 0L
    }
    lp_old <- lp_new
  }

  # Post-backbone M health check.  If M exploded (due to Lambda-cap zeroing M
  # and species-minibatch gradient steps overshooting), re-draw M from its prior
  # N(0, m_prior_var·I_k).  This gives a neutral, non-saturated starting point
  # for Stage 2 rather than inheriting a collapsed M with sdM ~ 3000.
  # Only condition: max|M| > 100 (prior sd ≈ 0.35, so sdM > 100 is clearly broken).
  if (max(abs(M)) > 100) {
    cat(sprintf(
      "  [M health] max|M|=%.1f > 100; M collapsed during backbone — re-initialising from N(0, %.4g·I_k).\n",
      max(abs(M)), m_prior_var))
    M <- matrix(rnorm(nrow(M) * k, sd = sqrt(m_prior_var)), nrow(M), k)
  }

  backbone <- list(
    Beta = Beta_Lambda[, seq_len(q), drop = FALSE],
    Lambda = Beta_Lambda[, q + seq_len(k), drop = FALSE],
    Beta_Lambda = Beta_Lambda,
    M = M,
    Gamma = Gamma,
    Xi_mat = Xi_mat,
    Phi_mat = Phi_mat,
    taxonomy_codes = taxonomy_codes,
    tau_xi   = tau_xi,
    xi_gamma = xi_gamma,
    sigma_sq = sigma_sq_vec,   # (q+k): per-dimension
    kappa_sq = kappa_sq_vec,   # (q+k): per-row of Gamma
    a = a, b = b, c_hyp = c_hyp, d_hyp = d_hyp,
    kappa_prior = kappa_prior, kappa_hc_scale = kappa_hc_scale
  )
  if (is.null(Y_all) || is.null(Xi_all)) {
    # post_process_plus only needed for final output; skip when Stage 2 follows
    if (do_post_process) {
      pp <- post_process_plus(X, M, Beta_Lambda, q)
      Beta_Lambda <- pp$Beta_Lambda
      M <- pp$M
      rp <- realign_plus_prior(Beta_Lambda, M, Gamma, Xi_mat,
                               sigma_sq_vec, kappa_sq_vec,
                               a, b, c_hyp, d_hyp, kappa_prior, kappa_hc_scale,
                               kappa_ub)
      Gamma <- rp$Gamma
      sigma_sq_vec <- rp$sigma_sq
      kappa_sq_vec <- rp$kappa_sq
      backbone <- list(
        Beta = Beta_Lambda[, seq_len(q), drop = FALSE],
        Lambda = Beta_Lambda[, q + seq_len(k), drop = FALSE],
        Beta_Lambda = Beta_Lambda,
        M = M,
        Gamma = Gamma,
        Xi_mat = Xi_mat,
        Phi_mat = Phi_mat,
        taxonomy_codes = taxonomy_codes,
        tau_xi   = tau_xi,
        xi_gamma = xi_gamma,
        sigma_sq = sigma_sq_vec,
        kappa_sq = kappa_sq_vec,
        a = a, b = b, c_hyp = c_hyp, d_hyp = d_hyp,
        kappa_prior = kappa_prior, kappa_hc_scale = kappa_hc_scale
      )
    }
    return(backbone)
  }

  # Stage 2: expand to all species using the backbone fit as the warm start.
  Y_num <- as_numeric_binary(Y_all)
  full_obs_all <- is_full_observed(observed_all)
  obs_num <- if (full_obs_all) NULL else as_numeric_binary(observed_all)

  if (identical(stage2_m_update, "all")) {
    rm(Y_sub_num, obs_sub_num)
    invisible(gc(FALSE))
  }

  bb_idx <- match(colnames(Y), colnames(Y_all))
  if (anyNA(bb_idx)) {
    stop("Backbone species columns from Y were not found in Y_all.")
  }
  if (xi_trainable && is.null(taxonomy_codes_all))
    stop("taxonomy_codes_all is required for all-species xi training.")
  if (xi_trainable) {
    Xi_all[, bb_idx] <- backbone$Xi_mat
    Phi_all[, bb_idx] <- backbone$Phi_mat
  }
  Beta_Lambda <- t(backbone$Gamma %*% Xi_all)
  Beta_Lambda[bb_idx, ] <- backbone$Beta_Lambda
  M            <- backbone$M
  # Apply same Lambda column-norm cap used at backbone init.  Focal species inherit
  # Xi_all norms which can reach ~1e9 at large H, blowing up Lambda and stalling M.
  {
    q_s2     <- ncol(X)
    Lam_s2   <- Beta_Lambda[, q_s2 + seq_len(k), drop = FALSE]
    ln_s2    <- sqrt(colSums(Lam_s2^2))
    cap_s2   <- 5 * sqrt(nrow(M))
    nc_s2    <- ln_s2 > cap_s2
    if (any(nc_s2)) {
      sf_s2 <- ifelse(nc_s2, cap_s2 / ln_s2, 1)
      Beta_Lambda[, q_s2 + seq_len(k)] <- Lam_s2 %*% diag(sf_s2, nrow = k)
      M <- M %*% diag(sf_s2, nrow = k)
      cat(sprintf("  [stage2 init] Lambda norm cap (cap=%.0f): norms %s → %s\n",
                  cap_s2,
                  paste(round(ln_s2, 1), collapse = "/"),
                  paste(round(ln_s2 * sf_s2, 1), collapse = "/")))
    }
  }
  Gamma        <- backbone$Gamma
  sigma_sq_vec <- backbone$sigma_sq   # (q+k): same for all species, no per-species expansion
  kappa_sq_vec <- backbone$kappa_sq   # (q+k): same
  if (!is.null(sigma_b_train)) b <- as.numeric(sigma_b_train)[1]
  if (!is.finite(b) || b <= 0)
    stop("sigma_b_train must be a positive finite value.")
  sigma_sq_vec <- update_sigma_sq_cpp(
    Beta_Lambda, Gamma, Xi_all, a, b, 0.01, 100
  )

  ord_base <- order(colSums(Y_all), decreasing = TRUE)
  idx_batches <- split(ord_base, ceiling(seq_along(ord_base) / BL_species_batch_size))
  effective_workers <- if (has_openmp) min(worker_cores, length(idx_batches)) else 1L
  X_tilde <- make_x_tilde(X, M)
  Mu <- t(Gamma %*% Xi_all)
  # Cache transposed embedding matrices (p×H) to avoid repeated large allocations.
  # Phi_t_all: fixed throughout stage 2 (Phi is the prior centre, never updated).
  # Xi_t_all:  refreshed once per iteration after the xi CG update, reused in the
  #             same iteration by xi_prior_term_cpp and update_xi_hyperparams_R.
  Phi_t_all <- if (xi_trainable) t(Phi_all) else NULL
  xi_spec   <- if (xi_trainable) xi_laplacian_spectrum(taxonomy_codes_all) else NULL
  # [L_norm]_jj = 1 unless species j shares no taxonomy level (top-level singleton)
  xi_has_rel <- if (xi_trainable) { top <- taxonomy_codes_all[, 1]; tabulate(top)[top] > 1 } else NULL
  es <- NULL  # V_diag cache, refreshed every vdiag_freq iterations
  S_V_em <- NULL  # sum_j Var(xi_j) from the trait E-step (xi_hyper_em only)
  Xi_t_all  <- if (xi_trainable) t(Xi_all)  else NULL
  ll_cur <- compute_data_term_cpp(
    Y_num,
    if (is.null(obs_num)) matrix(0, 0, 0) else obs_num,
    is.null(obs_num),
    X_tilde,
    Beta_Lambda
  )
  lp_bl_cur  <- compute_beta_prior_term_cpp(Beta_Lambda, Mu, sigma_sq_vec)
  lp_M_cur   <- compute_M_prior_term(M, m_prior_var)
  lp_G_cur   <- compute_Gamma_prior_term(Gamma, kappa_sq_vec)
  lp_sig_cur <- compute_sigma_prior_term(sigma_sq_vec, a, b)
  lp_kap_cur <- compute_kappa_prior_term(kappa_sq_vec, kappa_prior, c_hyp, d_hyp, kappa_hc_scale)
  lp_xi_cur <- if (xi_trainable) {
    xi_prior_term_cpp(Xi_all, Phi_all, taxonomy_codes_all,
                      tau_vec, xi_gamma,
                      Xi_t = Xi_t_all, Phi_t = Phi_t_all) +
      compute_xi_hyperparam_prior(tau_xi, xi_gamma,
                                  hyper_xi_prior, xi_e, xi_f, xi_g, xi_h,
                                  xi_s, xi_s_gamma) +
      xi_logdet_term(tau_xi, xi_gamma, xi_spec, nrow(Xi_all),
                     xi_post_eig(Gamma, sigma_sq_vec))
  } else 0
  lp_old <- ll_cur + lp_bl_cur + lp_M_cur + lp_G_cur + lp_sig_cur + lp_kap_cur + lp_xi_cur
  cat(sprintf("FLAIR+ minibatch all-species | n=%d  p=%d  q=%d  k=%d  h=%d  BL_species_batch=%d\n",
              nrow(Y_num), ncol(Y_num), q, k, nrow(Xi_all), BL_species_batch_size))
  cat(sprintf("  Sigma_sq prior: IG(a=%.4g, b=%.4g)\n", a, b))
  cat(sprintf("  M prior: eta_i ~ N(0, %.4g I_k); E||eta_i||^2 = %.4g\n",
              m_prior_var, k * m_prior_var))
  if (xi_trainable) {
    cat(sprintf("  Xi update: tau_xi=%.4g (shared)  gamma=%.4g  prior=%s  maxit=%d\n",
                tau_xi, xi_gamma, hyper_xi_prior, xi_maxit))
    if (xi_freeze_iters > 0L)
      cat(sprintf("  Xi freeze: Xi fixed for first %d outer iterations (M+BL+Gamma settle first).\n",
                  xi_freeze_iters))
  }
  cat(sprintf("  Stage2 M update = %s%s\n",
              stage2_m_update,
              if (identical(stage2_m_update, "backbone"))
                sprintf(" columns (%d/%d)", length(bb_idx), ncol(Y_num))
              else if (identical(stage2_m_update, "species_minibatch"))
                sprintf(" columns (global species blocks, block=%d)",
                        M_species_batch_size)
              else " columns"))
  print_runtime_diagnostics(cores, effective_workers, BL_species_batch_size)
  cat(sprintf("  Block workers used = %d%s\n",
              effective_workers,
              if (effective_workers > 1L) "" else " (sequential)"))
  cat(sprintf("  Init  | log-post = %12.2f\n", lp_old))
  all_delta    <- 1.0   # warm-start delta for all-species M updates
  m_step_size2 <- m_step_size  # warm-start for species_minibatch delta

  # Xi update frequency: update Xi every xi_update_freq iters; skip (reuse Mu)
  # on others; freeze Xi entirely after xi_late_freeze iterations.
  .xi_freq <- max(1L, as.integer(xi_update_freq))
  .xi_late_freeze <- max(0L, as.integer(xi_late_freeze))
  if (.xi_late_freeze > 0L)
    cat(sprintf("  [xi-late-freeze] Xi will be frozen after iter %d\n", .xi_late_freeze))

  for (iter in seq_len(iter_max_train)) {
    iter_t0 <- proc.time()[3]

    t_mu <- 0
    ord_iter <- build_batch_order_cpp(ord_base, randomize_batches && iter > 1L)

    m_idx <- if (identical(stage2_m_update, "backbone")) bb_idx else seq_len(nrow(Beta_Lambda))
    Beta_m   <- Beta_Lambda[m_idx, seq_len(q),     drop = FALSE]
    Lambda_m <- Beta_Lambda[m_idx, q + seq_len(k), drop = FALSE]

    if (iter == 1L) {
      .Y_m   <- if (identical(stage2_m_update, "backbone")) Y_sub_num else Y_num
      .obs_m <- if (identical(stage2_m_update, "backbone"))
        { if (full_obs_sub) matrix(0, 0, 0) else obs_sub_num }
      else
        { if (full_obs_all) matrix(0, 0, 0) else obs_num }
      cat(sprintf("  [ref] ||grad_M(M0)|| = %.4g  (Lin 2008 suggested Δ₀; current Δ₀ = %.4g)\n",
                  m_grad_norm_cpp(.Y_m, .obs_m, X, Lambda_m, Beta_m, M, m_prior_var),
                  all_delta))
      rm(.Y_m, .obs_m)
    }
    t_m0 <- proc.time()[3]
    m_lhat_time <- 0
    m_lhat_n <- 0L
    m_lhat_step <- NA_real_
    m_lhat_trials <- 0L
    m_batches <- 0L
    m_timing <- numeric(0)
    if (identical(stage2_m_update, "backbone")) {
      all_res <- if (full_obs_sub) {
        map_latent_factors_shared_hessian_fullobs(
          Y_sub_num, X, Lambda_m, Beta_m, M,
          prior_var = m_prior_var, epsilon = tol,
          n_threads = cores, m_block = m_block, step_init = all_delta
        )
      } else {
        map_latent_factors_shared_hessian(
          Y_sub_num, obs_sub_num, X, Lambda_m, Beta_m, M,
          prior_var = m_prior_var, epsilon = tol,
          n_threads = cores, m_block = m_block, step_init = all_delta
        )
      }
      M         <- all_res$M
      all_delta <- all_res$step_out
    } else if (identical(stage2_m_update, "species_minibatch")) {
      m_sord <- if (randomize_batches && iter > 1L) {
        if (M_species_batch_size == BL_species_batch_size)
          ord_iter
        else
          build_batch_order_cpp(ord_base, TRUE)
      } else {
        NULL
      }
      m_res <- if (full_obs_all) {
        map_latent_factors_shared_hessian_species_blocks_fullobs(
          Y_num, X, Lambda_m, Beta_m, M,
          prior_var = m_prior_var, epsilon = tol,
          n_threads = cores, species_block = M_species_batch_size,
          step_init = m_step_size2, max_ls = m_lhat_max_ls,
          species_order_r = if (!is.null(m_sord)) as.integer(m_sord) else NULL,
          ll_init = ll_cur
        )
      } else {
        map_latent_factors_shared_hessian_species_blocks(
          Y_num, obs_num, X, Lambda_m, Beta_m, M,
          prior_var = m_prior_var, epsilon = tol,
          n_threads = cores, species_block = M_species_batch_size,
          step_init = m_step_size2, max_ls = m_lhat_max_ls,
          species_order_r = if (!is.null(m_sord)) as.integer(m_sord) else NULL,
          ll_init = ll_cur
        )
      }
      M            <- m_res$M
      m_step_size2 <- m_res$step_out   # warm-start for next iter
      m_lhat_time  <- m_res$lhat_time
      m_lhat_n     <- m_res$n_species
      m_lhat_step  <- m_res$step
      m_lhat_trials <- m_res$ls_trials
      m_batches    <- ceiling(nrow(Beta_Lambda) / M_species_batch_size)
      m_timing     <- as.numeric(m_res$timing)
      names(m_timing) <- names(m_res$timing)
    } else {
      all_res <- if (full_obs_all) {
        map_latent_factors_shared_hessian_fullobs(
          Y_num, X, Lambda_m, Beta_m, M,
          prior_var = m_prior_var, epsilon = tol,
          n_threads = cores, m_block = m_block, step_init = all_delta
        )
      } else {
        map_latent_factors_shared_hessian(
          Y_num, obs_num, X, Lambda_m, Beta_m, M,
          prior_var = m_prior_var, epsilon = tol,
          n_threads = cores, m_block = m_block, step_init = all_delta
        )
      }
      M         <- all_res$M
      all_delta <- all_res$step_out
    }
    t_m <- proc.time()[3] - t_m0
    X_tilde <- update_x_tilde_M(X_tilde, X, M)
    t_data0 <- proc.time()[3]
    if (identical(stage2_m_update, "species_minibatch") && is.finite(m_res$ll)) {
      ll_cur <- m_res$ll
    } else {
      ll_cur <- compute_data_term_cpp(
        Y_num,
        if (is.null(obs_num)) matrix(0, 0, 0) else obs_num,
        is.null(obs_num),
        X_tilde,
        Beta_Lambda
      )
    }
    t_data <- proc.time()[3] - t_data0
    lp_M_cur <- compute_M_prior_term(M, m_prior_var)
    cat(sprintf("    [all-sp M]  ll_data=%12.2f  lp_M=%12.2f  subtotal=%12.2f\n",
                ll_cur, lp_M_cur, ll_cur + lp_M_cur + lp_bl_cur))

    t_bl0 <- proc.time()[3]
    blk_res <- map_beta_lambda_plus_blocks_cpp(
      Y_num,
      if (is.null(obs_num)) matrix(0, 0, 0) else obs_num,
      is.null(obs_num),
      X_tilde,
      Beta_Lambda,
      Mu,
      sigma_sq_vec,
      as.integer(ord_iter),
      BL_species_batch_size,
      C_clip,
      step_cap,
      effective_workers
    )
    t_bl <- proc.time()[3] - t_bl0

    t_merge0 <- proc.time()[3]
    Beta_Lambda <- blk_res$Beta_Lambda
    ll_cur <- ll_cur + sum(blk_res$delta_ll)
    lp_bl_cur <- lp_bl_cur + sum(blk_res$delta_prior)
    t_merge <- proc.time()[3] - t_merge0
    cat(sprintf("    [all-sp BL] ll_data=%12.2f  lp_BL=%12.2f  subtotal=%12.2f\n",
                ll_cur, lp_bl_cur, ll_cur + lp_M_cur + lp_bl_cur))

    t_g0 <- proc.time()[3]
    set_blas_threads_cpp(worker_cores)  # R-level GEMMs below (Gamma, Mu, sigma EM)
    # One V_diag refresh per vdiag_freq iterations, consumed by the sigma^2 EM
    # update below (standard EM; stale between recomputes).
    es_fresh <- is.null(es) || iter %% vdiag_freq == 0L
    if (es_fresh)
      es <- list(V_diag = compute_V_diag_from_BL_cpp(X_tilde, Beta_Lambda,
                   if (is.null(obs_num)) matrix(0, 0, 0) else obs_num,
                   is.null(obs_num), 1 / as.numeric(sigma_sq_vec), worker_cores))
    Gamma <- .gamma_fn(Beta_Lambda, Xi_all, sigma_sq_vec, kappa_sq_vec)
    if (!xi_trainable) Mu <- t(Gamma %*% Xi_all)
    t_g <- proc.time()[3] - t_g0

    t_xi0 <- proc.time()[3]
    xi_skipped <- FALSE
    if (xi_trainable) {
      if (iter > xi_freeze_iters &&
          ((.xi_freq > 1L && iter %% .xi_freq != 0L) ||
           (.xi_late_freeze > 0L && iter > .xi_late_freeze))) {
        # Skip full Xi update this iter; Gamma just changed, so recompute Mu only.
        xi_skipped <- TRUE
        Mu <- t(Gamma %*% Xi_all)
      } else if (iter <= xi_freeze_iters) {
        # Xi frozen: M and BL are still settling after the backbone→all-species
        # transition.  Updating Xi against poorly-initialised focal-species Lambda
        # would corrupt the embedding before it has a chance to be meaningful.
        # Update Mu from the freshly updated Gamma (Xi unchanged) so that the
        # Beta_Lambda prior term is computed with the correct mean.
        Mu <- t(Gamma %*% Xi_all)
        lp_xi_cur <- xi_prior_term_cpp(Xi_all, Phi_all, taxonomy_codes_all,
                                       tau_vec, xi_gamma,
                                       Xi_t = Xi_t_all, Phi_t = Phi_t_all) +
          compute_xi_hyperparam_prior(tau_xi, xi_gamma,
                                      hyper_xi_prior, xi_e, xi_f, xi_g, xi_h,
                                      xi_s, xi_s_gamma) +
          xi_logdet_term(tau_xi, xi_gamma, xi_spec, nrow(Xi_all),
                     xi_post_eig(Gamma, sigma_sq_vec))
        cat(sprintf("      Xi frozen (iter %d/%d): tau_xi=%.4g gamma=%.4g\n",
                    iter, xi_freeze_iters, tau_xi, xi_gamma))
      } else {
        xi_res <- update_xi_step(
          Beta_Lambda, Gamma, sigma_sq_vec, Xi_all, Phi_all,
          taxonomy_codes_all, tau_vec, xi_gamma,
          xi_maxit, xi_tol, cores,
          Xi_t = Xi_t_all, Phi_t = Phi_t_all
        )
        Xi_all   <- xi_res$Xi_mat
        Xi_t_all <- xi_res$Xi_t    # p×H from CG; avoids t(Xi_all) in subsequent calls
        if (trait_intercept) { Xi_all[nrow(Xi_all), ] <- 1; Xi_t_all[, ncol(Xi_t_all)] <- 1 }
        hyp <- if (xi_hyper_em) {
          hf <- seq_len(nrow(Xi_all) - trait_intercept)
          ex <- trait_estep_cpp(X_tilde, Beta_Lambda,
                                if (is.null(obs_num)) matrix(0, 0, 0) else obs_num, is.null(obs_num),
                                1 / as.numeric(sigma_sq_vec), Gamma[, hf, drop = FALSE],
                                2 * (tau_xi + xi_gamma * xi_has_rel), worker_cores)
          S_V_em <- matrix(0, nrow(Xi_all), nrow(Xi_all)); S_V_em[hf, hf] <- ex$S_xi
          update_xi_hyperparams_em(Xi_t_all[, hf, drop = FALSE], t(Phi_all[hf, , drop = FALSE]),
                                   taxonomy_codes_all, xi_spec, sum(diag(ex$S_xi)),
                                   xi_e, xi_f, xi_g, xi_h, tau_xi, xi_gamma, hyper_xi_prior,
                                   xi_s, xi_s_gamma)
        } else update_xi_hyperparams_R(Xi_all, Phi_all, taxonomy_codes_all, xi_spec,
                                       xi_e, xi_f, xi_g, xi_h,
                                       tau_xi, xi_gamma, hyper_xi_prior,
                                       xi_s, xi_s_gamma,
                                       Xi_t = Xi_t_all, Phi_t = Phi_t_all,
                                       a = xi_post_eig(Gamma, sigma_sq_vec))
        tau_xi   <- hyp$tau_xi
        tau_vec  <- rep(tau_xi, nrow(Xi_all))  # rebuild H-vector for C++ calls
        xi_gamma <- hyp$xi_gamma
        Mu <- t(Gamma %*% Xi_all)   # single Mu after both Gamma and Xi settled
        lp_xi_cur <- xi_prior_term_cpp(Xi_all, Phi_all, taxonomy_codes_all,
                                       tau_vec, xi_gamma,
                                       Xi_t = Xi_t_all, Phi_t = Phi_t_all) +
          compute_xi_hyperparam_prior(tau_xi, xi_gamma,
                                      hyper_xi_prior, xi_e, xi_f, xi_g, xi_h,
                                      xi_s, xi_s_gamma) +
          xi_logdet_term(tau_xi, xi_gamma, xi_spec, nrow(Xi_all),
                     xi_post_eig(Gamma, sigma_sq_vec))
      }
    }
    t_xi <- proc.time()[3] - t_xi0

    t_s0 <- proc.time()[3]
    sigma_sq_vec <- update_sigma_xi_em(Beta_Lambda, Gamma, Xi_all, es$V_diag,
                                       sigma_sq_vec, kappa_sq_vec, a, b,
                                       ub = sigma_em_ub, Mu = Mu, S_V = S_V_em)
    lp_bl_cur  <- compute_beta_prior_term_cpp(Beta_Lambda, Mu, sigma_sq_vec)
    lp_sig_cur <- compute_sigma_prior_term(sigma_sq_vec, a, b)
    kappa_sq_vec <- if (kappa_prior == "ig")
      update_kappa_sq_cpp(Gamma, c_hyp, d_hyp, 0.01, kappa_ub) else
      update_kappa_sq_halfcauchy_cpp(Gamma, kappa_hc_scale, 0.01, kappa_ub)
    lp_G_cur   <- compute_Gamma_prior_term(Gamma, kappa_sq_vec)
    lp_kap_cur <- compute_kappa_prior_term(kappa_sq_vec, kappa_prior, c_hyp, d_hyp, kappa_hc_scale)
    t_s <- proc.time()[3] - t_s0

    t_lp0 <- proc.time()[3]
    lp_new <- ll_cur + lp_bl_cur + lp_M_cur + lp_G_cur + lp_sig_cur + lp_kap_cur + lp_xi_cur
    t_lp <- proc.time()[3] - t_lp0

    rel_ch <- if (is.finite(lp_new) && is.finite(lp_old)) {
      (lp_new - lp_old) / max(abs(lp_old), 1)
    } else {
      NA_real_
    }
    cat(sprintf("  Iter %2d | log-post = %12.2f | rel delta = %+.3e\n",
                iter, lp_new, rel_ch))
    if (timing_diagnostics) {
      cat(sprintf("    step time | Mu=%.2fs  M=%.2fs  ll_cur=%.2fs  BetaLam=%.2fs  merge=%.2fs  Gamma=%.2fs  Xi=%.2fs  sigma^2+kappa^2=%.2fs  logpost=%.2fs  total=%.2fs\n",
                  t_mu, t_m, t_data, t_bl, t_merge, t_g, t_xi, t_s, t_lp, proc.time()[3] - iter_t0))
      kappa_ref <- if (kappa_prior == "ig") d_hyp / (c_hyp + 1) else kappa_hc_scale
      cat(sprintf("      kappa | min=%.4g q25=%.4g med=%.4g q75=%.4g max=%.4g  (%s ref=%.4g)\n",
                  min(kappa_sq_vec), quantile(kappa_sq_vec, 0.25), median(kappa_sq_vec),
                  quantile(kappa_sq_vec, 0.75), max(kappa_sq_vec),
                  if (kappa_prior == "ig") "prior_mode" else "hc_scale^2", kappa_ref))
      if (xi_trainable) {
        if (xi_skipped) {
          cat(sprintf("      Xi diag | SKIPPED (xi_freq=%d, iter%%freq=%d) Mu recomputed from Gamma\n",
                      .xi_freq, iter %% .xi_freq))
        } else if (iter <= xi_freeze_iters) {
          cat(sprintf("      Xi diag | FROZEN (%d/%d) tau_xi=%.4g gamma=%.4g lp_Xi=%12.2f\n",
                      iter, xi_freeze_iters, tau_xi, xi_gamma, lp_xi_cur))
        } else {
          cat(sprintf("      Xi diag | iter=%d rel_resid=%.3g converged=%s tau_xi=%.4g gamma=%.4g lp_Xi=%12.2f S_xi=%.4g\n",
                      xi_res$iter, xi_res$rel_resid,
                      if (isTRUE(xi_res$converged)) "TRUE" else "FALSE",
                      tau_xi, xi_gamma, lp_xi_cur, hyp$S_xi))
          if (!is.null(hyp$S_vec))
            cat(sprintf("      per-dim S | min=%.4g q25=%.4g med=%.4g q75=%.4g max=%.4g\n",
                        min(hyp$S_vec), quantile(hyp$S_vec, 0.25), median(hyp$S_vec),
                        quantile(hyp$S_vec, 0.75), max(hyp$S_vec)))
        }
      }
      if (identical(stage2_m_update, "species_minibatch")) {
        cat(sprintf("      M minibatch | blocks=%d  species=%d  Lhat_cpu=%.3fs  last_step=%.3g  ls_trials=%d\n",
                    m_batches, m_lhat_n, m_lhat_time, m_lhat_step, m_lhat_trials))
        cat(sprintf("      M TR diag    | %s\n", format_m_tr_diag(m_res)))
        cat(sprintf("      M coord diag | %s\n",
                    format_latent_coord_diag(M, Beta_Lambda[, q + seq_len(k), drop = FALSE])))
        cat(sprintf("      M similarity | %s\n",
                    format_latent_similarity_diag(M, Beta_Lambda[, q + seq_len(k), drop = FALSE])))
        cat(sprintf("      M timing    | %s\n", format_m_timing(m_timing)))
      }
    }
    if (!is.finite(lp_new)) {
      stop("Non-finite all-species log-posterior encountered in FLAIR_plus_minibatch_wrapper().")
    }
    # Only declare convergence on an iteration whose E-step is fresh: with a
    # stale E-step sigma^2 barely moves and the small lp delta is an artifact.
    if (is.finite(rel_ch) && rel_ch >= 0 && rel_ch < loss_tol && es_fresh) {
      cat("  Converged.\n")
      break
    }
    lp_old <- lp_new
  }

  map <- NULL
  if (do_post_process) {
    map <- list(Beta_Lambda = Beta_Lambda, M = M, Gamma = Gamma)   # MAP before post-processing, for UQ
    pp <- post_process_plus(X, M, Beta_Lambda, q)
    Beta_Lambda <- pp$Beta_Lambda
    M <- pp$M
    # The prior mean Gamma xi_j moves with b~_j: Gamma -> T Gamma (no refit, so the
    # prior is not re-optimized after the rotation).
    Gamma <- pp$T %*% Gamma
  }

  list(
    Beta = Beta_Lambda[, seq_len(q), drop = FALSE],
    Lambda = Beta_Lambda[, q + seq_len(k), drop = FALSE],
    Beta_Lambda = Beta_Lambda,
    M = M,
    Gamma = Gamma,
    Xi_mat = Xi_all,
    Phi_mat = Phi_all,
    taxonomy_codes = taxonomy_codes_all,
    tau_xi   = tau_xi,
    xi_gamma = xi_gamma,
    xi_spec  = xi_spec,
    sigma_sq = sigma_sq_vec,   # (q+k): per-dimension
    kappa_sq = kappa_sq_vec,   # (q+k): per-row of Gamma
    a = a, b = b, c_hyp = c_hyp, d_hyp = d_hyp,
    kappa_prior = kappa_prior, kappa_hc_scale = kappa_hc_scale,
    trait_intercept = trait_intercept, mu_prior_var = mu_prior_var,
    backbone = backbone,
    map = map
  )
}
