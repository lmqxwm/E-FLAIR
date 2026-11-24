################################################################################
# flair_plus_minibatch_tr.R
#
# Minibatch FLAIR+ variant with true trust-region M updates:
#   * sources helper_functions_trustregion.cpp (rho-ratio TR, warm-start δ)
#   * fits a standard backbone subset first, then expands to all species
#   * keeps one reusable X_tilde matrix
#   * avoids extra full-matrix copies for binary inputs
#   * sorts species by prevalence and updates them in blocks
#   * uses rho-ratio TR: expand delta when model is accurate, shrink when poor
################################################################################

library(parallel)
library(MASS)
library(matrixStats)
library(rsvd)
library(Rcpp)

FLAIR_CODE_DIR <- Sys.getenv("FLAIR_CODE_DIR",
  unset = tryCatch(dirname(sys.frame(1)$ofile), error = function(e) getwd()))
sourceCpp(file.path(FLAIR_CODE_DIR, "helper_functions_trustregion.cpp"))
source(file.path(FLAIR_CODE_DIR, "uq_scalable.R"))   # xi_logdet_hessian, compute_UQ_scalable

sigmoid <- function(x) 1 / (1 + exp(-x))

softplus_fp <- function(x) {
  ifelse(x > 35, x, ifelse(x < -35, exp(x), log1p(exp(x))))
}

safe_logit <- function(p, eps = 1e-6) {
  p <- pmax(pmin(p, 1 - eps), eps)
  log(p / (1 - p))
}

limit_embedding_rows <- function(Xi_mat, Xi_all = NULL, max_embedding_dim = 20L) {
  max_embedding_dim <- as.integer(max_embedding_dim)
  if (is.na(max_embedding_dim) || max_embedding_dim <= 0L)
    return(list(Xi_mat = Xi_mat, Xi_all = Xi_all))

  h_keep <- min(max_embedding_dim, nrow(Xi_mat))
  if (h_keep < nrow(Xi_mat)) {
    cat(sprintf("  Embedding rows capped: h=%d -> %d\n", nrow(Xi_mat), h_keep))
    Xi_mat <- Xi_mat[seq_len(h_keep), , drop = FALSE]
    if (!is.null(Xi_all))
      Xi_all <- Xi_all[seq_len(min(h_keep, nrow(Xi_all))), , drop = FALSE]
  }
  list(Xi_mat = Xi_mat, Xi_all = Xi_all)
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

compute_log_post_plus <- function(Y, obs, X_tilde, Beta_Lambda, M,
                                  Gamma, Xi_mat,
                                  sigma_sq_vec,   # (q+k): per-dimension
                                  a, b,
                                  kappa_sq_vec,   # (q+k): per-row of Gamma
                                  c_hyp, d_hyp,
                                  m_prior_var = 1) {
  Z <- X_tilde %*% t(Beta_Lambda)
  obs_01 <- if (is.null(obs)) TRUE else (obs == 1)
  ll <- sum((Y[obs_01] * Z[obs_01]) - softplus_fp(Z[obs_01]))
  Mu     <- t(Gamma %*% Xi_mat)
  Delta  <- Beta_Lambda - Mu
  p      <- nrow(Beta_Lambda)
  Delta2 <- sweep(Delta^2, 2, sigma_sq_vec, "/")
  lp_bl  <- -0.5 * sum(Delta2) - 0.5 * p * sum(log(sigma_sq_vec))
  lp_M   <- compute_M_prior_term(M, m_prior_var)
  gamma_ss <- rowSums(Gamma^2)
  h <- ncol(Gamma)
  lp_G   <- -0.5 * sum(gamma_ss / kappa_sq_vec) -
            0.5 * h * sum(log(kappa_sq_vec))
  lp_sig <- sum(a * log(b) - lgamma(a) - (a + 1) * log(sigma_sq_vec) - b / sigma_sq_vec)
  lp_kap <- sum(c_hyp * log(d_hyp) - lgamma(c_hyp) -
                (c_hyp + 1) * log(kappa_sq_vec) - d_hyp / kappa_sq_vec)
  ll + lp_bl + lp_M + lp_G + lp_sig + lp_kap
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

compute_kappa_prior_term <- function(kappa_sq_vec, c_hyp, d_hyp) {
  sum(c_hyp * log(d_hyp) - lgamma(c_hyp) - (c_hyp + 1) * log(kappa_sq_vec) - d_hyp / kappa_sq_vec)
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
                          gamma_ridge = 1e-6) {
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
  # Gamma above was fit to BL (OLS ridge), so Mu ≈ BL, Delta ≈ 0, and any
  # residual-based formula gives sigma_sq ≈ b/(a+p/2+1) ≈ 0.03 (near zero).
  # With 1/sigma_sq ≈ 33, the BL prior drowns the likelihood in every subsequent
  # update: BL never moves from Mu, residuals stay zero, and sigma_sq is trapped
  # in a degenerate fixed point.  Start at the prior mean instead (b/(a-1) = 1.0
  # for the default a=3, b=2) so the first BL update is data-driven.
  sigma_sq <- rep(pmax(sigma_lb, pmin(sigma_ub, 2.0 / (3.0 - 1.0))),
                  ncol(Beta_Lambda))   # = 1.0 for IG(3,2) prior mean
  kappa_sq <- pmax(pmin(rowMeans(Gamma^2), kappa_ub), kappa_lb)   # (q+k): per-row of Gamma
  if (!all(is.finite(Beta_Lambda)) || !all(is.finite(M_hat)) ||
      !all(is.finite(Gamma)) || !all(is.finite(sigma_sq)) ||
      !all(is.finite(kappa_sq))) {
    stop("FLAIR+ initialization produced non-finite values; check embedding scale/rank.")
  }
  list(Beta_Lambda = Beta_Lambda, M = M_hat, Gamma = Gamma,
       sigma_sq = sigma_sq, kappa_sq = kappa_sq,
       a = 3, b = 2, c_hyp = 3, d_hyp = 2)
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
  Lambda_new <- Lambda %*% s_M$v[, seq_len(k), drop = FALSE] %*%
    diag(s_M$d[seq_len(k)], nrow = k) / sqrt(n)
  list(Beta_Lambda = cbind(Beta_new, Lambda_new), M = M_new)
}

# EM-type update of sigma_r^2: the joint-MAP conditional mode uses the shrunken
# residuals only and collapses toward the lower clamp; adding the conditional
# posterior variance V_jr = [H_{b_j|-}^{-1}]_rr restores E[(b_jr - Gamma_r xi_j)^2].
update_sigma_sq_em <- function(Beta_Lambda, Gamma, Xi_mat, X_tilde, obs_num,
                               sigma_sq_vec, a, b, kappa_sq_vec) {
  V_diag <- compute_V_diag_from_BL_cpp(
    X_tilde, Beta_Lambda,
    if (is.null(obs_num)) matrix(0, 0, 0) else obs_num, is.null(obs_num),
    1 / sigma_sq_vec)
  Delta <- Beta_Lambda - t(Gamma %*% Xi_mat)
  # E||Gamma_r xi_j||^2 also carries the posterior variance of Gamma_r: tr(V_{Gamma_r} Xi Xi')
  XX   <- tcrossprod(Xi_mat)
  tr_G <- vapply(seq_along(sigma_sq_vec), function(r)
    sum(diag(solve(XX / sigma_sq_vec[r] + diag(nrow(XX)) / kappa_sq_vec[r], XX))), 0)
  pmin(pmax((b + 0.5 * (colSums(Delta^2 + V_diag) + tr_G)) / (a + nrow(Beta_Lambda) / 2 + 1),
            0.01), 100)
}

realign_plus_prior <- function(Beta_Lambda, M, Gamma, Xi_mat, X_tilde, obs_num,
                               sigma_sq_vec, kappa_sq_vec,
                               a, b, c_hyp, d_hyp) {
  Gamma <- update_Gamma_cpp(Beta_Lambda, Xi_mat, sigma_sq_vec, kappa_sq_vec)
  sigma_sq_vec <- update_sigma_sq_em(Beta_Lambda, Gamma, Xi_mat, X_tilde, obs_num,
                                     sigma_sq_vec, a, b, kappa_sq_vec)
  kappa_sq_vec <- update_kappa_sq_cpp(Gamma, c_hyp, d_hyp, 0.01, 100,
    gamma_post_trace_cpp(tcrossprod(Xi_mat), sigma_sq_vec, kappa_sq_vec, nrow(Xi_mat), 0))
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

sum_named_numeric <- function(items) {
  if (!length(items)) return(numeric(0))
  Reduce(function(a, b) {
    nm <- union(names(a), names(b))
    out <- setNames(numeric(length(nm)), nm)
    out[names(a)] <- out[names(a)] + a
    out[names(b)] <- out[names(b)] + b
    out
  }, items)
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

block_log_post_from_Z <- function(Y_blk, obs_blk, Z, BL_blk, Mu_blk, sigma_sq_vec) {
  if (is.null(obs_blk)) {
    ll <- sum(Y_blk * Z - softplus_fp(Z))
  } else {
    ll <- sum(((Y_blk * Z) - softplus_fp(Z)) * obs_blk)
  }
  Delta  <- BL_blk - Mu_blk
  Delta2 <- sweep(Delta^2, 2, sigma_sq_vec, "/")
  ll - 0.5 * sum(Delta2) - 0.5 * nrow(BL_blk) * sum(log(sigma_sq_vec))
}

update_block_shared_hessian <- function(Y_blk, obs_blk, X_tilde, BL_blk, Mu_blk,
                                        sigma_sq_vec, C_clip, step_cap) {
  eta <- X_tilde %*% t(BL_blk)
  pv <- sigmoid(eta)
  if (is.null(obs_blk)) {
    wt    <- pv * (1 - pv)
    resid <- Y_blk - pv
  } else {
    wt    <- (pv * (1 - pv)) * obs_blk
    resid <- (Y_blk - pv) * obs_blk
  }

  w_bar    <- pmax(rowMeans(wt), 1e-8)
  Xw       <- X_tilde * sqrt(w_bar)
  H_shared <- crossprod(Xw) + diag(1 / sigma_sq_vec)             # per-dimension diagonal
  Delta    <- BL_blk - Mu_blk
  grad_t   <- crossprod(X_tilde, resid) - t(sweep(Delta, 2, sigma_sq_vec, "/"))
  dir_blk  <- t(solve(H_shared, grad_t))

  row_norm <- sqrt(rowSums(dir_blk^2))
  dir_blk  <- sweep(dir_blk, 1, pmin(1, step_cap / pmax(row_norm, 1e-8)), "*")

  obj_old <- block_log_post_from_Z(Y_blk, obs_blk, eta, BL_blk, Mu_blk, sigma_sq_vec)
  step <- 1
  for (ls in 1:20) {
    cand   <- pmax(pmin(BL_blk + step * dir_blk, C_clip), -C_clip)
    Z_cand <- X_tilde %*% t(cand)
    if (block_log_post_from_Z(Y_blk, obs_blk, Z_cand, cand, Mu_blk, sigma_sq_vec) >= obj_old)
      return(cand)
    step <- step / 2
  }
  BL_blk
}

FLAIR_plus_minibatch_wrapper <- function(
    Y, X, Xi_mat, k,
    Y_all = NULL, Xi_all = NULL,
    observed = NA, observed_all = NA,
    iter_max_pretrain = 10,
    iter_max_train = 50,
    tol = 1e-4,
    loss_tol = 1e-3,
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
    max_embedding_dim = 20L,
    init_override = NULL,
    a = 3, b = 2, c_hyp = 3, d_hyp = 2) {

  stage1_m_update <- match.arg(stage1_m_update)
  stage2_m_update <- match.arg(stage2_m_update)
  Xi_limited <- limit_embedding_rows(Xi_mat, Xi_all, max_embedding_dim)
  Xi_mat <- Xi_limited$Xi_mat
  Xi_all <- Xi_limited$Xi_all

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
  Beta_Lambda  <- init$Beta_Lambda
  M            <- init$M
  Gamma        <- init$Gamma
  sigma_sq_vec <- init$sigma_sq   # (q+k): per-dimension
  kappa_sq_vec <- init$kappa_sq   # (q+k): per-row of Gamma

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
  lp_kap_cur <- compute_kappa_prior_term(kappa_sq_vec, c_hyp, d_hyp)
  lp_old <- ll_cur + lp_bl_cur + lp_M_cur + lp_G_cur + lp_sig_cur + lp_kap_cur
  ord_base <- order(colSums(Y), decreasing = TRUE)
  idx_batches <- split(ord_base, ceiling(seq_along(ord_base) / BL_species_batch_size))
  effective_workers <- if (has_openmp) min(worker_cores, length(idx_batches)) else 1L
  cat(sprintf("FLAIR+ minibatch backbone | n=%d  p=%d  q=%d  k=%d  h=%d  BL_species_batch=%d\n",
              nrow(Y_sub_num), ncol(Y_sub_num), q, k, nrow(Xi_mat), BL_species_batch_size))
  cat(sprintf("  M prior: eta_i ~ N(0, %.4g I_k); E||eta_i||^2 = %.4g\n",
              m_prior_var, k * m_prior_var))
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

  if (iter_max_pretrain == 0L && !is.null(Y_all) && !is.null(Xi_all)) {
    cat("  Stage1 pretrain skipped; using SVD backbone initialization for Stage2.
")
  }
  for (iter in seq_len(iter_max_pretrain)) {
    iter_t0 <- proc.time()[3]

    t_mu <- 0
    ord_iter <- build_batch_order_cpp(ord_base, randomize_batches && iter > 1L)

    Beta   <- Beta_Lambda[, seq_len(q),      drop = FALSE]
    Lambda <- Beta_Lambda[, q + seq_len(k),  drop = FALSE]

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
    Gamma <- update_Gamma_cpp(Beta_Lambda, Xi_mat, sigma_sq_vec, kappa_sq_vec)
    Mu <- t(Gamma %*% Xi_mat)
    lp_bl_cur <- compute_beta_prior_term_cpp(Beta_Lambda, Mu, sigma_sq_vec)
    lp_G_cur  <- compute_Gamma_prior_term(Gamma, kappa_sq_vec)
    t_g <- proc.time()[3] - t_g0

    t_s0 <- proc.time()[3]
    sigma_sq_vec <- update_sigma_sq_em(Beta_Lambda, Gamma, Xi_mat, X_tilde, obs_sub_num,
                                       sigma_sq_vec, a, b, kappa_sq_vec)
    lp_bl_cur  <- compute_beta_prior_term_cpp(Beta_Lambda, Mu, sigma_sq_vec)
    lp_sig_cur <- compute_sigma_prior_term(sigma_sq_vec, a, b)
    kappa_sq_vec <- update_kappa_sq_cpp(Gamma, c_hyp, d_hyp, 0.01, 100,
      gamma_post_trace_cpp(tcrossprod(Xi_mat), sigma_sq_vec, kappa_sq_vec, nrow(Xi_mat), 0))
    lp_G_cur   <- compute_Gamma_prior_term(Gamma, kappa_sq_vec)
    lp_kap_cur <- compute_kappa_prior_term(kappa_sq_vec, c_hyp, d_hyp)
    t_s <- proc.time()[3] - t_s0

    t_lp0 <- proc.time()[3]
    lp_new <- ll_cur + lp_bl_cur + lp_M_cur + lp_G_cur + lp_sig_cur + lp_kap_cur
    t_lp <- proc.time()[3] - t_lp0

    rel_ch <- if (is.finite(lp_new) && is.finite(lp_old)) {
      (lp_new - lp_old) / max(abs(lp_old), 1)
    } else {
      NA_real_
    }
    cat(sprintf("  Iter %2d | log-post = %12.2f | rel delta = %+.3e\n",
                iter, lp_new, rel_ch))
    if (timing_diagnostics) {
      cat(sprintf("    step time | Mu=%.2fs  M=%.2fs  ll_cur=%.2fs  BetaLam=%.2fs  merge=%.2fs  Gamma=%.2fs  sigma^2+kappa^2=%.2fs  logpost=%.2fs  total=%.2fs\n",
                  t_mu, t_m, t_data, t_bl, t_merge, t_g, t_s, t_lp, proc.time()[3] - iter_t0))
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
    lp_old <- lp_new
  }

  backbone <- list(
    Beta = Beta_Lambda[, seq_len(q), drop = FALSE],
    Lambda = Beta_Lambda[, q + seq_len(k), drop = FALSE],
    Beta_Lambda = Beta_Lambda,
    M = M,
    Gamma = Gamma,
    sigma_sq = sigma_sq_vec,   # (q+k): per-dimension
    kappa_sq = kappa_sq_vec,   # (q+k): per-row of Gamma
    a = a, b = b, c_hyp = c_hyp, d_hyp = d_hyp
  )
  if (is.null(Y_all) || is.null(Xi_all)) {
    # post_process_plus only needed for final output; skip when Stage 2 follows
    if (do_post_process) {
      pp <- post_process_plus(X, M, Beta_Lambda, q)
      Beta_Lambda <- pp$Beta_Lambda
      M <- pp$M
      rp <- realign_plus_prior(Beta_Lambda, M, Gamma, Xi_mat,
                               make_x_tilde(X, M), obs_sub_num,
                               sigma_sq_vec, kappa_sq_vec,
                               a, b, c_hyp, d_hyp)
      Gamma <- rp$Gamma
      sigma_sq_vec <- rp$sigma_sq
      kappa_sq_vec <- rp$kappa_sq
      backbone <- list(
        Beta = Beta_Lambda[, seq_len(q), drop = FALSE],
        Lambda = Beta_Lambda[, q + seq_len(k), drop = FALSE],
        Beta_Lambda = Beta_Lambda,
        M = M,
        Gamma = Gamma,
        sigma_sq = sigma_sq_vec,
        kappa_sq = kappa_sq_vec,
        a = a, b = b, c_hyp = c_hyp, d_hyp = d_hyp
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
  Beta_Lambda <- t(backbone$Gamma %*% Xi_all)
  Beta_Lambda[bb_idx, ] <- backbone$Beta_Lambda
  M            <- backbone$M
  Gamma        <- backbone$Gamma
  sigma_sq_vec <- backbone$sigma_sq   # (q+k): same for all species, no per-species expansion
  kappa_sq_vec <- backbone$kappa_sq   # (q+k): same

  ord_base <- order(colSums(Y_all), decreasing = TRUE)
  idx_batches <- split(ord_base, ceiling(seq_along(ord_base) / BL_species_batch_size))
  effective_workers <- if (has_openmp) min(worker_cores, length(idx_batches)) else 1L
  X_tilde <- make_x_tilde(X, M)
  Mu <- t(Gamma %*% Xi_all)
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
  lp_kap_cur <- compute_kappa_prior_term(kappa_sq_vec, c_hyp, d_hyp)
  lp_old <- ll_cur + lp_bl_cur + lp_M_cur + lp_G_cur + lp_sig_cur + lp_kap_cur
  cat(sprintf("FLAIR+ minibatch all-species | n=%d  p=%d  q=%d  k=%d  h=%d  BL_species_batch=%d\n",
              nrow(Y_num), ncol(Y_num), q, k, nrow(Xi_all), BL_species_batch_size))
  cat(sprintf("  M prior: eta_i ~ N(0, %.4g I_k); E||eta_i||^2 = %.4g\n",
              m_prior_var, k * m_prior_var))
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

  for (iter in seq_len(iter_max_train)) {
    iter_t0 <- proc.time()[3]

    t_mu <- 0
    ord_iter <- build_batch_order_cpp(ord_base, randomize_batches && iter > 1L)

    m_idx <- if (identical(stage2_m_update, "backbone")) bb_idx else seq_len(nrow(Beta_Lambda))
    Beta_m   <- Beta_Lambda[m_idx, seq_len(q),     drop = FALSE]
    Lambda_m <- Beta_Lambda[m_idx, q + seq_len(k), drop = FALSE]

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
    Gamma <- update_Gamma_cpp(Beta_Lambda, Xi_all, sigma_sq_vec, kappa_sq_vec)
    Mu <- t(Gamma %*% Xi_all)
    lp_bl_cur <- compute_beta_prior_term_cpp(Beta_Lambda, Mu, sigma_sq_vec)
    lp_G_cur  <- compute_Gamma_prior_term(Gamma, kappa_sq_vec)
    t_g <- proc.time()[3] - t_g0

    t_s0 <- proc.time()[3]
    sigma_sq_vec <- update_sigma_sq_em(Beta_Lambda, Gamma, Xi_all, X_tilde, obs_num,
                                       sigma_sq_vec, a, b, kappa_sq_vec)
    lp_bl_cur  <- compute_beta_prior_term_cpp(Beta_Lambda, Mu, sigma_sq_vec)
    lp_sig_cur <- compute_sigma_prior_term(sigma_sq_vec, a, b)
    kappa_sq_vec <- update_kappa_sq_cpp(Gamma, c_hyp, d_hyp, 0.01, 100,
      gamma_post_trace_cpp(tcrossprod(Xi_all), sigma_sq_vec, kappa_sq_vec, nrow(Xi_all), 0))
    lp_G_cur   <- compute_Gamma_prior_term(Gamma, kappa_sq_vec)
    lp_kap_cur <- compute_kappa_prior_term(kappa_sq_vec, c_hyp, d_hyp)
    t_s <- proc.time()[3] - t_s0

    t_lp0 <- proc.time()[3]
    lp_new <- ll_cur + lp_bl_cur + lp_M_cur + lp_G_cur + lp_sig_cur + lp_kap_cur
    t_lp <- proc.time()[3] - t_lp0

    rel_ch <- if (is.finite(lp_new) && is.finite(lp_old)) {
      (lp_new - lp_old) / max(abs(lp_old), 1)
    } else {
      NA_real_
    }
    cat(sprintf("  Iter %2d | log-post = %12.2f | rel delta = %+.3e\n",
                iter, lp_new, rel_ch))
    if (timing_diagnostics) {
      cat(sprintf("    step time | Mu=%.2fs  M=%.2fs  ll_cur=%.2fs  BetaLam=%.2fs  merge=%.2fs  Gamma=%.2fs  sigma^2+kappa^2=%.2fs  logpost=%.2fs  total=%.2fs\n",
                  t_mu, t_m, t_data, t_bl, t_merge, t_g, t_s, t_lp, proc.time()[3] - iter_t0))
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
    if (is.finite(rel_ch) && rel_ch >= 0 && rel_ch < loss_tol) {
      cat("  Converged.\n")
      break
    }
    lp_old <- lp_new
  }

  if (do_post_process) {
    pp <- post_process_plus(X, M, Beta_Lambda, q)
    Beta_Lambda <- pp$Beta_Lambda
    M <- pp$M
    # Gamma, sigma_sq and kappa_sq are the ridge fit on the post-processed
    # coefficients, the parameterization used by predictions and by the
    # variance corrector.
    rp <- realign_plus_prior(Beta_Lambda, M, Gamma, Xi_all,
                             make_x_tilde(X, M), obs_num,
                             sigma_sq_vec, kappa_sq_vec,
                             a, b, c_hyp, d_hyp)
    Gamma <- rp$Gamma
    sigma_sq_vec <- rp$sigma_sq
    kappa_sq_vec <- rp$kappa_sq
  }

  list(
    Beta = Beta_Lambda[, seq_len(q), drop = FALSE],
    Lambda = Beta_Lambda[, q + seq_len(k), drop = FALSE],
    Beta_Lambda = Beta_Lambda,
    M = M,
    Gamma = Gamma,
    sigma_sq = sigma_sq_vec,   # (q+k): per-dimension
    kappa_sq = kappa_sq_vec,   # (q+k): per-row of Gamma
    a = a, b = b, c_hyp = c_hyp, d_hyp = d_hyp,
    backbone = backbone
  )
}

# ---------------------------------------------------------------------------
# Uncertainty quantification for all FLAIR+ parameters after fitting
# ---------------------------------------------------------------------------
#
# Parameters and posterior approximations used:
#
#  Beta / Lambda  (b̃_j = (beta_j, lambda_j), p x d):
#    Laplace posterior: V_j = H_{j|nu}^{-1},
#      H_{j|nu} = X_tilde^T diag(w_j) X_tilde + diag(1/sigma_sq)
#    Marginal variance inflated by rho^2_j (Schur complement surrogate).
#    Uses C++ with OpenMP: O(p * (nd^2 + d^3)) / num_threads.
#
#  M  (n x k latent factors):
#    Laplace posterior per site i: V_{M_i} = H_{M_i}^{-1},
#      H_{M_i} = Lambda^T diag(w_i) Lambda + I_k / v_M,  w_i = obs_i * mu_i * (1-mu_i).
#    Cost O(n * (pk + k^3)).
#
#  Gamma  ((q+k) x H trait embedding rows):
#    Exact conjugate ridge posterior per row r:
#      Cov(Gamma_r) = (Xi_mat %*% t(Xi_mat) / sigma_sq[r] + I_H / kappa_sq[r])^{-1}
#    Requires Xi_mat.  Cost O(d * H^3).
#
#  sigma_sq  (d-vector, InvGamma posterior):
#    Posterior shape = a + p/2,  rate recovered from stored MAP.
#    Posterior mean = rate/(shape-1);  CI via qgamma on 1/sigma_sq.
#    No Xi_mat required.
#
#  kappa_sq  (d-vector, InvGamma posterior):
#    Posterior shape = c_hyp + H/2,  rate recovered from stored MAP.
#    CI via qgamma on 1/kappa_sq.  H inferred from ncol(fit$Gamma).
#
#  Xi  (H x p embedding, fixed during training): point estimate only.
#
# Args:
#   fit        : output list from FLAIR_plus_minibatch_wrapper().
#   Y          : n x p binary response matrix (training data).
#   X          : n x q covariate matrix.
#   Xi_mat     : H x p embedding matrix used during fitting (for Gamma UQ).
#                NULL skips Gamma CIs.
#   observed   : n x p 0/1 observation mask; NULL = fully observed.
#   m_prior_var: scalar prior variance for M rows (default 1/k, matching fitting default).
#   full_cov   : if TRUE, also return full d x d x p posterior covariance cube for b̃.
#   alpha      : miscoverage level; default 0.05 → 95% CIs.
#
# Returns a named list with sub-lists per parameter:
#   $Beta    : hat (p x q), sd (p x q), ci_lower (p x q), ci_upper (p x q)
#   $Lambda  : hat (p x k), sd (p x k), ci_lower (p x k), ci_upper (p x k)
#   $M       : hat (n x k), sd (n x k)  [Laplace approx, no CI tag]
#   $Gamma   : hat (d x H), sd (d x H), ci_lower (d x H), ci_upper (d x H)
#              [NULL if Xi_mat not supplied]
#   $sigma_sq: map (d), mean (d), sd (d), ci_lower (d), ci_upper (d)
#   $kappa_sq: map (d), mean (d), sd (d), ci_lower (d), ci_upper (d)
#   $Xi_hat  : H x p point estimate (Xi is fixed; no posterior)
#   Internal (for downstream use):
#   $V_cond_diag, $rho_sq, $V_marg_diag, $V_cond [if full_cov]
compute_UQ <- function(fit, Y, X, Xi_mat = NULL, observed = NULL,
                       m_prior_var = NULL, full_cov = FALSE, alpha = 0.05) {
  Y_num <- if (is.integer(Y)) Y else as_numeric_binary(Y)
  p  <- ncol(Y_num)
  n  <- nrow(Y_num)
  q  <- ncol(X)
  k  <- ncol(fit$M)
  d  <- q + k
  H  <- ncol(fit$Gamma)   # embedding dimension (nrow Xi_mat)

  if (is.null(observed))
    observed <- matrix(1.0, n, p)
  storage.mode(observed) <- "double"

  BL       <- fit$Beta_Lambda   # p x d
  sigma_sq <- fit$sigma_sq      # d-vector: per-dimension prior variances
  X_tilde  <- cbind(X, fit$M)  # n x d

  # Shared n x p sigmoid probabilities — computed once, reused by both the
  # Beta/Lambda C++ call and the M posterior loop below.
  mu_mat <- plogis(X_tilde %*% t(BL))   # n x p

  # -------------------------------------------------------------------------
  # 1.  Beta / Lambda: per-species conditional posterior (C++)
  #     cost O(p * (nd^2 + d^3)) — ~10-20x faster than equivalent R loop
  # -------------------------------------------------------------------------
  V_cond_diag <- compute_V_diag_all_cpp(
    mu_mat, observed, X_tilde, 1.0 / sigma_sq
  )  # p x d

  if (full_cov) {
    # Full d x d x p cube (rarely needed)
    V_cond_arr  <- array(NA_real_, c(d, d, p))
    Sigma_b_inv <- diag(1.0 / sigma_sq)
    for (j in seq_len(p)) {
      w_j <- observed[, j] * mu_mat[, j] * (1.0 - mu_mat[, j])
      V_cond_arr[,, j] <- solve(crossprod(X_tilde * sqrt(w_j)) + Sigma_b_inv)
    }
  }

  # Plug-in tier: the conditional covariance V_cond without inflation; the
  # Schur-complement correction is applied by compute_UQ_joint().
  rho_sq   <- rep(1.0, p)

  V_marg_diag <- sweep(V_cond_diag, 1L, rho_sq, `*`)
  z_alpha     <- qnorm(1.0 - alpha / 2.0)
  sd_marg     <- sqrt(V_marg_diag)

  Beta_sd    <- sd_marg[, seq_len(q),     drop = FALSE]
  Lambda_sd  <- sd_marg[, q + seq_len(k), drop = FALSE]

  # -------------------------------------------------------------------------
  # 2.  M: per-site Laplace posterior  (k x k Hessian per site)
  #         cost O(n * (pk + k^3)) — reuses mu_mat from above
  # -------------------------------------------------------------------------
  v_M <- if (is.null(m_prior_var)) 1.0 / k else as.numeric(m_prior_var)[1L]
  v_M <- max(1e-12, v_M)
  M_sd   <- matrix(NA_real_, n, k)
  prec_M_prior <- diag(k) / v_M              # k x k prior precision (shared)
  for (i in seq_len(n)) {
    w_i  <- observed[i, ] * mu_mat[i, ] * (1.0 - mu_mat[i, ])  # p-vector
    H_Mi <- crossprod(fit$Lambda * sqrt(w_i)) + prec_M_prior    # k x k
    M_sd[i, ] <- sqrt(diag(solve(H_Mi)))
  }

  # =========================================================================
  # 3–5 are coupled: Gamma CIs require V_{Γ_r}, REML correction requires
  # both V_{Γ_r} and V_j[r,r], and corrected σ²/κ² posteriors feed the
  # marginal Gamma variance.  Compute in order: (3a) V_{Γ_r} conditional,
  # (4/5) REML-corrected posteriors, (3b) Gamma marginal via MC.
  # =========================================================================

  # Base σ² posterior (plug-in, before REML correction)
  #   σ²_r ~ InvGamma(a_post, b_post)
  #   MAP = b_post / (a_post + 1)  →  b_post = MAP * (a_post + 1)
  a_post_sig <- fit$a + p / 2.0                       # scalar, broadcasts to d
  b_post_sig <- fit$sigma_sq * (a_post_sig + 1.0)    # d-vector

  # Base κ² posterior (plug-in)
  c_post_kap <- fit$c_hyp + H / 2.0
  d_post_kap <- fit$kappa_sq * (c_post_kap + 1.0)    # d-vector

  # -------------------------------------------------------------------------
  # 3a. Gamma conditional posterior (H x H ridge, one per row r)
  #     Also accumulate REML trace terms while V_{Γ_r} matrices are hot.
  # -------------------------------------------------------------------------
  Gamma_sd_marg <- NULL; Gamma_lower <- NULL; Gamma_upper <- NULL
  V_Gr_list     <- NULL

  if (!is.null(Xi_mat)) {
    Xi_mat <- as.matrix(Xi_mat)
    XiXit  <- tcrossprod(Xi_mat)              # H x H, shared across rows
    Xi_BL  <- Xi_mat %*% BL                  # H x d: Xi_mat %*% BL[,r] = Σ_j ξ_j b̃_{jr}

    V_Gr_list  <- vector("list", d)
    V_BL_trace <- colSums(V_cond_diag)        # d: Σ_j V_j[r,r] from step 1
    V_Gam_XiXi <- numeric(d)                  # d: tr(V_{Γ_r} Xi Xi^T)  for σ² REML
    V_Gam_diag <- numeric(d)                  # d: tr(V_{Γ_r})           for κ² REML

    for (r in seq_len(d)) {
      prec_r        <- XiXit / sigma_sq[r] + diag(H) / fit$kappa_sq[r]
      V_Gr          <- solve(prec_r)
      V_Gr_list[[r]] <- V_Gr
      V_Gam_XiXi[r] <- sum(diag(V_Gr %*% XiXit))   # tr(V_{Γ_r} ΞΞ^T)
      V_Gam_diag[r]  <- sum(diag(V_Gr))             # tr(V_{Γ_r})
    }

    # -----------------------------------------------------------------------
    # REML-corrected σ² posterior rate:
    #   b_corr_r = b_post_r + (1/2) [Σ_j V_j[r,r]  +  tr(V_{Γ_r} ΞΞ^T)]
    #
    # First term: posterior variance of b̃_{jr} propagated into SS_r.
    # Second term: posterior variance of μ̂_{jr} = Γ_{r·} ξ_j; summing over j
    #   gives Σ_j ξ_j^T V_{Γ_r} ξ_j = tr(V_{Γ_r} ΞΞ^T).
    # -----------------------------------------------------------------------
    b_post_sig <- b_post_sig + (V_BL_trace + V_Gam_XiXi) / 2.0

    # REML-corrected κ² posterior rate:
    #   d_corr_r = d_post_r + (1/2) tr(V_{Γ_r})
    # Accounts for posterior variance of Γ_{r·} when estimating its own scale.
    d_post_kap <- d_post_kap + V_Gam_diag / 2.0

    # -----------------------------------------------------------------------
    # 3b. Gamma MARGINAL posterior via Monte Carlo law of total variance.
    #     Samples σ²^(s) from corrected InvGamma; for each sample computes
    #     conditional V_{Γ_r}(σ²) and E[Γ_r|σ²].  Then:
    #       Var_marg(Γ_{r,h}) = E[V_{Γ_r,hh}] + Var(E[Γ_{r,h}|σ²])
    # Cost: n_mc × d × H^3 (e.g. 300 × 15 × 10^3 ≈ 4.5 M ops)
    # -----------------------------------------------------------------------
    n_mc          <- 300L
    Gam_mean_mc   <- array(NA_real_, c(d, H, n_mc))   # conditional mean per draw
    Gam_var_mc    <- array(NA_real_, c(d, H, n_mc))   # conditional var  per draw
    I_H           <- diag(H)

    for (s in seq_len(n_mc)) {
      ss_s <- 1.0 / rgamma(d, shape = a_post_sig, rate = b_post_sig)  # d-vector
      for (r in seq_len(d)) {
        prec_rs           <- XiXit / ss_s[r] + I_H / fit$kappa_sq[r]
        V_rs              <- solve(prec_rs)
        Gam_mean_mc[r,,s] <- V_rs %*% Xi_BL[, r] / ss_s[r]
        # Schur propagation of BL_MAP uncertainty into Gamma_r.  Gamma_r =
        # V_{Gamma_r} Xi BL_{.r}/sigma^2 treats BL as data, but each BL_{jr} has
        # posterior variance V_j[r,r].  With Cov(BL_{.r}) = diag(V_cond_diag[,r]),
        # this adds diag(V_rs Xi diag(v_r) Xi^T V_rs)/sigma^4 to the Gamma variance.
        M_rs              <- V_rs %*% Xi_mat                          # H x p
        bl_prop           <- rowSums(sweep(M_rs^2, 2L, V_cond_diag[, r], `*`)) / ss_s[r]^2
        Gam_var_mc[r,,s]  <- diag(V_rs) + bl_prop
      }
    }

    # Law of total variance: mean of conditional variances + variance of means
    Gamma_var_marg <- apply(Gam_var_mc,  c(1L, 2L), mean) +
                      apply(Gam_mean_mc, c(1L, 2L), var)
    Gamma_sd_marg  <- sqrt(Gamma_var_marg)    # d x H

    Gamma_lower <- fit$Gamma - z_alpha * Gamma_sd_marg
    Gamma_upper <- fit$Gamma + z_alpha * Gamma_sd_marg
  }

  # -------------------------------------------------------------------------
  # 4.  σ² marginal CIs (uses REML-corrected b_post if Xi_mat was supplied)
  #     1/σ²_r ~ Gamma(a_post, rate = b_post)
  # -------------------------------------------------------------------------
  sigma_sq_mean  <- b_post_sig / (a_post_sig - 1.0)
  sigma_sq_var   <- b_post_sig^2 / ((a_post_sig - 1.0)^2 * (a_post_sig - 2.0))
  sigma_sq_sd    <- sqrt(pmax(sigma_sq_var, 0))
  sigma_sq_lower <- 1.0 / qgamma(1.0 - alpha / 2.0, shape = a_post_sig, rate = b_post_sig)
  sigma_sq_upper <- 1.0 / qgamma(      alpha / 2.0, shape = a_post_sig, rate = b_post_sig)

  # -------------------------------------------------------------------------
  # 5.  κ² marginal CIs (uses REML-corrected d_post if Xi_mat was supplied)
  # -------------------------------------------------------------------------
  kappa_sq_mean  <- d_post_kap / (c_post_kap - 1.0)
  kappa_sq_var   <- d_post_kap^2 / ((c_post_kap - 1.0)^2 * (c_post_kap - 2.0))
  kappa_sq_sd    <- sqrt(pmax(kappa_sq_var, 0))
  kappa_sq_lower <- 1.0 / qgamma(1.0 - alpha / 2.0, shape = c_post_kap, rate = d_post_kap)
  kappa_sq_upper <- 1.0 / qgamma(      alpha / 2.0, shape = c_post_kap, rate = d_post_kap)

  # -------------------------------------------------------------------------
  # Build output — all SDs are *marginal* (nuisance-integrated where possible)
  # -------------------------------------------------------------------------
  out <- list(
    Beta = list(
      hat      = fit$Beta,
      sd       = Beta_sd,                          # p x q  (inflated, cond. on MAP M)
      ci_lower = fit$Beta - z_alpha * Beta_sd,
      ci_upper = fit$Beta + z_alpha * Beta_sd
    ),
    Lambda = list(
      hat      = fit$Lambda,
      sd       = Lambda_sd,                        # p x k  (inflated, cond. on MAP M)
      ci_lower = fit$Lambda - z_alpha * Lambda_sd,
      ci_upper = fit$Lambda + z_alpha * Lambda_sd
    ),
    M = list(
      hat  = fit$M,    # n x k  MAP
      sd   = M_sd      # n x k  Laplace approx SD (cond. on MAP BL)
    ),
    Gamma = if (!is.null(Xi_mat)) list(
      hat      = fit$Gamma,      # d x H  MAP
      sd       = Gamma_sd_marg,  # d x H  marginal (law of total variance over σ²)
      ci_lower = Gamma_lower,    # d x H
      ci_upper = Gamma_upper     # d x H
    ) else list(hat = fit$Gamma),
    sigma_sq = list(
      map      = fit$sigma_sq,    # d  MAP (mode of plug-in posterior)
      mean     = sigma_sq_mean,   # d  posterior mean (REML-corrected if Xi_mat given)
      sd       = sigma_sq_sd,     # d
      ci_lower = sigma_sq_lower,  # d
      ci_upper = sigma_sq_upper   # d
    ),
    kappa_sq = list(
      map      = fit$kappa_sq,    # d  MAP
      mean     = kappa_sq_mean,   # d  posterior mean (REML-corrected if Xi_mat given)
      sd       = kappa_sq_sd,     # d
      ci_lower = kappa_sq_lower,  # d
      ci_upper = kappa_sq_upper   # d
    ),
    Xi_hat      = Xi_mat,          # H x p; fixed (not estimated)
    V_cond_diag = V_cond_diag,    # p x d  (internal)
    rho_sq      = rho_sq,         # p      (internal)
    V_marg_diag = V_marg_diag     # p x d  (internal)
  )
  if (full_cov)
    out$V_cond <- V_cond_arr      # d x d x p

  out
}

# ---------------------------------------------------------------------------
# compute_UQ_marginal
#
# Extends compute_UQ() with one additional source of joint uncertainty:
#
# (A) Beta / Lambda: adds Γ-propagation variance per species:
#       Var_marg(b̃_{jr}) = rho²_j · V_cond[j,r] + xi_j^T V_{Γ_r} xi_j
#     where V_{Γ_r} = (ΞΞ^T/σ²_r + I_H/κ²_r)^{-1} is the conjugate
#     posterior covariance of the r-th row of Γ.
#
# sigma_sq: the REML InvGamma CI from compute_UQ() is kept as-is.
#   A Laplace on log(σ²) was previously used here, but it is inferior
#   because MAP residuals BL_MAP - Gamma_MAP*xi_j are already shrunk toward
#   zero by the joint MAP optimisation.  The REML correction in compute_UQ()
#   (b_post += 0.5*(V_BL_trace + V_Gam_XiXi)) properly accounts for both
#   BL and Gamma posterior uncertainty and gives a robustly higher CI centre.
#
# Returns the same structure as compute_UQ() (drop-in compatible).
# Extra fields added to the return list:
#   $V_gamma_diag     : p x d  -- xi_j^T V_{Γ_r} xi_j per (j,r)
#   $V_marg_total_diag: p x d  -- V_marg_diag + V_gamma_diag
# ---------------------------------------------------------------------------
compute_UQ_marginal <- function(fit, Y, X, Xi_mat,
                                observed    = NULL,
                                m_prior_var = NULL,
                                full_cov    = FALSE,
                                alpha       = 0.05) {
  stopifnot(!is.null(Xi_mat))
  base <- compute_UQ(fit, Y, X, Xi_mat = Xi_mat, observed = observed,
                     m_prior_var = m_prior_var, full_cov = full_cov,
                     alpha = alpha)

  p     <- nrow(fit$Beta_Lambda)
  q     <- ncol(X)
  k     <- ncol(fit$M)
  d     <- q + k
  H     <- ncol(fit$Gamma)
  z_alp <- qnorm(1.0 - alpha / 2.0)

  Xi_mat <- as.matrix(Xi_mat)   # H x p
  XiXit  <- tcrossprod(Xi_mat)  # H x H

  # ---- (A) Gamma-propagation term for Beta / Lambda -----------------------
  # V_gamma_diag[j, r] = xi_j^T V_{Γ_r} xi_j
  V_gamma_diag <- matrix(NA_real_, p, d)
  for (r in seq_len(d)) {
    V_Gr <- solve(XiXit / fit$sigma_sq[r] + diag(H) / fit$kappa_sq[r])
    V_gamma_diag[, r] <- colSums(Xi_mat * (V_Gr %*% Xi_mat))
  }

  V_total_diag <- base$V_marg_diag + V_gamma_diag  # p x d
  sd_total     <- sqrt(pmax(V_total_diag, 0))

  Beta_sd_m   <- sd_total[, seq_len(q),     drop = FALSE]
  Lambda_sd_m <- sd_total[, q + seq_len(k), drop = FALSE]

  base$Beta$sd       <- Beta_sd_m
  base$Beta$ci_lower <- fit$Beta   - z_alp * Beta_sd_m
  base$Beta$ci_upper <- fit$Beta   + z_alp * Beta_sd_m

  base$Lambda$sd       <- Lambda_sd_m
  base$Lambda$ci_lower <- fit$Lambda - z_alp * Lambda_sd_m
  base$Lambda$ci_upper <- fit$Lambda + z_alp * Lambda_sd_m

  base$V_gamma_diag      <- V_gamma_diag
  base$V_marg_total_diag <- V_total_diag

  # ---- (B) sigma_sq: keep REML InvGamma CI from base (compute_UQ) ----------
  # The Laplace approach (integrating out BL at MAP residuals) was removed
  # because: (i) MAP residuals BL_MAP - Gamma_MAP*xi_j are already shrunk
  # toward zero by the joint MAP optimisation, so the Laplace finds a MAP
  # close to the training sigma_sq_MAP rather than the REML-corrected value;
  # (ii) the effective BL variance v_r = V_cond_diag should also include the
  # Gamma propagation term xi_j^T V_Gamma_r xi_j, which the Laplace did not.
  #
  # compute_UQ() already builds the principled REML InvGamma CI:
  #   b_post_reml = sigma_sq_MAP*(a_post+1) + 0.5*(V_BL_trace + V_Gam_XiXi)
  # where V_BL_trace = sum_j V_j[r,r] and V_Gam_XiXi = tr(V_Gamma_r * XiXi^T)
  # correct for BL and Gamma posterior uncertainty respectively.
  # base$sigma_sq already carries this REML InvGamma CI — no override needed.

  base
}

# ---------------------------------------------------------------------------
# compute_UQ_joint
#
# Joint-uncertainty corrections on top of compute_UQ_marginal():
#
# (1) Schur complement of the M block for each species:
#       H_{j,M} = X̃^T diag(w_eff_j) X̃ + diag(1/σ²)
#       w_eff_{ij} = w_{ij} - w_{ij}^2 · λ_j^T B_i^{-1} λ_j
#       B_i = Σ_{j'} w_{ij'} λ_{j'} λ_{j'}^T + I_k / v_M
#
# (2) Gamma rows: Schur complement of the b̃ block, so species j enters with
#     weight u_jr = h_jr / (1 + σ²_r h_jr), h_jr = 1/[H_{j,M}^{-1}]_rr - 1/σ²_r
#     its data precision (u ≈ 0 for rare species), marginalised over σ²_r by
#     Monte Carlo:  Var(Γ_{r,h}) = E[V_{Γ_r,hh}(σ²)] + Var(E[Γ_{r,h}|σ²]).
#
# (3) Beta / Lambda: σ² and Γ uncertainty inflate the prior variance of b̃_{jr}
#     from the MAP σ²_r to E[σ²_r] + ξ_j^T V_{Γ_r} ξ_j (REML posterior mean):
#       H_{j,marg} = X̃^T diag(w_eff_j) X̃ + diag(1/(E[σ²] + D_j))
#
# Trainable embedding (Q_xi = prior precision of each column of Ξ, i.e.
# 2(τ_ξ I + γ_ξ L_norm) in the parameterisation of the fitter): (2) is replaced by
# the joint Gaussian of (vec Γ, Ξ) after integrating out each b̃_j, in which
# species j observes Γξ_j with precision W_j = (E[Σ_b] + A_j^{-1})^{-1}; then
# D_j = Var(Γξ_j) is a full d x d matrix carrying the uncertainty of both Γ
# and ξ_j, and Γ intervals carry the uncertainty of Ξ.
#
# Cost: O(n·p·k² + p·d³ + d·S_gamma·H³) — inexpensive for typical sizes;
# the trainable branch solves a dense (d+p)H system.
# ---------------------------------------------------------------------------
compute_UQ_joint <- function(fit, Y, X, Xi_mat,
                             observed    = NULL,
                             m_prior_var = NULL,
                             alpha       = 0.05,
                             S_gamma     = 200L,
                             Q_xi        = NULL) {
  stopifnot(!is.null(Xi_mat))
  # Laplace at the MAP before post-processing, then map through the
  # post-processing b̃_j -> T b̃_j, linear on the chart (X̃' δM = 0).
  if (!is.null(fit$map)) {
    f0 <- fit; f0$map <- NULL
    f0$Beta_Lambda <- fit$map$Beta_Lambda; f0$M <- fit$map$M; f0$Gamma <- fit$map$Gamma
    q <- ncol(X); k <- ncol(f0$M)
    f0$Beta <- f0$Beta_Lambda[, seq_len(q), drop = FALSE]
    f0$Lambda <- f0$Beta_Lambda[, q + seq_len(k), drop = FALSE]
    u <- compute_UQ_joint(f0, Y, X, Xi_mat, observed, m_prior_var, alpha, S_gamma, Q_xi)
    A <- solve(crossprod(X), crossprod(X, f0$M))
    s <- svd(f0$M - X %*% A, nu = 0, nv = k)
    Tm <- rbind(cbind(diag(q), A), cbind(matrix(0, k, q), t(s$v %*% diag(s$d, k)) / sqrt(nrow(X))))
    z_alp <- qnorm(1.0 - alpha / 2.0)
    BLc <- f0$Beta_Lambda %*% t(Tm)
    sdBL <- t(sapply(u$V_BL, function(V) sqrt(pmax(diag(Tm %*% V %*% t(Tm)), 0))))
    for (nm in c("Beta", "Lambda")) {
      cols <- if (nm == "Beta") seq_len(q) else q + seq_len(k)
      u[[nm]]$sd <- sdBL[, cols, drop = FALSE]
      u[[nm]]$ci_lower <- BLc[, cols, drop = FALSE] - z_alp * u[[nm]]$sd
      u[[nm]]$ci_upper <- BLc[, cols, drop = FALSE] + z_alp * u[[nm]]$sd
    }
    if (!is.null(u$V_Gamma)) {
      TG <- kronecker(diag(ncol(f0$Gamma)), Tm)
      G_hat <- Tm %*% f0$Gamma
      sdG <- matrix(sqrt(pmax(diag(TG %*% u$V_Gamma %*% t(TG)), 0)), nrow(G_hat))
      u$Gamma <- list(hat = G_hat, sd = sdG, ci_lower = G_hat - z_alp * sdG, ci_upper = G_hat + z_alp * sdG)
    }
    return(u)
  }

  Y_num <- if (is.integer(Y)) Y else as_numeric_binary(Y)
  p  <- nrow(fit$Beta_Lambda)
  n  <- nrow(Y_num)
  q  <- ncol(X)
  k  <- ncol(fit$M)
  d  <- q + k
  H  <- ncol(fit$Gamma)
  z_alp <- qnorm(1.0 - alpha / 2.0)

  if (is.null(observed)) observed <- matrix(1.0, n, p)
  storage.mode(observed) <- "double"

  v_M <- if (is.null(m_prior_var)) 1.0 / k else as.numeric(m_prior_var)[1L]
  v_M <- max(1e-12, v_M)

  # Shared quantities
  BL       <- fit$Beta_Lambda                   # p x d
  X_tilde  <- cbind(X, fit$M)                  # n x d
  mu_mat   <- plogis(X_tilde %*% t(BL))        # n x p
  W        <- observed * mu_mat * (1.0 - mu_mat)  # n x p

  Xi_mat <- as.matrix(Xi_mat)  # H x p

  base <- compute_UQ_marginal(fit, Y, X, Xi_mat = Xi_mat, observed = observed,
                               m_prior_var = m_prior_var, full_cov = FALSE,
                               alpha = alpha)

  # ---- (1) Schur complement marginalising over M -------------------------
  # Per-site M Hessian inverse:  B_i = Λ^T diag(w_i) Λ + I_k/v_M
  prec_M  <- diag(k) / v_M
  B_inv   <- vector("list", n)
  for (i in seq_len(n)) {
    H_Mi    <- crossprod(fit$Lambda * sqrt(W[i, ])) + prec_M
    B_inv[[i]] <- solve(H_Mi)
  }

  A_list   <- vector("list", p)       # data part X̃^T diag(w_eff_j) X̃
  V_M_diag <- matrix(NA_real_, p, d)
  for (j in seq_len(p)) {
    lam_j <- fit$Lambda[j, ]     # k-vec

    # Per-site Schur scalar: w_{ij}^2 * λ_j^T B_i^{-1} λ_j
    s_ij <- numeric(n)
    for (i in seq_len(n))
      s_ij[i] <- W[i, j]^2 * sum(lam_j * (B_inv[[i]] %*% lam_j))

    w_eff_j <- pmax(W[, j] - s_ij, 0)   # guard against numerical negativity

    A_list[[j]]   <- crossprod(X_tilde * sqrt(w_eff_j))
    V_M_diag[j, ] <- diag(solve(A_list[[j]] + diag(1.0 / as.numeric(fit$sigma_sq))))
  }

  # ---- (2) Gamma: Schur over b̃, marginalised over σ² via Monte Carlo -----
  # σ²_r ~ REML-corrected InvGamma(a + p/2, b_post_r) of compute_UQ().
  # h b^d = h b̂ + (b̂ - Γ̂ξ)/σ̂² undoes the prior shrinkage of b̂ toward Γ̂ξ.
  a_post  <- fit$a + p / 2.0
  sig_mat <- matrix(fit$sigma_sq, p, d, byrow = TRUE)
  h_mat   <- pmax(1.0 / V_M_diag - 1.0 / sig_mat, 0)
  hb_mat  <- h_mat * BL + (BL - t(fit$Gamma %*% Xi_mat)) / sig_mat

  Gamma_sd_jt    <- matrix(NA_real_, d, H)
  Gamma_lower_jt <- matrix(NA_real_, d, H)
  Gamma_upper_jt <- matrix(NA_real_, d, H)
  D_mat          <- matrix(NA_real_, p, d)   # ξ_j^T V_{Γ_r} ξ_j
  D_list         <- NULL                     # full Var(Γξ_j), trainable branch
  I_H            <- diag(H)

  if (!is.null(Q_xi)) {
    # Joint precision of (vec Γ [column-major, dH], Ξ [species-major, pH]).
    sig_bar <- as.numeric(base$sigma_sq$mean)
    S_inv   <- diag(1.0 / sig_bar)
    G       <- fit$Gamma
    dH      <- d * H
    J       <- matrix(0.0, dH + p * H, dH + p * H)
    J[seq_len(dH), seq_len(dH)] <- diag(rep(1.0 / fit$kappa_sq, times = H))
    J[dH + seq_len(p * H), dH + seq_len(p * H)] <- kronecker(Q_xi, I_H)
    for (j in seq_len(p)) {
      W_j  <- S_inv - S_inv %*% solve(A_list[[j]] + S_inv, S_inv)
      xi_j <- Xi_mat[, j]
      ij   <- dH + (j - 1L) * H + seq_len(H)
      J[seq_len(dH), seq_len(dH)] <- J[seq_len(dH), seq_len(dH)] + kronecker(tcrossprod(xi_j), W_j)
      J[ij, ij]           <- J[ij, ij] + crossprod(G, W_j %*% G)
      # Observed cross block: Gauss-Newton part minus the residual term
      # (Σ_b^{-1} e_j) ⊗ I_H, e_j = b̃_j - Γξ_j, which does not average out over species.
      J[seq_len(dH), ij]  <- kronecker(xi_j, W_j %*% G) -
        kronecker(I_H, S_inv %*% (BL[j, ] - G %*% xi_j))
      J[ij, seq_len(dH)]  <- t(J[seq_len(dH), ij])
    }
    ev_Q <- eigen(Q_xi, symmetric = TRUE, only.values = TRUE)$values
    if (isTRUE(fit$trait_intercept)) {
      # last embedding coordinate is the constant 1 (fixed); last Gamma column is mu
      ic <- dH + (seq_len(p) - 1L) * H + H
      J[cbind(ic, ic)] <- J[cbind(ic, ic)] + 1e8
      ig <- (H - 1L) * d + seq_len(d)
      J[cbind(ig, ig)] <- J[cbind(ig, ig)] - 1.0 / fit$kappa_sq + 1.0 / fit$mu_prior_var
      LH <- matrix(0, dH, dH)
      LH[seq_len(dH - d), seq_len(dH - d)] <- xi_logdet_hessian(G[, -H, drop = FALSE], sig_bar, ev_Q)
      J[seq_len(dH), seq_len(dH)] <- J[seq_len(dH), seq_len(dH)] + LH
    } else
      J[seq_len(dH), seq_len(dH)] <- J[seq_len(dH), seq_len(dH)] + xi_logdet_hessian(G, sig_bar, ev_Q)
    Sig <- solve(J)
    V_G <- Sig[seq_len(dH), seq_len(dH)]
    Gamma_sd_jt <- matrix(sqrt(pmax(diag(V_G), 0)), d, H)
    # The prior of b̃_j must not use b̃_j itself: remove species j's term
    # Jac' W_j Jac from the precision (Woodbury), D_j -> D_j + D_j W_j (I - D_j W_j)^{-1} D_j.
    # The residual part of the cross block stays: it is not part of the b̃_j-(Γ,ξ_j) coupling.
    D_list <- vector("list", p)
    for (j in seq_len(p)) {
      ij  <- dH + (j - 1L) * H + seq_len(H)
      Jac <- cbind(kronecker(t(Xi_mat[, j]), diag(d)), G)   # d(Γξ_j)/d(vec Γ, ξ_j)
      idx <- c(seq_len(dH), ij)
      Dj  <- Jac %*% Sig[idx, idx] %*% t(Jac)
      W_j <- S_inv - S_inv %*% solve(A_list[[j]] + S_inv, S_inv)
      D_list[[j]] <- Dj + Dj %*% W_j %*% solve(diag(d) - Dj %*% W_j, Dj)
      D_mat[j, ]  <- diag(D_list[[j]])
    }
    Gamma_lower_jt <- fit$Gamma - z_alp * Gamma_sd_jt
    Gamma_upper_jt <- fit$Gamma + z_alp * Gamma_sd_jt
  }

  for (r in seq_len(if (is.null(Q_xi)) d else 0L)) {
    ridge_r     <- rep(1.0 / fit$kappa_sq[r], H)
    if (isTRUE(fit$trait_intercept)) ridge_r[H] <- 1.0 / fit$mu_prior_var
    b_post_r    <- base$sigma_sq$mean[r] * (a_post - 1.0)
    sigma2_draws <- 1.0 / rgamma(S_gamma, shape = a_post, rate = b_post_r)

    V_avg    <- matrix(0.0, H, H)
    M_sum    <- numeric(H)
    M_sq_sum <- matrix(0.0, H, H)

    for (s in seq_len(S_gamma)) {
      den <-1.0 + sigma2_draws[s] * h_mat[, r]
      V_s <- solve(Xi_mat %*% (h_mat[, r] / den * t(Xi_mat)) + diag(ridge_r))
      Mn  <- as.numeric(V_s %*% (Xi_mat %*% (hb_mat[, r] / den)))
      V_avg    <- V_avg + V_s
      M_sum    <- M_sum + Mn
      M_sq_sum <- M_sq_sum + tcrossprod(Mn)
    }
    V_avg  <- V_avg  / S_gamma
    M_bar  <- M_sum  / S_gamma
    M_cov  <- M_sq_sum / S_gamma - tcrossprod(M_bar)
    V_tot  <- V_avg + M_cov            # law of total variance

    D_mat[, r]          <- colSums(Xi_mat * (V_tot %*% Xi_mat))
    sd_r                <- sqrt(pmax(diag(V_tot), 0))
    Gamma_sd_jt[r, ]    <- sd_r
    Gamma_lower_jt[r, ] <- fit$Gamma[r, ] - z_alp * sd_r
    Gamma_upper_jt[r, ] <- fit$Gamma[r, ] + z_alp * sd_r
  }

  base$Gamma <- list(
    hat      = fit$Gamma,
    sd       = Gamma_sd_jt,
    ci_lower = Gamma_lower_jt,
    ci_upper = Gamma_upper_jt
  )

  # ---- (3) Beta / Lambda: σ² and Γ uncertainty inflate the prior variance -
  V_total_BL <- matrix(NA_real_, p, d)
  V_BL <- vector("list", p)
  for (j in seq_len(p)) {
    prior_cov <- if (is.null(D_list)) diag(as.numeric(base$sigma_sq$mean) + D_mat[j, ])
                 else diag(as.numeric(base$sigma_sq$mean)) + D_list[[j]]
    V_BL[[j]] <- solve(A_list[[j]] + solve(prior_cov))
    V_total_BL[j, ] <- diag(V_BL[[j]])
  }
  base$V_BL    <- V_BL
  base$V_Gamma <- if (!is.null(Q_xi)) V_G
  sd_BL_jt   <- sqrt(pmax(V_total_BL, 0))

  Beta_sd_j   <- sd_BL_jt[, seq_len(q),     drop = FALSE]
  Lambda_sd_j <- sd_BL_jt[, q + seq_len(k), drop = FALSE]

  base$Beta$sd       <- Beta_sd_j
  base$Beta$ci_lower <- fit$Beta   - z_alp * Beta_sd_j
  base$Beta$ci_upper <- fit$Beta   + z_alp * Beta_sd_j

  base$Lambda$sd       <- Lambda_sd_j
  base$Lambda$ci_lower <- fit$Lambda - z_alp * Lambda_sd_j
  base$Lambda$ci_upper <- fit$Lambda + z_alp * Lambda_sd_j

  base$V_marg_BL_diag    <- V_M_diag
  base$V_gamma_diag      <- D_mat
  base$V_marg_total_diag <- V_total_BL

  base
}

# ---------------------------------------------------------------------------
# compute_sigma_sq_reml
#
# Standalone REML-style MAP de-biasing for sigma_sq.
#
# The conditional MAP equation (used inside FLAIR_plus_minibatch_wrapper)
# treats BL_{jr} and Γ_{r·} as observed at their MAP values, ignoring their
# posterior variance.  This shrinks σ²_r toward zero.  The REML rate adds
# the two missing variance components back:
#
#   σ²_r_REML = [ b + ½ Σ_j( (BL_{jr} - μ̂_{jr})² + V_j[r,r]
#                              + xi_j^T V_{Γ_r} xi_j ) ] / (a + p/2 + 1)
#
# where μ̂_{jr} = Γ̂_{r·} ξ_j and V_{Γ_r} = (ΞΞ^T/σ²_r + I/κ²_r)^{-1}.
#
# Returns a length-d vector.  Intended as a post-fit diagnostic; to widen
# downstream CIs, swap fit$sigma_sq <- compute_sigma_sq_reml(fit, ...) before
# calling compute_UQ*.
# ---------------------------------------------------------------------------
compute_sigma_sq_reml <- function(fit, Y, X, Xi_mat, observed = NULL,
                                   m_prior_var = NULL) {
  Y_num <- if (is.integer(Y)) Y else as_numeric_binary(Y)
  p  <- nrow(fit$Beta_Lambda)
  n  <- nrow(Y_num)
  q  <- ncol(X)
  k  <- ncol(fit$M)
  d  <- q + k
  H  <- ncol(fit$Gamma)

  if (is.null(observed)) observed <- matrix(1.0, n, p)
  storage.mode(observed) <- "double"

  BL      <- fit$Beta_Lambda
  X_tilde <- cbind(X, fit$M)
  mu_mat  <- plogis(X_tilde %*% t(BL))   # n x p

  Xi_mat  <- as.matrix(Xi_mat)            # H x p
  XiXit   <- tcrossprod(Xi_mat)           # H x H
  pred_BL <- t(fit$Gamma %*% Xi_mat)      # p x d

  V_cond_diag <- compute_V_diag_all_cpp(
    mu_mat, observed, X_tilde, 1.0 / fit$sigma_sq
  )   # p x d

  # Per-species Γ-propagation variance: xi_j^T V_{Γ_r} xi_j
  V_gamma_diag <- matrix(NA_real_, p, d)
  I_H <- diag(H)
  for (r in seq_len(d)) {
    V_Gr <- solve(XiXit / fit$sigma_sq[r] + I_H / fit$kappa_sq[r])
    V_gamma_diag[, r] <- colSums(Xi_mat * (V_Gr %*% Xi_mat))
  }

  res_sq    <- (BL - pred_BL)^2                                 # p x d
  numerator <- fit$b + 0.5 * colSums(res_sq + V_cond_diag + V_gamma_diag)
  denom     <- fit$a + p / 2.0 + 1.0
  numerator / denom
}
