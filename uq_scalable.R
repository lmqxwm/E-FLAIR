# Scalable componentwise variance corrector (same quantities as
# compute_UQ_joint in flair_plus_minibatch_tr.R, without n x p matrices).
#
#   rho_sq[j, r] = [V_marg]_rr / [V_cond]_rr   componentwise inflation
#   Beta / Lambda intervals from diag(V_marg), Gamma rows by Monte Carlo over σ².
#
# Cost: O(npk² + nk³) for M, O(npd² + pd³) per species pass (two passes),
#       O{S d (pH² + H³)} for Gamma; trainable = TRUE (Xi integrated out,
#       block-Jacobi in species) costs O{p (d³ + dH² + H³) + p d² H²}.

library(Rcpp)
sourceCpp(file.path(if (exists("FLAIR_PROJ_DIR")) FLAIR_PROJ_DIR else ".",
                    "helper_functions_uq.cpp"))

# Hessian in vec(Gamma) (column-major) of (1/2) sum_i w_i log det(q_i I_H + Gamma' diag(1/sig) Gamma),
# i.e. of (1/2) log det(Q_xi (x) I_H + I_p (x) Gamma' Sigma^{-1} Gamma) with q_i the eigenvalues of Q_xi
# (w_i = quadrature weights).  Integrating Xi out exactly adds it to the Gamma precision; the joint
# Laplace in (Gamma, Xi) omits it, although it is of the same order p.
xi_logdet_hessian <- function(Gamma, sig, q, w = rep(1, length(q))) {
  d <- nrow(Gamma); H <- ncol(Gamma); S <- 1 / as.numeric(sig)
  e  <- eigen(crossprod(Gamma, Gamma * S), symmetric = TRUE); U <- e$vectors
  R  <- 1 / outer(q, pmax(e$values, 0), `+`)
  cc <- crossprod(R * w, R)
  g  <- (Gamma %*% U) * S
  L  <- matrix(0, H * H, d * H)          # vec(delta Gamma U) -> vec of the change of U' A U
  for (h in seq_len(H)) for (h2 in seq_len(H)) {
    row <- h + (h2 - 1) * H
    L[row, (h - 1) * d + seq_len(d)]  <- L[row, (h - 1) * d + seq_len(d)]  + g[, h2]
    L[row, (h2 - 1) * d + seq_len(d)] <- L[row, (h2 - 1) * d + seq_len(d)] + g[, h]
  }
  Ht <- kronecker(diag(colSums(w * R), H), diag(S, d)) - 0.5 * crossprod(L, as.vector(cc) * L)
  T  <- kronecker(U, diag(d))
  T %*% Ht %*% t(T)
}

compute_UQ_scalable <- function(fit, X, Xi_mat, observed = NULL,
                                m_prior_var = NULL, alpha = 0.05,
                                S_gamma = 20L, block = 5000L, cores = 1L,
                                trainable = FALSE) {
  BL  <- fit$Beta_Lambda
  p   <- nrow(BL); q <- ncol(X); k <- ncol(fit$M); d <- q + k
  H   <- ncol(fit$Gamma)
  z   <- qnorm(1 - alpha / 2)
  v_M <- if (is.null(m_prior_var)) 1 / k else m_prior_var
  obs <- if (is.null(observed)) matrix(0, 0, 0) else observed * 1.0
  X_tilde <- cbind(X, fit$M)
  Xi_mat  <- as.matrix(Xi_mat)              # H x p
  sig     <- as.vector(fit$sigma_sq)
  sig_mat <- matrix(sig, p, d, byrow = TRUE)
  # with a community mean mu (last Gamma column, constant last embedding row),
  # mu has prior variance mu_prior_var instead of the ridge kappa_r^2
  mu_col  <- isTRUE(fit$trait_intercept)
  ridge   <- function(r, Hr) c(rep(1 / fit$kappa_sq[r], Hr - mu_col), if (mu_col) 1 / fit$mu_prior_var)

  # (1) integrate out M
  V_eta <- uq_site_cov_cpp(X_tilde, BL, obs, k, v_M, block)
  s1    <- uq_species_diag_cpp(X_tilde, BL, obs, V_eta, sig, sig_mat, block, cores)

  # (2) REML-corrected σ² posterior, as in compute_UQ()
  XiXit  <- tcrossprod(Xi_mat)
  a_post <- fit$a + p / 2
  tr_G   <- vapply(seq_len(d), function(r)
              sum(diag(solve(XiXit / sig[r] + diag(ridge(r, H)), XiXit))), 0)
  b_post <- sig * (a_post + 1) + (colSums(s1$V_cond) + tr_G) / 2
  sig_bar <- b_post / (a_post - 1)

  if (trainable) {
    # (3'-4') Xi integrated out jointly with Gamma, block-Jacobi in species
    # (Xi prior precision 2(tau I + gamma L_norm), only its diagonal kept); the
    # p x H^2 products are accumulated over species blocks.  With mu (last Gamma
    # column, constant last embedding row) Gamma keeps all H columns, and only
    # the other Hf = H - 1 embedding coordinates are integrated out.
    Hf  <- H - mu_col
    G_f <- fit$Gamma[, seq_len(Hf), drop = FALSE]
    A   <- uq_species_A_cpp(X_tilde, BL, obs, V_eta, block, cores)
    # [L_norm]_jj = 1 unless species j shares no taxonomy level with another
    # species; a singleton at the top level is taken as isolated.
    top <- if (is.null(fit$taxonomy_codes)) rep(1L, p) else fit$taxonomy_codes[, 1]
    q_d <- 2 * (fit$tau_xi + fit$xi_gamma * (tabulate(top)[top] > 1))
    p1  <- uq_trainable_pass1_cpp(A, sig_bar, G_f, q_d, cores)
    vecxx <- function(J) { Xt <- t(Xi_mat[, J, drop = FALSE])     # rows vec(xi_j xi_j')
      Xt[, rep(seq_len(H), H), drop = FALSE] * Xt[, rep(seq_len(H), each = H), drop = FALSE] }
    blocks <- split(seq_len(p), ceiling(seq_len(p) / block))
    TT <- matrix(0, H * H, d * d)
    for (J in blocks) TT <- TT + crossprod(vecxx(J), t(p1$M[, J, drop = FALSE]))
    spec <- if (!is.null(fit$xi_spec)) fit$xi_spec else xi_laplacian_spectrum(fit$taxonomy_codes)  # log-det Hessian of exact Xi integration
    LH  <- matrix(0, d * H, d * H); i_f <- seq_len(d * Hf)
    LH[i_f, i_f] <- xi_logdet_hessian(G_f, sig_bar, 2 * (fit$tau_xi + fit$xi_gamma * pmax(spec$theta, 0)), spec$w)
    V_G <- solve(matrix(aperm(array(TT, c(H, H, d, d)), c(3, 1, 4, 2)), d * H) +
                 diag(as.vector(t(sapply(seq_len(d), ridge, Hr = H)))) + LH)
    Vp  <- matrix(aperm(array(V_G, c(d, H, d, H)), c(1, 3, 2, 4)), d * d)
    K   <- matrix(0, d * d, p)
    for (J in blocks) K[, J] <- Vp %*% t(vecxx(J))
    p2  <- uq_trainable_pass2_cpp(A, p1$W, K, p1$U, sig_bar, cores)
    G_sd <- matrix(sqrt(pmax(diag(V_G), 0)), d, H); D_mat <- p2$D; V_marg <- p2$V
  } else {
  # (3) Gamma: species weighted by data precision h_jr, Monte Carlo over σ²
  h_mat  <- pmax(1 / s1$V_eff - 1 / sig_mat, 0)
  hb_mat <- h_mat * BL + (BL - t(fit$Gamma %*% Xi_mat)) / sig_mat
  G_sd   <- matrix(NA_real_, d, H)
  D_mat  <- matrix(NA_real_, p, d)
  for (r in seq_len(d)) {
    draws <- 1 / rgamma(S_gamma, shape = a_post, rate = b_post[r])
    V_avg <- matrix(0, H, H); M_s <- matrix(0, H, S_gamma)
    for (s in seq_len(S_gamma)) {
      den <- 1 + draws[s] * h_mat[, r]
      V_s <- solve(Xi_mat %*% (h_mat[, r] / den * t(Xi_mat)) + diag(ridge(r, H)))
      V_avg     <- V_avg + V_s / S_gamma
      M_s[, s]  <- V_s %*% (Xi_mat %*% (hb_mat[, r] / den))
    }
    V_tot      <- V_avg + if (S_gamma > 1) cov(t(M_s)) * (S_gamma - 1) / S_gamma else 0
    G_sd[r, ]  <- sqrt(diag(V_tot))
    D_mat[, r] <- colSums(Xi_mat * (V_tot %*% Xi_mat))
  }

  # (4) species coefficients: prior widened to σ̄² + D_j
  V_marg <- uq_species_diag_cpp(X_tilde, BL, obs, V_eta, sig,
                                matrix(sig_bar, p, d, byrow = TRUE) + D_mat,
                                block, cores)$V_eff
  }
  sd_BL  <- sqrt(V_marg)
  ci <- function(est, sd) list(hat = est, sd = sd,
                               ci_lower = est - z * sd, ci_upper = est + z * sd)
  list(Beta     = ci(BL[, seq_len(q), drop = FALSE], sd_BL[, seq_len(q), drop = FALSE]),
       Lambda   = ci(BL[, q + seq_len(k), drop = FALSE], sd_BL[, q + seq_len(k), drop = FALSE]),
       Gamma    = ci(fit$Gamma, G_sd),
       sigma_sq = list(mean = sig_bar,
                       ci_lower = 1 / qgamma(1 - alpha / 2, a_post, rate = b_post),
                       ci_upper = 1 / qgamma(alpha / 2, a_post, rate = b_post)),
       V_cond = s1$V_cond, V_marg = V_marg, rho_sq = V_marg / s1$V_cond)
}
