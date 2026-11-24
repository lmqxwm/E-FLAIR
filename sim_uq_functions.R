# Shared DGP and method runners for the UQ simulation tables
# (tab:uq_coverage, tab:uq_compare, coverage-vs-p). Sourced by
# numerical_exp.rmd (reproduction) and tuning/sim_uq_run.R (cluster driver);
# all seeds are explicit so both produce identical numbers.
# DGP: q = 2 (+ intercept), k = 3, H = 20; sigma^2 = 0.25, kappa^2 = 1 (all rows);
# two-level taxonomy (genus of 10, species-group of 5); Phi by genus;
# Xi0 ~ MN(Phi, Q^{-1}, I), Q = 2(tau I + 2 L_norm) with tau = 100, so the
# taxonomy anchor explains the embedding and the species-level deviation is a
# refinement smaller than the residual; mu_0 = (-3, 0.5, -0.3, 0, 0, 0).
# Rarity arises inside the hierarchy (sim_rep, rare = "model"), with the
# paper's prevalence regimes as categories.

# sigma2 / kappa2: fixed residual and trait-map variances (all rows); NULL draws
# them from InvGamma(3, b_scale). The fitter's InvGamma(3, b) priors are matched
# to them through b_sigma, b_kappa (prior mean b / 2 equals the fixed value).
sim_setting <- function(p, H = 20L, q = 2L, k = 3L, seed = 4242L,
                        tau = 100, mu_alpha = -3, b_scale = 0.5, C_alpha = 3,
                        sigma2 = 0.25, kappa2 = 1, gamma = 2) {
  d <- q + 1L + k
  set.seed(seed + p)
  sigma0 <- if (is.null(sigma2)) 1 / rgamma(d, 3, rate = b_scale) else rep(sigma2, d)
  kappa0 <- if (is.null(kappa2)) 1 / rgamma(d, 3, rate = b_scale) else rep(kappa2, d)
  b_sigma <- if (is.null(sigma2)) b_scale else 2 * sigma2
  b_kappa <- if (is.null(kappa2)) b_scale else 2 * kappa2
  tax <- cbind(rep(seq_len(p / 10), each = 10), rep(seq_len(p / 5), each = 5))
  Phi <- matrix(rnorm(H * (p / 10), sd = sqrt(4 / H)), H, p / 10)[, tax[, 1]]
  A <- outer(tax[, 2], tax[, 2], "==") * 1; diag(A) <- 0
  Dg <- rowSums(A)
  L <- diag(p) - sweep(sweep(A, 1, sqrt(Dg), "/"), 2, sqrt(Dg), "/")
  eL <- eigen(L, symmetric = TRUE)
  Qih <- eL$vectors %*% (t(eL$vectors) / sqrt(2 * (tau + gamma * pmax(eL$values, 0))))
  Xi0 <- Phi + matrix(rnorm(H * p), H, p) %*% Qih
  # hold the coefficient signal constant across tau, as the paper's E||xi_j||^2 = 4 does
  # across H: kappa^2 is scaled so that Var(Gamma xi_j) = kappa2 * 4 for every tau
  if (!is.null(kappa2)) {
    kappa0 <- kappa0 * 4 / mean(colSums(Xi0^2)); b_kappa <- 2 * kappa0[1]
  }
  mu0 <- c(mu_alpha, 0.5, -0.3, rep(0, k))
  Gamma0 <- matrix(rnorm(d * H), d, H) * matrix(sqrt(kappa0), d, H)
  list(p = p, H = H, q = q, k = k, d = d, sigma0 = sigma0, kappa0 = kappa0,
       tax = tax, Phi = Phi, L = L, Xi0 = Xi0, mu0 = mu0, Gamma0 = Gamma0,
       C_alpha = C_alpha, b_sigma = b_sigma, b_kappa = b_kappa, tau = tau, gamma = gamma)
}

# rare = "model": rarity arises inside the hierarchy (intercepts spread by
# mu_0 + Gamma xi_j + e_j, so it follows the taxonomy); categories as in the
# paper (Section prevalence regimes), with pi_j = n_j / n:
# (c) super-rare pi_j < n^{-1/2}; (b) semi-rare alpha_0j < -C_alpha; (a) common.
# rare = "shift": 40% of intercepts get an extra shift s_j unrelated to the
# taxonomy, on an interval widened until each category holds >= 15%.
sim_rep <- function(st, n, r, seed = 20000L, rare = c("model", "shift")) {
  rare_mode <- match.arg(rare)
  p <- st$p; q <- st$q; k <- st$k; d <- st$d
  set.seed(seed * r + st$p + n)
  BL <- matrix(st$mu0, p, d, byrow = TRUE) + t(st$Gamma0 %*% st$Xi0) +
        matrix(rnorm(p * d), p, d) * matrix(sqrt(st$sigma0), p, d, byrow = TRUE)
  rare <- sample(p, round(0.4 * p))
  u_j  <- runif(length(rare))
  X  <- cbind(1, matrix(rnorm(n * q), n, q))
  Et <- matrix(rnorm(n * k), n, k) / sqrt(k)
  Xt  <- cbind(1, matrix(rnorm(n * q), n, q))
  Ett <- matrix(rnorm(n * k), n, k) / sqrt(k)
  lin <- function(B) X %*% t(B[, seq_len(q + 1)]) + Et %*% t(B[, q + 1L + seq_len(k)])
  if (rare_mode == "model") {
    BL0 <- BL; s_j <- numeric(0); rare <- integer(0); iv <- NA
    Z0 <- lin(BL0); nj <- colSums(plogis(Z0))
    cat3 <- ifelse(nj / n < n^-0.5, "c", ifelse(BL0[, 1] < -st$C_alpha, "b", "a"))
  } else for (iv in list(c(-5, -1), c(-7, -3), c(-3, -0.5), c(-9, -5))) {
    BL0 <- BL; s_j <- iv[1] + u_j * (iv[2] - iv[1])
    BL0[rare, 1] <- BL0[rare, 1] + s_j
    Z0 <- lin(BL0); nj <- colSums(plogis(Z0))
    cat3 <- ifelse(!(seq_len(p) %in% rare), "a", ifelse(nj > log(p), "b", "c"))
    if (min(table(factor(cat3, c("a", "b", "c")))) >= 0.15 * p) break
  }
  Y  <- matrix(rbinom(n * p, 1, plogis(Z0)), n, p); colnames(Y) <- paste0("sp", seq_len(p))
  Zt <- Xt %*% t(BL0[, seq_len(q + 1)]) + Ett %*% t(BL0[, q + 1L + seq_len(k)])
  Yt <- matrix(rbinom(n * p, 1, plogis(Zt)), n, p)
  # probit twin data from the same parameters, for HMSC evaluated on its own link
  Yp  <- matrix(rbinom(n * p, 1, pnorm(Z0)), n, p); colnames(Yp) <- colnames(Y)
  Ytp <- matrix(rbinom(n * p, 1, pnorm(Zt)), n, p)
  A0 <- solve(crossprod(X), crossprod(X, Et))
  Beta_c <- BL0[, seq_len(q + 1)] + BL0[, q + 1L + seq_len(k)] %*% t(A0)
  Om_T <- tcrossprod(BL0[, q + 1L + seq_len(k)]) / k
  list(n = n, r = r, BL0 = BL0, s_j = s_j, rare = rare, interval = iv,
       X = X, Y = Y, Xt = Xt, Yt = Yt, Yp = Yp, Ytp = Ytp, nj = nj, cat3 = cat3,
       A0 = A0, Beta_c = Beta_c, Om_T = Om_T)
}

sim_coverage <- function(lo, hi, tr) mean(lo <= tr & tr <= hi, na.rm = TRUE)
sim_auc <- function(y, s) { if (min(y) == max(y)) return(NA_real_)
  r <- rank(s); n1 <- sum(y); (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * (sum(1 - y))) }

# marginal test-set probabilities, eta integrated by MC (shared draws)
sim_pred <- function(Xt, Beta, Lambda, k, link = plogis, S = 200L, seed = 1L) {
  set.seed(seed)
  Hs <- matrix(rnorm(S * k), S, k) / sqrt(k)
  lin <- Xt %*% t(Beta)                 # n x p
  P <- 0
  for (s in seq_len(S)) P <- P + link(lin + rep(Hs[s, ] %*% t(Lambda), each = nrow(Xt)))
  P / S
}

sim_pred_rows <- function(rp, Beta, Lambda, k, link = plogis) {
  P <- sim_pred(rp$Xt, Beta, Lambda, k, link)
  auc <- vapply(seq_len(ncol(P)), function(j) sim_auc(rp$Yt[, j], P[, j]), 0)
  ll  <- -colMeans(rp$Yt * log(pmax(P, 1e-12)) + (1 - rp$Yt) * log(pmax(1 - P, 1e-12)))
  list(auc = auc, logloss = ll)
}

# per-category aggregate rows for one replicate
sim_agg <- function(method, flavor, param, cat3, cov_by, wid_by, rmse_by, rp, time = NA) {
  data.frame(method = method, flavor = flavor, param = param, rep = rp$r,
             cat = names(cov_by), cov = as.numeric(cov_by),
             width = as.numeric(wid_by[names(cov_by)]),
             rmse = as.numeric(rmse_by[names(cov_by)]), time = time)
}

# elementwise Omega coverage/width/rmse by category (pairs -> rarer side)
sim_omega_stats <- function(lo, hi, hat, rp) {
  ci3 <- c(a = 1, b = 2, c = 3)[rp$cat3]
  pcat <- outer(ci3, ci3, pmax); up <- upper.tri(rp$Om_T, diag = TRUE)
  f <- function(v, cc) v[up & pcat == cc]
  sapply(1:3, function(cc) c(
    cov  = sim_coverage(f(lo, cc), f(hi, cc), f(rp$Om_T, cc)),
    wid  = mean(f(hi, cc) - f(lo, cc), na.rm = TRUE),
    rmse = sqrt(mean((f(hat, cc) - f(rp$Om_T, cc))^2, na.rm = TRUE))))
}

sim_cat_stats <- function(lo, hi, hat, tr, cat3) {
  sapply(c("a", "b", "c"), function(cc) { j <- cat3 == cc
    c(cov  = sim_coverage(lo[j, , drop = FALSE], hi[j, , drop = FALSE], tr[j, , drop = FALSE]),
      wid  = mean(hi[j, ] - lo[j, ], na.rm = TRUE),
      rmse = sqrt(mean((hat[j, ] - tr[j, ])^2, na.rm = TRUE))) })
}

sim_pack <- function(method, flavor, rp, lo, hi, hat, om, time, pred) {
  sa <- sim_cat_stats(lo[, 1, drop = FALSE], hi[, 1, drop = FALSE],
                      hat[, 1, drop = FALSE], rp$Beta_c[, 1, drop = FALSE], rp$cat3)
  sb <- sim_cat_stats(lo[, -1, drop = FALSE], hi[, -1, drop = FALSE],
                      hat[, -1, drop = FALSE], rp$Beta_c[, -1, drop = FALSE], rp$cat3)
  agg <- rbind(
    sim_agg(method, flavor, "alpha", rp$cat3, sa["cov", ], sa["wid", ], sa["rmse", ], rp, time),
    sim_agg(method, flavor, "beta",  rp$cat3, sb["cov", ], sb["wid", ], sb["rmse", ], rp, time))
  if (!is.null(om)) {
    so <- sim_omega_stats(om$lo, om$hi, om$hat, rp); colnames(so) <- c("a", "b", "c")
    agg <- rbind(agg, sim_agg(method, flavor, "Omega", rp$cat3,
                              so["cov", ], so["wid", ], so["rmse", ], rp, time))
  }
  sp <- data.frame(method = method, flavor = flavor, rep = rp$r, sp = seq_len(nrow(hat)),
                   cat = rp$cat3, n_j = rp$nj, alpha_true = rp$Beta_c[, 1],
                   alpha_hat = hat[, 1], alpha_lo = lo[, 1], alpha_hi = hi[, 1],
                   auc = pred$auc, logloss = pred$logloss)
  list(agg = agg, species = sp)
}

# ── FLAIR+ (pretrain wrapper, trainable embedding initialised at Xi0 and anchored at Phi) ─
# xi_init: starting embedding ("Xi0" the truth, "Phi" the anchor); with
# trainable = FALSE the embedding stays at xi_init
sim_run_flairp <- function(st, rp, cores = 4L, trainable = TRUE, hyper_em = TRUE,
                           xi_init = c("Xi0", "Phi"), hyper_fixed = FALSE) {
  X0 <- st[[match.arg(xi_init)]]
  common <- which(rp$cat3 == "a")
  t0 <- proc.time()[3]
  # priors match the DGP; kappa^2 lives in the fitter's frame, where each
  # embedding row is standardized by its backbone sd s_h, so Gamma* = Gamma0 diag(s).
  # Works with the root fitter (kappa_b, xi_hyper_em) and with the paper fitter
  # FLAIR+_code/flair_plus.R (kappa_c, kappa_d): unsupported arguments are dropped.
  s2 <- mean(apply(X0[, common, drop = FALSE], 1, var)); kd <- st$b_kappa * s2
  args <- list(rp$Y[, common, drop = FALSE], rp$X, X0[, common, drop = FALSE], st$k,
    Y_all = rp$Y, Xi_all = X0,
    Phi_mat = st$Phi[, common, drop = FALSE], Phi_all = st$Phi,
    taxonomy_codes = st$tax[common, , drop = FALSE], taxonomy_codes_all = st$tax,
    xi_trainable = trainable, trait_intercept = TRUE, xi_hyper_em = hyper_em, xi_gamma_cov = TRUE,
    xi_hyper_fixed = hyper_fixed, xi_lambda = st$tau * s2, xi_gamma = st$gamma * s2,
    sigma_a = 3, sigma_b_pretrain = st$b_sigma, sigma_b_train = st$b_sigma,
    kappa_b = kd, kappa_c = 3, kappa_d = kd,
    loss_tol = 1e-5, iter_max_train = 200L, cores = cores)
  fm <- names(formals(FLAIR_plus_minibatch_pretrain_wrapper))
  fit <- do.call(FLAIR_plus_minibatch_pretrain_wrapper, args[names(args) == "" | names(args) %in% fm])
  # scalable UQ at every p: it works in the post-processed coordinates of the
  # fit (compute_UQ_joint's map->Tm branch mis-scales Lambda for pretrain fits)
  uq <- compute_UQ_scalable(fit, rp$X, fit$Xi_mat, trainable = trainable)
  Vm <- uq$V_marg; Vc <- uq$V_cond
  tt <- proc.time()[3] - t0
  q <- st$q; k <- st$k
  hat <- cbind(uq$Beta$hat, uq$Lambda$hat)
  lo_m <- cbind(uq$Beta$ci_lower, uq$Lambda$ci_lower)
  hi_m <- cbind(uq$Beta$ci_upper, uq$Lambda$ci_upper)
  sdc <- sqrt(Vc); lo_c <- hat - 1.96 * sdc; hi_c <- hat + 1.96 * sdc
  omega_ci <- function(V) {
    Lh <- hat[, q + 1L + seq_len(k)]; L2 <- Lh^2
    Vl <- V[, q + 1L + seq_len(k)]
    vx <- Vl %*% t(L2); v <- vx + t(vx); diag(v) <- 4 * rowSums(L2 * Vl)
    sdv <- sqrt(pmax(v, 0)); OmH <- tcrossprod(Lh)
    list(lo = OmH - 1.96 * sdv, hi = OmH + 1.96 * sdv, hat = OmH)
  }
  pred <- sim_pred_rows(rp, hat[, seq_len(q + 1)], hat[, q + 1L + seq_len(k)], k)
  out_m <- sim_pack("FLAIR+", "marginal", rp, lo_m[, seq_len(q + 1)], hi_m[, seq_len(q + 1)],
                    hat[, seq_len(q + 1)], omega_ci(Vm), tt, pred)
  out_c <- sim_pack("FLAIR+", "plug-in", rp, lo_c[, seq_len(q + 1)], hi_c[, seq_len(q + 1)],
                    hat[, seq_len(q + 1)], omega_ci(Vc), tt, pred)
  # Gamma_{[1:q+1,]} and mu in the fitter frame (backbone standardization + canonical A0);
  # mu_alpha truth is the shift-adjusted population mean.
  m_bb <- rowMeans(X0[, common]); s_bb <- apply(X0[, common], 1, sd)
  Gs0 <- st$Gamma0 * matrix(s_bb, st$d, st$H, byrow = TRUE)
  mu_s <- st$mu0 + as.vector(st$Gamma0 %*% m_bb); mu_s[1] <- mu_s[1] + sum(rp$s_j) / st$p
  Gc  <- Gs0[seq_len(q + 1), ] + rp$A0 %*% Gs0[q + 1L + seq_len(k), ]
  muc <- mu_s[seq_len(q + 1)] + as.vector(rp$A0 %*% mu_s[q + 1L + seq_len(k)])
  Hf <- ncol(fit$Gamma) - 1L
  gl <- uq$Gamma$ci_lower; gh <- uq$Gamma$ci_upper
  agg_g <- data.frame(method = "FLAIR+", flavor = "marginal",
    param = c("Gamma", "mu"), rep = rp$r, cat = "global",
    cov = c(sim_coverage(gl[seq_len(q + 1), seq_len(Hf)], gh[seq_len(q + 1), seq_len(Hf)],
                         Gc[, seq_len(Hf)]),
            sim_coverage(gl[seq_len(q + 1), Hf + 1L], gh[seq_len(q + 1), Hf + 1L], muc)),
    width = c(mean(gh[seq_len(q + 1), seq_len(Hf)] - gl[seq_len(q + 1), seq_len(Hf)]),
              mean(gh[seq_len(q + 1), Hf + 1L] - gl[seq_len(q + 1), Hf + 1L])),
    rmse = c(sqrt(mean((uq$Gamma$hat[seq_len(q + 1), seq_len(Hf)] - Gc[, seq_len(Hf)])^2)),
             sqrt(mean((uq$Gamma$hat[seq_len(q + 1), Hf + 1L] - muc)^2))), time = tt)
  list(agg = rbind(out_m$agg, out_c$agg, agg_g),
       species = rbind(out_m$species, out_c$species))
}

# ── plain FLAIR (FLAIR-main; CC-corrected CIs, Omega from posterior samples) ──
sim_run_flair <- function(st, rp, proj = Sys.getenv("FLAIR_PROJ_DIR", "/work/ml752/FLAIR++")) {
  q <- st$q; k <- st$k; p <- st$p; n <- rp$n
  t0 <- proc.time()[3]
  fit <- FLAIR_wrapper(rp$Y, rp$X, k = k, alternate_max = 10, max_it = 200,
    tol = 1e-3, loss_tol = 1e-3, post_process = TRUE,
    observed = matrix(1L, n, p), randomized_svd = FALSE,
    provide_posterior_samples = TRUE, n_MC = 200,
    C_lambda = 10, C_mu = 10, C_beta = 10,
    subsample_index = seq_len(p), backend = "c++")
  tt <- proc.time()[3] - t0
  ci <- compute_ci_Beta(fit$Beta_tilde, fit$Vs[, seq_len(q + 1), drop = FALSE],
                        fit$rho_max, alpha = 0.05)
  os <- matrix(fit$Lambda_outer_samples_cc, p * p)   # (p x p x S) -> p^2 x S
  qs <- apply(os, 1, quantile, probs = c(0.025, 0.975))
  om <- list(lo = matrix(qs[1, ], p, p), hi = matrix(qs[2, ], p, p),
             hat = tcrossprod(fit$Lambda_tilde))
  pred <- sim_pred_rows(rp, fit$Beta_tilde, fit$Lambda_tilde, k)
  sim_pack("FLAIR", "marginal", rp, ci$ls, ci$us, fit$Beta_tilde, om, tt, pred)
}

# ── HMSC (probit; taxonomy correlation C, TrData = Phi; chunk conventions) ────
# HMSC is fitted on the probit twin data (same parameters, probit link) and
# evaluated on its own, correctly specified link: no probit-to-logit rescaling.
# traits = "Xi0": the true embedding (oracle trait data); "Phi": the taxonomy
# anchor, the information FLAIR+'s embedding prior is centered on
sim_run_hmsc <- function(st, rp, samples = 2000L, thin = 5L, transient = 1000L,
                         nChains = 2L, traits = c("Xi0", "Phi"), phylo = FALSE) {
  traits <- match.arg(traits)
  # phylo = TRUE: residual taxonomy correlation C (shared-level proportion, nested),
  # the HMSC analogue of the taxonomy prior on a trainable embedding around Phi
  C <- if (phylo) {
    Z1 <- outer(st$tax[, 1], st$tax[, 1], "==") * 1
    Z2 <- outer(st$tax[, 2], st$tax[, 2], "==") * 1
    structure((Z1 + Z2 + diag(st$p)) / 3, dimnames = list(colnames(rp$Y), colnames(rp$Y)))
  }
  q <- st$q; k <- st$k; p <- st$p; n <- rp$n
  XData <- as.data.frame(rp$X[, -1, drop = FALSE]); colnames(XData) <- paste0("x", seq_len(q))
  XFormula <- as.formula(paste("~", paste(colnames(XData), collapse = "+")))
  TrData <- as.data.frame(t(st[[traits]])); colnames(TrData) <- paste0("tr", seq_len(st$H))
  rownames(TrData) <- colnames(rp$Y)
  # trait intercept, matching FLAIR+'s community mean mu; no residual taxonomy
  # correlation C, since FLAIR+ with a fixed embedding has independent residuals
  TrFormula <- as.formula(paste("~", paste(colnames(TrData), collapse = "+")))
  sd_tr <- data.frame(sample = as.factor(seq_len(n)))
  # mild factor shrinkage: the default multiplicative-gamma prior (a1 = a2 = 50)
  # shrinks the loadings threefold here and collapses the Omega intervals
  rL <- setPriors(HmscRandomLevel(units = sd_tr$sample), nfMax = k, a1 = 2, a2 = 2)
  t0 <- proc.time()[3]
  m <- Hmsc(Y = rp$Yp, distr = "probit", XData = XData, XFormula = XFormula,
            TrData = TrData, TrFormula = TrFormula, C = C,
            studyDesign = sd_tr, ranLevels = list(sample = rL))
  m <- sampleMcmc(m, samples = samples, thin = thin, transient = transient,
                  nChains = nChains, verbose = 0)
  tt <- proc.time()[3] - t0
  # convergence and mixing of the sampler: PSRF and effective sample size for
  # Beta (all elements) and Omega (random subset of 300 entries)
  co <- convertToCodaObject(m, spNamesNumbers = c(TRUE, FALSE),
                            covNamesNumbers = c(TRUE, FALSE))
  ps_b <- coda::gelman.diag(co$Beta, multivariate = FALSE)$psrf[, 1]
  es_b <- coda::effectiveSize(co$Beta)
  set.seed(1); jo <- sort(sample(ncol(co$Omega[[1]][[1]]), 300))
  omc  <- coda::as.mcmc.list(lapply(co$Omega[[1]], function(ch) ch[, jo]))
  ps_o <- coda::gelman.diag(omc, multivariate = FALSE)$psrf[, 1]
  es_o <- coda::effectiveSize(omc)
  # Beta draws are nc x ns column-major: covariate index varies fastest
  pm <- t(matrix(ps_b, q + 1L, p))
  by_cat <- unlist(lapply(c("a", "b", "c"), function(cc) { j <- rp$cat3 == cc
    c(max(pm[j, 1]), max(pm[j, -1])) }))
  mcmc_diag <- data.frame(method = "HMSC", flavor = "marginal", param = "mcmc",
    rep = rp$r, cat = c("psrfB", "essB", "psrfO", "essO",
                        paste0(rep(c("psrfA_", "psrfBeta_"), 3), rep(c("a", "b", "c"), each = 2))),
    cov = c(max(ps_b, na.rm = TRUE), median(es_b), max(ps_o, na.rm = TRUE), median(es_o), by_cat),
    width = c(mean(ps_b > 1.1, na.rm = TRUE), min(es_b), mean(ps_o > 1.1, na.rm = TRUE), min(es_o),
              rep(NA, 6)),
    rmse = NA, time = tt)
  pe <- getPostEstimate(m, parName = "Beta", q = c(0.025, 0.975))
  lo <- t(pe$q[1, , ]); hi <- t(pe$q[2, , ]); hat <- t(pe$mean)
  po <- getPostEstimate(m, parName = "Omega", r = 1, q = c(0.025, 0.975))
  om <- list(lo = matrix(po$q[1, , ], p, p), hi = matrix(po$q[2, , ], p, p),
             hat = matrix(po$mean, p, p))
  sd_te <- data.frame(sample = as.factor(n + seq_len(nrow(rp$Xt))))
  XD_te <- as.data.frame(rp$Xt[, -1, drop = FALSE]); colnames(XD_te) <- colnames(XData)
  pr <- predict(m, XData = XD_te, studyDesign = sd_te, ranLevels = list(sample = rL),
                expected = TRUE)
  P <- Reduce(`+`, pr) / length(pr)
  pred <- list(auc = vapply(seq_len(p), function(j) sim_auc(rp$Ytp[, j], P[, j]), 0),
               logloss = -colMeans(rp$Ytp * log(pmax(P, 1e-12)) +
                                   (1 - rp$Ytp) * log(pmax(1 - P, 1e-12))))
  # the data identify the coefficients only up to the part of the factors that
  # is collinear with X, so HMSC is scored against the canonical value like the
  # other methods; the raw value is kept as a secondary flavor
  out <- sim_pack("HMSC", "marginal", rp, lo, hi, hat, om, tt, pred)
  rp_h <- rp; rp_h$Beta_c <- rp$BL0[, seq_len(q + 1)]
  raw <- sim_pack("HMSC", "raw", rp_h, lo, hi, hat, NULL, tt, pred)$agg
  out$agg <- rbind(out$agg, raw, mcmc_diag)
  out$est <- list(hat = hat, lo = lo, hi = hi)
  out
}

# run one grid cell (method x p x n x replicates) and save per-rep results
sim_run_cell <- function(method, p, n, reps, out_rds, cores = 4L, hmsc_args = list()) {
  st <- sim_setting(p)
  agg <- list(); spc <- list()
  for (r in reps) {
    rp <- sim_rep(st, n, r)
    res <- switch(method,
      flairp = sim_run_flairp(st, rp, cores = cores),
      flair  = sim_run_flair(st, rp),
      hmsc   = do.call(sim_run_hmsc, c(list(st, rp), hmsc_args)))
    res$agg$n <- n; res$agg$p <- p; res$species$n <- n; res$species$p <- p
    agg[[length(agg) + 1]] <- res$agg; spc[[length(spc) + 1]] <- res$species
    cat(sprintf("[%s p=%d n=%d] rep %d done (cats %s; interval %s)\n", method, p, n,
                r, paste(table(factor(rp$cat3, c("a","b","c"))), collapse = "/"),
                paste(rp$interval, collapse = ",")))
    saveRDS(list(agg = do.call(rbind, agg), species = do.call(rbind, spc)), out_rds)
  }
  invisible(readRDS(out_rds))
}
