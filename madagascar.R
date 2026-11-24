# madagascar.R — reproducible main results on the Madagascar arthropod data.
#
# Fits the reported FLAIR++ configuration (K = 40 latent factors, H = 40 trait
# dimensions, centered quadratic climate, quadratic spatial trend) and prints the Task-1 (new sites,
# 2 folds) and Task-2 (masked species, 2 folds) AUC by prevalence stratum.
# Run:  Rscript madagascar.R           (one node, ~30 cores, ~1.5 h total)
# Outputs: results_madagascar/{summary.txt, results.rds, fits.RData, prior_t1_fold*.rds}

options(warn = -1)
suppressMessages({ library(MASS); library(matrixStats); library(rsvd); library(Rcpp) })

FLAIR_CODE_DIR <- Sys.getenv("FLAIR_CODE_DIR", unset = {
  f <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(f)) dirname(normalizePath(sub("^--file=", "", f[1]))) else
    tryCatch(dirname(sys.frame(1)$ofile), error = function(e) getwd())
})
Sys.setenv(FLAIR_CODE_DIR = FLAIR_CODE_DIR)
DATA_DIR <- Sys.getenv("FLAIR_DATA_DIR", unset = "/work/ml752/FLAIR++/data/madagascar")
EMB_FILE <- Sys.getenv("FLAIR_EMB_FILE",
  unset = "/work/ml752/FLAIR++/tuning/trait_embeddings_trail.RData")
RESULTS_DIR <- Sys.getenv("FLAIR_RESULTS_DIR", unset = file.path(FLAIR_CODE_DIR, "results_madagascar"))
dir.create(RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)

# ---- reported configuration ----
K <- 40L; H <- 40L
PRE <- 80L; IT <- 70L            # pretrain / train iterations
VDIAG_FREQ <- 10L                # V_diag refresh every this many iterations
XI_LATE_FREEZE <- 50L            # freeze Xi after this iteration
AB <- 0.020; AL <- as.numeric(Sys.getenv("FLAIR_ALPHA_LAM", "0.016"))        # blockwise sigma^2 floors: b_r = alpha_r * p
MPS <- 0.5                       # m_prior_var = MPS / K
XI_LAMBDA <- 10; C_CLIP <- 50; M_STEP <- 0.5
XI_HYPER_EM <- Sys.getenv("FLAIR_XI_HYPER_EM", "0") == "1"   # EM (tau_xi, xi_gamma) update
KAPPA_D <- as.numeric(Sys.getenv("FLAIR_KAPPA_D", "4"))     # kappa^2 ~ IG(3, KAPPA_D)
TRAIT_INTERCEPT <- Sys.getenv("FLAIR_TRAIT_INTERCEPT", "1") == "1"   # community mean mu
set.seed(42)

source(file.path(FLAIR_CODE_DIR, "flair_plus.R"))

# ---- xi hyperparam stats cache: reuse the prior sufficient statistics in the
# Xi log-prior instead of a second D + Laplacian pass each iteration ----
.xi_ss_cache <- new.env(parent = emptyenv())
.compute_xi_ss_raw <- compute_xi_ss_cpp
compute_xi_ss_cpp <- function(Xi_t, Phi_t, codes, xi_gamma) {
  res <- .compute_xi_ss_raw(Xi_t, Phi_t, codes, xi_gamma)
  .xi_ss_cache$key <- sum(abs(Xi_t[1L, ] - Phi_t[1L, ])) + nrow(Xi_t) + 1e6 * ncol(Xi_t)
  .xi_ss_cache$sumsq <- as.numeric(res$sumsq)
  .xi_ss_cache$tr_DLD <- res$tr_DLD
  res
}
xi_prior_term_taxonomy_cpp <- function(Xi_t, Phi_t, codes, tau_vec, xi_gamma) {
  key <- sum(abs(Xi_t[1L, ] - Phi_t[1L, ])) + nrow(Xi_t) + 1e6 * ncol(Xi_t)
  if (!is.null(.xi_ss_cache$key) && isTRUE(all.equal(key, .xi_ss_cache$key))) {
    return(-sum(tau_vec * .xi_ss_cache$sumsq) - xi_gamma * .xi_ss_cache$tr_DLD)
  }
  ss <- compute_xi_ss_cpp(Xi_t, Phi_t, codes, xi_gamma)
  -sum(tau_vec * ss$sumsq) - xi_gamma * ss$tr_DLD
}

# ---- blockwise sigma^2: per-dimension IG scale b_r = alpha_r * p gives the
# soft floor sigma^2_r ~ 2*alpha_r (beta block: AB; Lambda block: AL). ----
Q_DIM <- NA_integer_
.alpha_vec <- function(d) c(rep(AB, Q_DIM), rep(AL, d - Q_DIM))
.p_current <- NA_real_
install_blockwise <- function(p_species) {
  .p_current <<- p_species
  update_sigma_sq_cpp <<- function(Beta_Lambda, Gamma, Xi_mat, a, b, lb, ub) {
    .p_current <<- nrow(Beta_Lambda)
    update_sigma_sq_blockwise_cpp(Beta_Lambda, Gamma, Xi_mat, a,
                                  .alpha_vec(ncol(Beta_Lambda)), ub)
  }
  .update_sigma_xi_em_raw <<- update_sigma_xi_em
  update_sigma_xi_em <<- function(Beta_Lambda, Gamma, Xi_mat, V_diag, sigma_sq_vec,
                                  kappa_sq_vec, a, b, ...) {
    .p_current <<- nrow(Beta_Lambda)
    .update_sigma_xi_em_raw(Beta_Lambda, Gamma, Xi_mat, V_diag, sigma_sq_vec, kappa_sq_vec,
                            a, .alpha_vec(ncol(Beta_Lambda)) * nrow(Beta_Lambda), ...)
  }
  compute_sigma_prior_term <<- function(sigma_sq_vec, a, b) {
    bv <- .alpha_vec(length(sigma_sq_vec)) * .p_current
    sum(a * log(bv) - lgamma(a) - (a + 1) * log(sigma_sq_vec) - bv / sigma_sq_vec)
  }
}

# ---- data ----
load(file.path(DATA_DIR, "allData.RData"))
load(EMB_FILE)
mt <- new.env(); load(file.path(DATA_DIR, "MT_Madagascar.RData"), envir = mt)

da <- as.Date(meta$COLL_DATE)
jday <- as.numeric(julian(da) - julian(as.Date("2021-01-01")))
XData <- data.frame(seqdepth = log(meta$readCount), temp = meta$temp,
                    prec = meta$prec, aridity = meta$aridity,
                    co1 = cos(2*pi*jday/365), si1 = sin(2*pi*jday/365),
                    co2 = cos(4*pi*jday/365), si2 = sin(4*pi*jday/365))
X <- cbind(intercept = 1, as.matrix(scale(XData)))
zt <- scale(XData$temp); zp <- scale(XData$prec)  # centered quadratic climate
X <- cbind(X, scale(cbind(temp2 = zt^2, prec2 = zp^2, tp = zt * zp)))
zla <- scale(as.numeric(meta$SITE_LATITUDE))  # smooth spatial trend in site coordinates
zlo <- scale(as.numeric(meta$SITE_LONGITUDE))
X <- cbind(X, scale(cbind(la = zla, lo = zlo, la2 = zla^2, lo2 = zlo^2, lalo = zla * zlo)))
Q_DIM <- ncol(X)
n <- nrow(Y)
sample_ids <- rownames(Y)
P1.F1 <- mt$P1.F1; P2.F1.full <- mt$P2.F1
if (!is.null(sample_ids) && !is.null(names(P1.F1))) P1.F1 <- P1.F1[sample_ids]
if (!is.null(sample_ids) && !is.null(rownames(P2.F1.full)))
  P2.F1.full <- P2.F1.full[sample_ids, , drop = FALSE]
P1.F1 <- as.logical(P1.F1)
if (is.null(colnames(mt$sel.sp))) colnames(mt$sel.sp) <- colnames(mt$Y)
if (is.null(colnames(mt$common.sp))) colnames(mt$common.sp) <- colnames(mt$sel.sp)
if (is.null(colnames(P2.F1.full))) colnames(P2.F1.full) <- colnames(mt$sel.sp)

sigmoid <- function(x) 1 / (1 + exp(-x))
safe_mean <- function(x) { x <- x[!is.na(x)]; if (!length(x)) NA_real_ else mean(x) }
fmt_f <- function(x, d = 3) ifelse(is.na(x) | is.nan(x), "NA", formatC(x, format = "f", digits = d))
stableAUC <- function(y, p) {
  ok <- !is.na(y) & !is.na(p); y <- as.logical(y[ok]); p <- p[ok]
  if (!length(y) || !any(y) || all(y)) return(NA_real_)
  mean(rank(p)[y] - seq_len(sum(y))) / sum(!y)
}
logloss <- function(y, p, eps = 1e-12) {
  p <- pmin(pmax(p, eps), 1 - eps)
  -mean(y * log(p) + (1 - y) * log(1 - p))
}
compute_metrics_from_vectors <- function(auc_v, r2_v, is_common, prev_v,
                                         task, elapsed_s, ll_v = NULL) {
  is_focal <- !is_common
  list(task = task, elapsed_s = elapsed_s, H = H, K = K,
       aucMean = safe_mean(auc_v), aucVec = auc_v,
       aucCommon = safe_mean(auc_v[is_common]),
       aucFocal = safe_mean(auc_v[is_focal]),
       aucFocalPrev1 = safe_mean(auc_v[is_focal & prev_v == 1L]),
       aucFocalPrevG1 = safe_mean(auc_v[is_focal & prev_v > 1L]),
       llVec = ll_v,
       llCommon = safe_mean(ll_v[is_common]),
       llFocalPrev1 = safe_mean(ll_v[is_focal & prev_v == 1L]),
       llFocalPrevG1 = safe_mean(ll_v[is_focal & prev_v > 1L]),
       tjurR2Mean = safe_mean(r2_v), tjurR2Focal = safe_mean(r2_v[!is_common]))
}
mask_response <- function(Ymat, obs) {
  Y_masked <- Ymat; obs_bool <- obs == 1L
  fill <- vapply(seq_len(ncol(Ymat)), function(j) {
    vals <- Ymat[obs_bool[, j], j]; if (!length(vals)) 0 else mean(vals)
  }, numeric(1))
  for (j in seq_len(ncol(Ymat))) Y_masked[!obs_bool[, j], j] <- fill[j]
  Y_masked
}
available_cores_safe <- function() {
  slurm_n <- suppressWarnings(as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "")))
  if (!is.na(slurm_n) && slurm_n >= 1L) return(slurm_n)
  max(1L, parallel::detectCores(logical = FALSE))
}
make_t1 <- function(fold) {
  idx_tr <- if (fold == 1L) which(P1.F1) else which(!P1.F1)
  list(idx_tr = idx_tr, idx_te = setdiff(seq_len(n), idx_tr))
}
make_t2 <- function(fold, P2, common_names) {
  mat <- if (fold == 1L) P2 else !P2
  mat[, common_names] <- TRUE
  list(matTrain = mat, YTestMask = !mat)
}
stream_metrics <- function(Yo, fold_infos, task, is_common, prev_v, elapsed_s,
                           species_block = 1000L) {
  p <- ncol(Yo); auc_v <- r2_v <- ll_v <- rep(NA_real_, p)
  blocks <- split(seq_len(p), ceiling(seq_len(p) / species_block))
  res <- parallel::mclapply(blocks, function(cols) {
    fa <- fr <- fl <- matrix(NA_real_, 2L, length(cols))
    for (fi in 1:2) {
      info <- fold_infos[[fi]]
      if (task == 1L) {
        idx <- info$idx_pred
        P <- sigmoid(X[idx, , drop = FALSE] %*% t(info$Beta[cols, , drop = FALSE]))
        rows_list <- rep(list(seq_along(idx)), length(cols))
        Ysub <- Yo[idx, cols, drop = FALSE]
      } else {
        P <- sigmoid(X %*% t(info$Beta[cols, , drop = FALSE]) +
                       info$M %*% t(info$Lambda[cols, , drop = FALSE]))
        mask <- info$mask[, cols, drop = FALSE]
        rows_list <- lapply(seq_along(cols), function(jj) which(mask[, jj]))
        Ysub <- Yo[, cols, drop = FALSE]
      }
      for (jj in seq_along(cols)) {
        rows <- rows_list[[jj]]; if (!length(rows)) next
        y_j <- Ysub[rows, jj]; p_j <- P[rows, jj]
        fl[fi, jj] <- logloss(y_j, p_j)
        if (!any(y_j == 1L) || !any(y_j == 0L)) next
        fa[fi, jj] <- stableAUC(y_j, p_j)
        cls <- as.logical(y_j); fr[fi, jj] <- mean(p_j[cls]) - mean(p_j[!cls])
      }
    }
    list(auc = colMeans(fa, na.rm = TRUE), r2 = colMeans(fr, na.rm = TRUE),
         ll = colMeans(fl, na.rm = TRUE))
  }, mc.cores = available_cores_safe())
  for (b in seq_along(blocks)) {
    auc_v[blocks[[b]]] <- res[[b]]$auc; r2_v[blocks[[b]]] <- res[[b]]$r2
    ll_v[blocks[[b]]] <- res[[b]]$ll
  }
  compute_metrics_from_vectors(auc_v, r2_v, is_common, prev_v, task, elapsed_s, ll_v)
}

# ---- species selection (all species) and embedding ----
sp <- colnames(mt$sel.sp)[which(mt$sel.sp[1L, ])]
common_names <- intersect(colnames(mt$common.sp)[which(mt$common.sp[1L, ])], sp)
YA <- Y[, sp, drop = FALSE]
prevA <- colSums(YA)
is_common <- colnames(YA) %in% common_names
codes <- make_taxonomy_path_codes(taxonomy, sp)
emb <- embeddings$method3_diffusion
stopifnot(all(sp %in% rownames(emb)), H <= ncol(emb))
Xi_sel <- t(emb[sp, seq_len(H), drop = FALSE])
rownames(Xi_sel) <- paste0("trait_", seq_len(H)); colnames(Xi_sel) <- sp
cat(sprintf("Madagascar: n=%d p=%d common=%d focal=%d q=%d K=%d H=%d\n",
            n, ncol(YA), sum(is_common), sum(!is_common), Q_DIM, K, H))

install_blockwise(ncol(YA))

bb_cols <- match(common_names, colnames(YA)); bb_cols <- bb_cols[!is.na(bb_cols)]
Y_bb_names <- colnames(YA)[bb_cols]
Xi_bb <- Xi_sel[, bb_cols, drop = FALSE]
codes_bb <- codes[bb_cols, , drop = FALSE]
P2.F1_all <- P2.F1.full[, colnames(YA), drop = FALSE]
cores_use <- max(1L, available_cores_safe() - 2L)
Sys.setenv(OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1", VECLIB_MAXIMUM_THREADS = "1")

fit_one <- function(Y_fit, X_fit, obs_fit) {
  full_obs <- length(obs_fit) == 1L && is.na(obs_fit)
  obs_bb <- if (full_obs) NA else obs_fit[, bb_cols, drop = FALSE]
  FLAIR_plus_minibatch_pretrain_wrapper(
    Y_fit[, Y_bb_names, drop = FALSE], X_fit, Xi_bb, k = K,
    Phi_mat = Xi_bb, taxonomy_codes = codes_bb,
    Y_all = Y_fit, Xi_all = Xi_sel, Phi_all = Xi_sel,
    taxonomy_codes_all = codes,
    xi_trainable = TRUE, xi_lambda = XI_LAMBDA, xi_gamma = 0.1,
    xi_e = 0.1, xi_f = 0.5, xi_g = 0.1, xi_h = 0.2,
    hyper_xi_prior = "ig", xi_maxit = 50L,
    observed = obs_bb, observed_all = obs_fit,
    iter_max_pretrain = PRE, iter_max_train = IT,
    C_clip = C_CLIP, BL_species_batch_size = 1000L, cores = cores_use,
    randomize_batches = TRUE,
    stage1_m_update = "species_minibatch", stage2_m_update = "species_minibatch",
    M_species_batch_size = 1000L,
    m_step_size = M_STEP, m_lhat_max_ls = 8L,
    m_prior_var = MPS / K,
    do_post_process = TRUE,
    vdiag_freq = VDIAG_FREQ, xi_late_freeze = XI_LATE_FREEZE,
    kappa_prior = "ig", kappa_d = KAPPA_D, xi_hyper_em = XI_HYPER_EM,
    trait_intercept = TRAIT_INTERCEPT,
    sigma_a = 3,
    sigma_b_pretrain = length(Y_bb_names),
    sigma_b_train = 5000
  )
}

# ---- main results: Task 1 (new sites) and Task 2 (masked species), 2 folds ----
fold_infos_t1 <- vector("list", 2L); fold_infos_t2 <- vector("list", 2L)
xi_task1 <- vector("list", 2L)
el_t1 <- el_t2 <- 0
for (fold in 1:2) {
  set.seed(1000L + fold)
  s1 <- make_t1(fold)
  t0 <- proc.time()
  fit <- fit_one(YA[s1$idx_tr, , drop = FALSE], X[s1$idx_tr, , drop = FALSE], NA)
  el_t1 <- el_t1 + (proc.time() - t0)[3]
  fold_infos_t1[[fold]] <- list(Beta = fit$Beta, idx_pred = s1$idx_te)
  xi_task1[[fold]] <- fit$Xi_mat
  saveRDS(list(Gamma = fit$Gamma, sigma_sq = fit$sigma_sq, kappa_sq = fit$kappa_sq,
               tau_xi = fit$tau_xi, xi_gamma = fit$xi_gamma),
          file.path(RESULTS_DIR, sprintf("prior_t1_fold%d.rds", fold)))
  cat(sprintf("== task1 fold%d done (%.0fs)\n", fold, (proc.time() - t0)[3]))

  set.seed(2000L + fold)
  s2 <- make_t2(fold, P2.F1_all, common_names)
  obs <- matrix(as.integer(s2$matTrain), nrow(YA), ncol(YA))
  t0 <- proc.time()
  fit2 <- fit_one(mask_response(YA, obs), X, obs)
  el_t2 <- el_t2 + (proc.time() - t0)[3]
  fold_infos_t2[[fold]] <- list(Beta = fit2$Beta, Lambda = fit2$Lambda,
                                M = fit2$M, mask = s2$YTestMask)
  cat(sprintf("== task2 fold%d done (%.0fs)\n", fold, (proc.time() - t0)[3]))
}

res_t1 <- stream_metrics(YA, fold_infos_t1, 1L, is_common, prevA, el_t1 / 2)
res_t2 <- stream_metrics(YA, fold_infos_t2, 2L, is_common, prevA, el_t2 / 2)
results <- list(t1 = res_t1, t2 = res_t2)
xi_trained <- list(0.5 * (xi_task1[[1]] + xi_task1[[2]]))

fmt_row <- function(task, r) sprintf("%-4d %-8s %-10s %-10s %-10s %-10s %-11s",
  task, fmt_f(r$elapsed_s, 0), fmt_f(r$aucMean), fmt_f(r$aucCommon), fmt_f(r$aucFocal),
  fmt_f(r$aucFocalPrev1), fmt_f(r$aucFocalPrevG1))
lines <- c(strrep("=", 100),
           sprintf("FLAIR++ Madagascar main results  K=%d H=%d PRE=%d IT=%d", K, H, PRE, IT),
           sprintf("%-4s %-8s %-10s %-10s %-10s %-10s %-11s",
                   "Task", "Time(s)", "AUC", "AUC_common", "AUC_focal", "AUC_foc1", "AUC_foc>1"),
           fmt_row(1, res_t1), fmt_row(2, res_t2), strrep("=", 100))
cat(paste(lines, collapse = "\n"), "\n")
writeLines(lines, file.path(RESULTS_DIR, "summary.txt"))
saveRDS(results, file.path(RESULTS_DIR, "results.rds"))
save(results, xi_trained, P1.F1, file = file.path(RESULTS_DIR, "fits.RData"))
cat("saved to", RESULTS_DIR, "\n")

# ---- optional repeats / sweeps (commented out) --------------------------------
# To repeat with a different seed, rerun with set.seed(<seed>) above; to sweep
# the K x H grid or baselines, submit one job per setting with K/H overridden.
