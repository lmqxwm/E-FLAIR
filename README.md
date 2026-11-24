# E-FLAIR

Scalable joint species distribution modelling with a taxonomy-informed trait
embedding and calibrated uncertainty. E-FLAIR (FLAIR+) extends FLAIR
(Mauri et al., 2024) to hundreds of thousands of species: rare species
borrow strength from relatives through a trainable embedding of the taxonomy,
and a variance corrector gives per-coordinate credible intervals.

## Model

For sample $i$ and species $j$, with $y_{ij}\in\{0,1\}$,

$$
\mathrm{logit}\,\Pr(y_{ij}=1)=\alpha_j+x_i^\top\beta_j+\eta_i^\top\lambda_j,\qquad \eta_i\sim N(0,v_M^2I_k),
$$

and the species coefficients $\tilde b_j=(\alpha_j,\beta_j^\top,\lambda_j^\top)^\top$ follow a trait prior

$$
\tilde b_j\sim N(\mu+\Gamma\xi_j,\ \Sigma_b),\qquad \Xi\sim N(\Phi,\ Q_\xi^{-1}),\quad Q_\xi=2(\tau_\xi I+\gamma_\xi L),
$$

where $\Phi$ is a prior embedding of the taxonomy (e.g. a diffusion map),
$L$ the normalized taxonomy Laplacian, and $\Gamma$ the trait map. The
residual association between species is $\Omega=\Lambda\Lambda^\top$.

Estimation is MAP by block coordinate ascent, in two stages: a backbone of
common species first, then all species, starting from their prior means.

## Requirements

- R (>= 4.3) with a C++ compiler supporting OpenMP
- R packages: `Rcpp`, `RcppArmadillo`, `MASS`, `matrixStats`, `rsvd`, `parallelly`
- Optional: `Hmsc` (simulation comparison only)

The C++ helpers are compiled by `Rcpp::sourceCpp` when the R files are sourced.

## Files

| File | Purpose |
|---|---|
| `flair_plus.R` | Main fitter, `FLAIR_plus_minibatch_pretrain_wrapper()` |
| `uq_scalable.R` | Variance corrector, `compute_UQ_scalable()` |
| `helper_functions_trustregion.cpp`, `flair_plus_minibatch_tr_pretrain.cpp`, `flair_tuning_helpers.cpp`, `helper_functions_uq.cpp` | C++ kernels (sourced automatically) |
| `madagascar.R`, `madagascar.py` | Reproduce the Madagascar application |
| `sim_uq_functions.R` | Simulation data-generating process and method runners |
| `numerical_exp.rmd` | Simulation study notebook |
| `flair_plus_minibatch_tr.R` | Earlier fitter used by parts of the notebook |

## Quick start

Run from the repository root:

```r
source("flair_plus.R")
source("uq_scalable.R")
source("sim_uq_functions.R")              # only for the toy data below

st <- sim_setting(p = 120)                # taxonomy, prior embedding, true parameters
rp <- sim_rep(st, n = 200, r = 1)         # one data set
bb <- which(rp$cat3 == "a")               # backbone: the common species

fit <- FLAIR_plus_minibatch_pretrain_wrapper(
  rp$Y[, bb], rp$X, st$Phi[, bb], k = 3,            # backbone data
  Y_all = rp$Y, Xi_all = st$Phi, Phi_all = st$Phi,  # all species
  taxonomy_codes = st$tax[bb, ], taxonomy_codes_all = st$tax,
  xi_trainable = TRUE, cores = 2)

uq <- compute_UQ_scalable(fit, rp$X, fit$Xi_mat, trainable = TRUE)
head(uq$Beta$ci_lower); head(uq$Beta$ci_upper)
```

Warnings about recycling a 1x1 array are harmless.

## Inputs

| Argument | Shape | Meaning |
|---|---|---|
| `Y`, `Y_all` | $n\times p_{\mathrm{bb}}$, $n\times p$ | 0/1 occurrences; `Y` holds the backbone columns of `Y_all` |
| `X` | $n\times(q+1)$ | design matrix, first column the intercept |
| `Xi_mat`, `Xi_all` | $H\times p_{\mathrm{bb}}$, $H\times p$ | initial embedding (usually equal to $\Phi$) |
| `Phi_mat`, `Phi_all` | $H\times p_{\mathrm{bb}}$, $H\times p$ | prior mean of the embedding |
| `taxonomy_codes`, `taxonomy_codes_all` | $p\times$ levels | integer group code of each species at each taxonomic level, coarse to fine |
| `k` | scalar | number of latent factors |

`make_taxonomy_path_codes(taxonomy, species, tax_levels)` builds the
taxonomy codes from a table with one column per rank. Unobserved cells are
passed as a 0/1 matrix `observed_all` (and `observed` for the backbone).

Frequently used options:

- `xi_trainable` — train the embedding (`TRUE`) or hold $\Xi=\Phi$ (`FALSE`)
- `trait_intercept` — add the community mean $\mu$ (default `TRUE`)
- `iter_max_pretrain`, `iter_max_train` — iteration caps of the two stages
- `xi_late_freeze` — freeze the embedding after this iteration (early stopping)
- `m_prior_var` — prior variance $v_M^2$ of the latent factors
- `sigma_a`, `sigma_b_pretrain`, `sigma_b_train` — InvGamma prior on $\sigma_r^2$
- `cores`, `BL_species_batch_size` — parallelism and species block size

## Outputs

`fit` is a list. The main elements:

| Element | Shape | Meaning |
|---|---|---|
| `Beta` | $p\times(q+1)$ | intercepts and environmental coefficients |
| `Lambda` | $p\times k$ | factor loadings ($\Omega=\Lambda\Lambda^\top$) |
| `M` | $n\times k$ | latent factors, orthogonal to `X` (post-processing is on by default) |
| `Gamma` | $(q+1+k)\times(H+1)$ | trait map; last column is $\mu$ when `trait_intercept = TRUE` |
| `Xi_mat` | $(H+1)\times p$ | fitted embedding; last row is the constant 1 |
| `sigma_sq`, `kappa_sq` | vectors | residual and trait-map variances |
| `tau_xi`, `xi_gamma` | scalars | embedding precision parameters |

`compute_UQ_scalable()` returns `Beta`, `Lambda` and `Gamma`, each with
`hat`, `sd`, `ci_lower`, `ci_upper` (95%), and the conditional and marginal
variances `V_cond`, `V_marg`.

For new samples `Xnew`, `plogis(Xnew %*% t(fit$Beta))` gives the occurrence
probabilities with the latent factors set to their prior mean 0.

## Reproducing the paper

Application (Madagascar arthropods, 255188 species):

```sh
FLAIR_DATA_DIR=/path/to/madagascar \
FLAIR_EMB_FILE=/path/to/trait_embeddings.RData \
Rscript madagascar.R            # or: python madagascar.py
```

The data are those of the benchmark of Ovaskainen et al. (2025)
(`allData.RData`, `MT_Madagascar.RData`) and are not included here. The script
fits the reported configuration ($k=40$, $H=40$) on both prediction tasks and
writes `results_madagascar/summary.txt` with the AUC by prevalence group. It
needs one node with about 32 cores and 64 GB of memory, and runs in about
1.5 hours.

Simulation: `numerical_exp.rmd` reproduces the simulation study. Set
`FLAIR_PROJ_DIR` at the top of the notebook to your project directory. The
comparison with plain FLAIR needs the original FLAIR code in
`FLAIR_PROJ_DIR/FLAIR-main`, and the HMSC comparison needs the `Hmsc` package.
