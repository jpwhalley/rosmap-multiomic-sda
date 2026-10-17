# Independent proportional-odds check with MASS::polr (different code path from the Python PO used by neuronal_c21/dossier).
# Score as ordered response (each distinct value a level), logistic link; OR = odds of a higher score. Aggregates only.
suppressMessages(library(MASS))
# paths: G_ROS_RESULTS_DIR as in scripts/round2/config.py (run after v00_frame.py)
RES <- Sys.getenv("G_ROS_RESULTS_DIR"); if (RES == "") stop("set G_ROS_RESULTS_DIR")
ME <- file.path(RES, "revision_analyses/verify_neuronal_c21")
F <- read.csv(file.path(ME, "cache/frame.csv"))
COV <- c("age", "pmi", "ros", "male")
fitpo <- function(y, d, terms) {
  d$Y <- factor(rank(y, ties.method = "min"), ordered = TRUE)
  f <- as.formula(paste("Y ~", paste(c(terms, COV), collapse = " + ")))
  m <- polr(f, data = d, method = "logistic", Hess = TRUE, control = list(maxit = 5000, reltol = 1e-12))
  m
}
res <- list()
add <- function(k, set, model, m, terms, d, y, blocks) {
  ci <- tryCatch(suppressMessages(confint(m, parm = terms)), error = function(e) NULL)
  ll <- as.numeric(logLik(m))
  for (b in names(blocks)) {
    others <- setdiff(unlist(blocks), blocks[[b]])
    m0 <- fitpo(y, d, others)
    pLR <- pchisq(2 * (ll - as.numeric(logLik(m0))), df = length(blocks[[b]]), lower.tail = FALSE)
    for (t in blocks[[b]]) {
      lo <- if (is.null(ci)) NA else if (is.matrix(ci)) ci[t, 1] else ci[1]
      hi <- if (is.null(ci)) NA else if (is.matrix(ci)) ci[t, 2] else ci[2]
      res[[length(res) + 1]] <<- data.frame(component = k, set = set, model = model, n = nrow(d), term = t,
        OR = exp(coef(m)[t]), OR_lo = exp(lo), OR_hi = exp(hi), block_LR_p = pLR, conv = m$convergence)
    }
  }
}
for (k in c(172, 277)) {
  for (set in c("cogdx124", "all276", "cogdx124_noRNAmm", "all276_noRNAmm")) {
    d <- F
    if (grepl("cogdx124", set)) d <- d[d$cogdx %in% c(1, 2, 4), ]
    if (grepl("noRNAmm", set)) d <- d[d$rna_mismatch == 0, ]
    y <- d[[paste0("s", k)]]
    B <- list(CERAD = "cerad_ad", Braak = "braak", cogdx = c("mci", "addem"))
    if (grepl("all276", set)) { d$mci <- as.numeric(d$cogdx %in% c(2, 3)); d$addem <- as.numeric(d$cogdx %in% c(4, 5, 6)) }
    combos <- list("CERAD", "Braak", "cogdx", c("CERAD", "Braak", "cogdx"))
    for (cb in combos) {
      bl <- B[cb]; terms <- unlist(bl)
      m <- fitpo(y, d, terms); add(k, set, paste(cb, collapse = "+"), m, terms, d, y, bl)
    }
    m <- fitpo(y, d, "cerad_burden"); add(k, set, "CERADburden", m, "cerad_burden", d, y, list(CERAD = "cerad_burden"))
    d$mmse5 <- -d$mmse / 5
    bl <- list(CERAD = "cerad_ad", Braak = "braak", MMSE = "mmse5"); m <- fitpo(y, d, unlist(bl)); add(k, set, "CERAD+Braak+MMSE", m, unlist(bl), d, y, bl)
  }
}
R <- do.call(rbind, res)
write.csv(R, file.path(ME, "out/v01_polr.csv"), row.names = FALSE)
options(width = 200); print(R, digits = 3, row.names = FALSE)
