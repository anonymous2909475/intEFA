# =============================================================================
# calc_polychoric.R
# =============================================================================
# DESCRIPTION:
#   MATLAB-to-R Bridge Execution Engine for Polychoric and Mixed Correlations.
#   This script is invoked automatically by the MATLAB routine `processData.m`
#   when at least one ordinal variable is detected in raw data matrix X.
#
# INTERACTION PROTOCOL:
#   Input File  : 'temp_X_data.mat'  (Contains matrix X_proc, ordinal_cols, continuous_cols)
#   Output File : 'temp_R_poly.mat'  (Exports R_poly, num_ordinal, num_continuous)
#   Engine      : R psych package (psych::polychoric, psych::mixedCor)
#
# ARCHITECTURAL ROLE:
#   Variable classification and integer rounding are performed upstream in MATLAB 
#   prior to standardization. This script receives exact column indices directly, 
#   bypassing redundant data-auditing loops and executing correlation 
#   estimation directly via R's `psych` framework.
#
# REFERENCES:
#   1. R psych Package:
#      Revelle, W. (2024). psych: Procedures for Psychological, Psychometric, 
#      and Personality Research. Northwestern University, Evanston, Illinois. 
#      R package version 2.4.3. https://CRAN.R-project.org/package=psych
#
#   2. Polychoric Estimation Theory:
#      Olsson, U. (1979). Maximum likelihood estimation of the polychoric correlation 
#      coefficient. Psychometrika, 44(4), 443–460. 
#      https://doi.org/10.1007/BF02296207
#
#   3. Mixed/Polyserial Estimation Theory:
#      Olsson, U., Drasgow, F., & Dorans, N. J. (1982). The polyserial correlation 
#      coefficient. Psychometrika, 47(3), 337–347. 
#      https://doi.org/10.1007/BF02294164
# =============================================================================

suppressPackageStartupMessages({
  suppressWarnings({
    library(psych)
    library(R.matlab)
  })
})

input_file  <- "temp_X_data.mat"
output_file <- "temp_R_poly.mat"

if (file.exists(input_file)) {
  
  # ---------------------------------------------------------------------------
  # 1. Load Data & Classification Metadata from MATLAB MAT-File
  # ---------------------------------------------------------------------------
  data_struct <- readMat(input_file)
  
  # Note: R.matlab converts MATLAB underscores ('X_proc') to dots ('X.proc')
  X    <- as.matrix(data_struct$X.proc)
  J    <- ncol(X)
  X_df <- as.data.frame(X)
  
  # Extract explicit index vectors passed directly from MATLAB's audit
  ordinal_cols    <- as.vector(data_struct$ordinal.cols)
  continuous_cols <- as.vector(data_struct$continuous.cols)
  
  # Clean potential NULL or NaN empty vectors from MAT-file import
  if (length(ordinal_cols) == 1 && is.na(ordinal_cols[1])) ordinal_cols <- c()
  if (length(continuous_cols) == 1 && is.na(continuous_cols[1])) continuous_cols <- c()
  
  # ---------------------------------------------------------------------------
  # 2. Correlation Matrix Estimation via psych::mixedCor / psych::polychoric
  # ---------------------------------------------------------------------------
  R_poly <- NULL
  
  tryCatch({
    if (length(continuous_cols) == 0 || is.null(continuous_cols)) {
      # --- PATH A: PURE ORDINAL MATRIX ---
      # Convert all columns to ordered factors required by psych::polychoric
      X_factors <- as.data.frame(lapply(X_df, function(col) factor(col, ordered = TRUE)))
      poly_res  <- suppressWarnings(psych::polychoric(X_factors, smooth = TRUE, global = FALSE, na.rm = TRUE))
      R_poly    <- poly_res$rho
      
    } else if (length(ordinal_cols) == 0 || is.null(ordinal_cols)) {
      # --- PATH B: PURE CONTINUOUS FALLBACK ---
      # Handled upstream by MATLAB, but guarded here for completeness
      R_poly <- cor(X, use = "pairwise.complete.obs")
      
    } else {
      # --- PATH C: MIXED (ORDINAL + CONTINUOUS) MATRIX ---
      # Cast ordinal variables as integer vectors required by psych::mixedCor
      for (ord_idx in ordinal_cols) {
        X_df[[ord_idx]] <- as.integer(round(X_df[[ord_idx]]))
      }
      mixed_res <- suppressWarnings(psych::mixedCor(data = X_df, c = continuous_cols, p = ordinal_cols, use = "pairwise"))
      R_poly    <- mixed_res$rho
    }
  }, error = function(e) {
    cat("[R-Bridge] ERROR during mixed/polychoric correlation computation:\n")
    cat(sprintf("          %s\n", e$message))
  })
  
  # ---------------------------------------------------------------------------
  # 3. Fallback Guardrail & Export Payload to MATLAB
  # ---------------------------------------------------------------------------
  if (is.null(R_poly) || !is.matrix(R_poly) || any(dim(R_poly) != c(J, J))) {
    cat("[R-Bridge] WARNING: Optimization failed or returned invalid matrix. Falling back to Pearson correlation.\n")
    R_poly  <- cor(X, use = "pairwise.complete.obs")
    num_ord <- 0
    num_con <- J
  } else {
    num_ord <- length(ordinal_cols)
    num_con <- length(continuous_cols)
  }
  
  # Export estimated correlation matrix and variable metadata back to MATLAB
  writeMat(output_file, 
           R_poly         = as.matrix(R_poly), 
           num_ordinal    = as.numeric(num_ord), 
           num_continuous = as.numeric(num_con))
}