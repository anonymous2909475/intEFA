function [Lambda_unrot, R_smooth, Z, data_info] = processData(X, Q, opts)
% PROCESSDATA Handles pre-standardization ordinal detection, smart correlation
%             matrix estimation (Pearson vs. Polychoric R-Bridge bypass), 
%             positive semi-definite (PSD) spectral projection, multivariate 
%             normality diagnostics, and unrotated factor extraction (ML or PAF).
%
% Syntax:
%   [Lambda_unrot, R_smooth, Z, data_info] = processData(X, Q)
%   [Lambda_unrot, R_smooth, Z, data_info] = processData(X, Q, opts)

    % =========================================================================
    % INPUT VALIDATION & INITIALIZATION
    % =========================================================================
    if nargin < 3, opts = struct(); end
    
    [N, J] = size(X);
    data_info = struct();
    data_info.NumObservations = N;
    data_info.NumVariables    = J;
    data_info.NumFactors      = Q;
    
    fprintf('\n-----------------------------------------------------------------------\n');
    fprintf('[processData.m] AUDIT & PRE-PROCESSING REPORT\n');
    fprintf('-----------------------------------------------------------------------\n');
    fprintf('Dimensions                : Observations (N) = %d | Variables (J) = %d | Factors (Q) = %d\n', N, J, Q);

    % =========================================================================
    % STEP 1: VARIABLE CLASSIFICATION & DISCREPANCY AUDIT
    % =========================================================================
    data_type_opt = get_opt(opts, 'data_type', 2);
    
    % Always execute automated heuristic check for baseline cross-validation
    [auto_ord_cols, auto_cont_cols, audit_details] = detect_variable_types(X, opts);
    auto_mask = zeros(1, J);
    auto_mask(auto_ord_cols) = 1;
    
    has_type_mismatch = false;
    mismatch_msg = '';
    
    % Parse Input Mode
    if isnumeric(data_type_opt) && numel(data_type_opt) == J
        % --- MODE A: Binary Mask Vector ---
        user_mask = data_type_opt(:)';
        ordinal_cols    = find(user_mask == 1);
        continuous_cols = find(user_mask == 0);
        fprintf('Data Type Mode            : Custom Binary Mask Vector Provided\n');
        
        if ~isequal(user_mask, auto_mask)
            has_type_mismatch = true;
            mismatch_msg = sprintf('  [!] WARNING: User mask differs from automated heuristic check.\n      Input Mask    : %s\n      Estimated Mask: %s\n      Note: Ignore automated check if ground truth is known. User mask retained.', ...
                mat2str(user_mask), mat2str(auto_mask));
        end
        
    elseif isscalar(data_type_opt)
        switch data_type_opt
            case 0
                % --- MODE B: Force Pure Continuous ---
                user_mask = zeros(1, J);
                ordinal_cols    = [];
                continuous_cols = 1:J;
                fprintf('Data Type Mode            : Forced Pure Continuous (Pearson Correlation)\n');
                
                if ~isempty(auto_ord_cols)
                    has_type_mismatch = true;
                    mismatch_msg = sprintf('  [!] WARNING: Analysis forced to Continuous, but automated heuristic detected %d ordinal item(s).\n      Input Mask    : %s\n      Estimated Mask: %s\n      Note: Ignore automated check if ground truth is known.', ...
                        length(auto_ord_cols), mat2str(user_mask), mat2str(auto_mask));
                end
                
            case 1
                % --- MODE C: Force Pure Ordinal ---
                user_mask = ones(1, J);
                ordinal_cols    = 1:J;
                continuous_cols = [];
                fprintf('Data Type Mode            : Forced Pure Ordinal (Polychoric Correlation)\n');
                
                if ~isempty(auto_cont_cols)
                    has_type_mismatch = true;
                    mismatch_msg = sprintf('  [!] WARNING: Analysis forced to Ordinal, but automated heuristic detected %d continuous item(s).\n      Input Mask    : %s\n      Estimated Mask: %s\n      Note: Ignore automated check if ground truth is known.', ...
                        length(auto_cont_cols), mat2str(user_mask), mat2str(auto_mask));
                end
                
            case 2
                % --- MODE D: Automated 3-Check Voting Engine ---
                user_mask       = auto_mask;
                ordinal_cols    = auto_ord_cols;
                continuous_cols = auto_cont_cols;
                fprintf('Data Type Mode            : Automatic Detection Engine Enabled\n');
                fprintf('Detected Binary Mask      : %s\n', mat2str(auto_mask));
                
            otherwise
                error('processData:InvalidDataTypeOpt', ...
                    'Invalid scalar opts.data_type. Expected 0 (Continuous), 1 (Ordinal), 2 (Auto), or a 1xJ mask.');
        end
    else
        error('processData:InvalidDataTypeOpt', ...
            'opts.data_type must be a scalar (0, 1, 2) or a vector matching variable length J.');
    end
    
    num_ordinal    = length(ordinal_cols);
    num_continuous = length(continuous_cols);
    
    data_info.NumOrdinalItems   = num_ordinal;
    data_info.NumContinuous     = num_continuous;
    data_info.OrdinalIndices    = ordinal_cols;
    data_info.ContinuousIndices = continuous_cols;
    data_info.DetectedMask      = auto_mask;
    data_info.AuditDetails      = audit_details;
    
    fprintf('Variable Classification   : %d Ordinal Item(s) | %d Continuous Item(s)\n', ...
        num_ordinal, num_continuous);

    % Vectorized rounding for Ordinal columns
    X_proc = X;
    if ~isempty(ordinal_cols)
        X_proc(:, ordinal_cols) = round(X_proc(:, ordinal_cols));
    end

    % Standardize data matrix Z (with zero-variance protection)
    var_x = var(X, 0, 1);
    Z = zeros(size(X));
    if any(var_x == 0)
        warning('processData:ZeroVariance', ...
            'Zero-variance variables detected. Standardizing with zero-variance protection.');
        valid_cols = var_x > 0;
        Z(:, valid_cols) = zscore(X(:, valid_cols));
    else
        Z = zscore(X);
    end

    % =========================================================================
    % STEP 2: CORRELATION MATRIX ESTIMATION (Dynamic Routing)
    % =========================================================================
    if num_ordinal == 0
        % --- PATH A: Pure Continuous Data ---
        R = corrcoef(Z);
        data_info.CorrelationType = 'Pearson (Pure Continuous - MATLAB)';
        
    else
        % --- PATH B: Mixed / Ordinal Data (Invoke R-Bridge Engine) ---
        user_docs = fileparts(userpath); 
        r_folder  = fullfile(user_docs, 'rfiles', 'matlab_bridge');
        if ~exist(r_folder, 'dir'), mkdir(r_folder); end
        
        mat_in   = fullfile(r_folder, 'temp_X_data.mat');
        mat_out  = fullfile(r_folder, 'temp_R_poly.mat');
        r_script = fullfile(r_folder, 'calc_polychoric.R');
        
        % Export without overwriting local X variable
        save(mat_in, 'X_proc', 'ordinal_cols', 'continuous_cols', '-v7');
        
        % Locate dynamic Rscript binary on Windows systems
        rscript_cmd = 'Rscript';
        if ispc
            r_dirs = dir('C:\Program Files\R\R-*');
            if ~isempty(r_dirs)
                latest_r = r_dirs(end).name;
                rscript_bin = fullfile('C:\Program Files\R', latest_r, 'bin', 'Rscript.exe');
                if exist(rscript_bin, 'file')
                    rscript_cmd = sprintf('"%s"', rscript_bin);
                end
            end
        end
        
        cmd = sprintf('%s "%s"', rscript_cmd, r_script);
        curr_dir = cd(r_folder);
        status = system(cmd);
        cd(curr_dir); % Restore active directory
        
        if status == 0 && exist(mat_out, 'file')
            loaded_data = load(mat_out);
            R = loaded_data.R_poly;
            
            if isempty(R) || ~isequal(size(R), [J, J])
                warning('processData:EmptyRFromBridge', ...
                    'R bridge returned an invalid matrix size. Falling back to Pearson correlation.');
                R = corrcoef(Z);
                data_info.CorrelationType = 'Pearson (Fallback from Failed R-Bridge)';
            else
                if num_continuous == 0
                    data_info.CorrelationType = 'Polychoric (Pure Ordinal - via R psych::polychoric)';
                else
                    data_info.CorrelationType = 'Mixed Polyserial/Polychoric (via R psych::mixedCor)';
                end
            end
            
            if exist(mat_in, 'file'), delete(mat_in); end
            if exist(mat_out, 'file'), delete(mat_out); end
        else
            error('processData:RBridgeFailed', ...
                'Failed to execute Rscript for Polychoric computation. Verify R installation and script path.');
        end
    end

    % =========================================================================
    % STEP 3: POSITIVE SEMI-DEFINITE (PSD) SPECTRAL PROJECTION
    % =========================================================================
    if any(~isfinite(R(:)))
        warning('processData:NonFiniteValues', 'Non-finite values found in correlation matrix R. Replacing with zeros.');
        R(~isfinite(R)) = 0;
        R(1:J+1:end) = 1;
    end
    
    tol = 1e-6;
    R_sym = (R + R') / 2; % Enforce exact mathematical symmetry
    
    [V, D] = eig(R_sym);
    eig_vals = real(diag(D));
    V = real(V);
    
    min_eig = min(eig_vals);
    data_info.MinEigenvalue = min_eig;
    
    if min_eig < tol
        data_info.WasPSDProjected = true;
        fprintf('PSD Projection Status     : Applied (Matrix was Non-PSD | Smallest Eigenvalue = %.4e)\n', min_eig);
        
        eig_vals_psd = max(eig_vals, tol);
        R_smooth = V * diag(eig_vals_psd) * V';
        
        % Rescale diagonal elements cleanly to 1.0
        d_scale = sqrt(diag(R_smooth));
        R_smooth = R_smooth ./ (d_scale * d_scale');
    else
        data_info.WasPSDProjected = false;
        fprintf('PSD Projection Status     : Not Required (Smallest Eigenvalue = %.4e)\n', min_eig);
        R_smooth = R_sym;
    end
    
    R_smooth(1:J+1:end) = 1; % Force exact 1.0 main diagonal

    % =========================================================================
    % STEP 4: NORMALITY DIAGNOSTICS & FACTOR EXTRACTION
    % =========================================================================
    [Lambda_unrot, Psi, diag_info] = select_efa_method(R_smooth, Q, Z, num_ordinal);
    
    % Merge diagnostic statistics into output metadata struct
    fields = fieldnames(diag_info);
    for k = 1:length(fields)
        data_info.(fields{k}) = diag_info.(fields{k});
    end
    data_info.Uniquenesses = Psi;
    
    fprintf('Multivariate Normality    : %s\n', string(diag_info.IsGaussian));
    fprintf('Factor Extraction Method  : %s\n', diag_info.ChosenMethod);
    
    % Output variable classification discrepancy warning if user input overrides heuristic
    if has_type_mismatch
        fprintf('\n%s\n', mismatch_msg);
    end
    
    fprintf('-----------------------------------------------------------------------\n\n');
end

% =========================================================================
% INTERNAL HELPER 1: MULTI-CRITERIA VARIABLE TYPE DETECTION ENGINE
% =========================================================================
function [ordinal_cols, continuous_cols, audit_report] = detect_variable_types(X, opts)
    if nargin < 2, opts = struct(); end
    
    max_ord_levels = get_opt(opts, 'max_ord_levels', 15);   
    max_card_ratio = get_opt(opts, 'max_card_ratio', 0.05);  
    min_step_ratio = get_opt(opts, 'min_step_ratio', 0.95);  
    
    [~, J] = size(X);
    ordinal_cols = [];
    continuous_cols = [];
    audit_report = struct();
    
    for j = 1:J
        col = X(:, j);
        col_finite = col(isfinite(col));
        N_valid = length(col_finite);
        
        if N_valid == 0
            continuous_cols = [continuous_cols, j]; %#ok<AGROW>
            continue;
        end
        
        u_vals = unique(col_finite);
        U = length(u_vals);
        card_ratio = U / N_valid;
        
        % --- CHECK 1: Cardinality Cutoff ---
        check1_cutoff = (U <= max_ord_levels);
        
        % --- CHECK 2: Cardinality Ratio ---
        check2_ratio = (card_ratio <= max_card_ratio);
        
        % --- CHECK 3: Step/Delta Uniformity ---
        if U > 1
            diffs_rounded = round(diff(sort(u_vals)), 4);
            [step_counts, ~] = groupcounts(diffs_rounded);
            step_uniform_ratio = max(step_counts) / length(diffs_rounded);
            check3_step = (step_uniform_ratio >= min_step_ratio);
        else
            step_uniform_ratio = 1.0;
            check3_step = true;
        end
        
        % --- MAJORITY VOTING RULE (2 out of 3) ---
        votes = [check1_cutoff, check2_ratio, check3_step];
        is_ordinal = (sum(votes) >= 2);
        
        if is_ordinal
            ordinal_cols = [ordinal_cols, j]; %#ok<AGROW>
        else
            continuous_cols = [continuous_cols, j]; %#ok<AGROW>
        end
        
        audit_report(j).Column            = j;
        audit_report(j).UniqueCount       = U;
        audit_report(j).CardinalityRatio  = card_ratio;
        audit_report(j).StepUniformRatio  = step_uniform_ratio;
        audit_report(j).Votes             = votes;
        audit_report(j).TotalVotes        = sum(votes);
        audit_report(j).ClassifiedAs      = ternary(is_ordinal, 'Ordinal', 'Continuous');
    end
end

% =========================================================================
% INTERNAL HELPER 2: SELECT EFA EXTRACTION METHOD
% =========================================================================
function [Lambda_unrot, Psi, selection_info] = select_efa_method(R, Q, Z, num_ordinal_found)
    [N, J] = size(Z);
    
    % 1. Univariate Skewness and Excess Kurtosis
    sk = skewness(Z, 0);          
    kt = kurtosis(Z, 0) - 3;      
    max_abs_skew = max(abs(sk));
    max_abs_kurt = max(abs(kt));
    
    % 2. Mardia's Normalized Multivariate Kurtosis
    S_cov = cov(Z);
    Z_centered = Z - mean(Z, 1);
    try
        if rcond(S_cov) < 1e-12
            D_sq = diag(Z_centered * (pinv(S_cov) * Z_centered'));
        else
            D_sq = diag(Z_centered * (S_cov \ Z_centered'));
        end
        raw_mardia = sum(D_sq.^2) / N;
        mardia_norm = (raw_mardia - J*(J+2)) / sqrt(8*J*(J+2)/N);
    catch
        mardia_norm = Inf;
    end
    
    % 3. Calculate Normalized Cutoff Bound
    mardia_norm_bound = 0.25 * sqrt(N * J * (J + 2) / 8);
    
    % 4. Decision Rule
    if num_ordinal_found > 0
        is_gaussian = false;
        selection_reason = sprintf('Ordinal data detected (%d item(s)). Polychoric matrix requires PAF.', num_ordinal_found);
    else
        is_gaussian = (max_abs_skew < 2.0) && ...
                      (max_abs_kurt < 7.0) && ...
                      (abs(mardia_norm) < mardia_norm_bound);
                  
        if is_gaussian
            selection_reason = 'Data passed univariate and multivariate Gaussian diagnostics.';
        else
            selection_reason = 'Continuous data failed Gaussianity thresholds (non-normal distribution).';
        end
    end
              
    % 5. Execute Factor Extraction
    if is_gaussian
        chosen_method = 'Maximum Likelihood (ML)';
        try
            [Lambda_unrot, Psi] = factoran(Z, Q, 'Rotate', 'none');
            
            if any(Psi <= 0.0051)
                warning('select_efa_method:HeywoodCase', ...
                    'ML hit a boundary constraint (Heywood Case). Falling back to PAF.');
                chosen_method = 'PAF (Fallback from ML Heywood Case)';
                [Lambda_unrot, Psi] = paf(R, Q);
            end
        catch
            warning('select_efa_method:MLConvergenceFailed', ...
                'ML factoran failed to converge. Falling back to PAF.');
            chosen_method = 'PAF (Fallback from ML Convergence Failure)';
            [Lambda_unrot, Psi] = paf(R, Q);
        end
    else
        chosen_method = sprintf('Principal Axis Factoring (PAF) [%s]', selection_reason);
        [Lambda_unrot, Psi] = paf(R, Q);
    end
    
    selection_info = struct();
    selection_info.ChosenMethod             = chosen_method;
    selection_info.SelectionReason          = selection_reason;
    selection_info.NumOrdinalItems          = num_ordinal_found;
    selection_info.IsGaussian               = is_gaussian;
    selection_info.MaxAbsSkewness           = max_abs_skew;
    selection_info.MaxAbsExcessKurtosis     = max_abs_kurt;
    selection_info.MardiaNormalizedKurtosis = mardia_norm;
    selection_info.MardiaNormalizedBound    = mardia_norm_bound;
end

% =========================================================================
% INTERNAL HELPER 3: UTILITY FUNCTIONS
% =========================================================================
function val = get_opt(opts, field, default_val)
    if isfield(opts, field) && ~isempty(opts.(field))
        val = opts.(field);
    else
        val = default_val;
    end
end

function res = ternary(cond, val_true, val_false)
    if cond, res = val_true; else, res = val_false; end
end