% =========================================================================
% MASTER DRIVER SCRIPT: EFA & MULTI-ROTATION EVALUATION PIPELINE
% =========================================================================
% OVERVIEW:
%   This script provides an end-to-end Exploratory Factor Analysis (EFA) 
%   pipeline that ingests raw observational data, executes dynamic variable 
%   classification (continuous vs. ordinal), extracts unrotated factor 
%   loadings via ML or PAF, benchmarks multiple factor rotations via `manyrot`, 
%   and evaluates interpretability metrics using `interp`.
%
% INPUT FILE REQUIREMENTS:
%   The script expects a workspace dataset or MAT-file containing a 
%   2D matrix variable named `X` of dimension (N x J), where:
%     - N = Number of observations (rows)
%     - J = Number of manifest variables / items (columns)
%
%   Example MAT-File Structure:
%     IpiData.mat  --> contains variable 'X' [N x J]
%
% PARAMETER SETTINGS (Configurable in Section 1):
%   1. dataset_name  : String name of the MAT-file or script (e.g., 'IpiData').
%   2. Q             : Target factor dimension / number of factors to extract.
%   3. proc_opts.data_type : Variable classification engine mode:
%                      - 0 : Force Pure Continuous (Pearson Correlation Matrix)
%                      - 1 : Force Pure Ordinal (Polychoric via R-Bridge)
%                      - 2 : Automated 3-Check Voting Detection Engine (Default)
%                      - Vector [1 x J] : Custom binary mask (0=Continuous, 1=Ordinal)
%
% OUTPUT:
%   - Terminal execution log with data audit & classification results.
%   - `summary_table`: Formatted MATLAB table comparing factor rotations across
%     Index, ECV, Generalized Salience (GS_w), and Orthogonality metrics.
% =========================================================================

clear; clc; close all;
rng(123);

% -------------------------------------------------------------------------
%% DATASET CONFIGURATION & PARAMETER SETTINGS
% -------------------------------------------------------------------------
dataset_name = 'WineData'; 
fprintf('Loading dataset: %s...\n', dataset_name);

% Load dataset MAT-file (Must populate workspace variable X [N x J])
eval(dataset_name); 

% Validate input dataset existence and structure
if ~exist('X', 'var')
    error('MasterDriver:MissingData', ...
        'Dataset file "%s" did not load a matrix named "X". Verify file structure.', dataset_name);
end

Q = 3; % Target factor dimension

% Pre-processing options (0: Force Continuous, 1: Force Ordinal, 2: Auto, Vector: Mask)
proc_opts = struct();
proc_opts.data_type = 0; 

% -------------------------------------------------------------------------
%% STEP 1: PRE-PROCESSING & FACTOR EXTRACTION
% -------------------------------------------------------------------------
[Lambda_unrot, R_smooth, Z, data_info] = processData(X, Q, proc_opts);

% -------------------------------------------------------------------------
%% STEP 2: EXECUTE MULTI-ROTATION EVALUATION VIA MANYROT
% -------------------------------------------------------------------------
fprintf('--- Step 2: Executing Factor Rotations via manyrot ---\n');

% Run benchmark rotations and prepend unrotated baseline directly
rotations_struct = manyrot(Lambda_unrot, 'all');
rotations_struct.UNROTATED.Lambda = Lambda_unrot;
rotations_struct.UNROTATED.T      = eye(Q);

% -------------------------------------------------------------------------
%% STEP 3: EVALUATE INTERPRETABILITY PROFILE VIA INTERP
% -------------------------------------------------------------------------
fprintf('\n--- Step 3: Evaluating Multi-Metric Interpretability ---\n');

opts_base = struct();
opts_base.p1      = 2;
opts_base.p2      = 4;
opts_base.weights = 'orthogonalized';
opts_base.Method  = data_info.ChosenMethod;

method_names = fieldnames(rotations_struct);
num_methods  = length(method_names);

% Pre-allocate diagnostic arrays
methods_col       = method_names;
index_col         = zeros(num_methods, 1);
ecv_col           = zeros(num_methods, 1);
gsw_col           = zeros(num_methods, 1);
orthogonality_col = zeros(num_methods, 1);
all_reports       = struct();

for i = 1:num_methods
    method_key = method_names{i};
    
    % Extract pre-computed rotated loadings and transformation matrix
    Lambda_rot = rotations_struct.(method_key).Lambda;
    T_method   = rotations_struct.(method_key).T;
    
    % Compute factor correlation matrix Phi
    Phi_method = inv(T_method' * T_method);
    
    opts_eval = opts_base;
    opts_eval.Rotation = method_key;
    
    report = interp(Lambda_rot, Phi_method, opts_eval);
    all_reports.(method_key) = report;
    
    index_col(i)         = report.CoreMetrics.Index;
    ecv_col(i)           = report.CoreMetrics.ECV;
    gsw_col(i)           = report.CoreMetrics.GS_w;
    orthogonality_col(i) = report.OrthogonalityCheck;
end

% Display final formatted results table
summary_table = table(methods_col, index_col, ecv_col, gsw_col, orthogonality_col, ...
    'VariableNames', {'Method', 'Index', 'ECV', 'GS_w', 'Orthogonality'});

% Move UNROTATED (the last row) to the top of the table
summary_table = [summary_table(end, :); summary_table(1:end-1, :)];

fprintf('\n=======================================================================\n');
fprintf('        FINAL ROTATION COMPARISON TABLE (%s)\n', dataset_name);
fprintf('=======================================================================\n');
disp(summary_table);