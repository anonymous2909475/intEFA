%% ========================================================================
%  FIGURE 6: FACTOR RETENTION & INTERPRETABILITY DIAGNOSTICS (fig6.m)
%  ========================================================================
%  Description:
%    Executes a dual-criterion dimensionality evaluation to determine the 
%    optimal number of retained latent factors (Q*). The script integrates:
%      1. Empirical Scree Analysis vs. Parallel Analysis (PA) null model 
%         simulations (50 iterations) and Kaiser's Criterion (\lambda = 1).
%      2. Multi-Metric Interpretability Profiling across Q \in [1, Q_max]
%         evaluating Explained Common Variance (ECV), Generalized Gini-
%         Simpson Sparsity (GS_w), and a unified Composite Index.
%
%  Data Inputs:
%    - X : Raw observation matrix (N x J) or Correlation Matrix (J x J)
%
%  Outputs:
%    - (Figure) Scree Plot & Interpretability Profile dual panel
%    - ProfileTable : Summary table of interpretability metrics across Q
%  ========================================================================
clearvars -except X; clc; close all;

%% 1. CONFIGURATION & DATA INITIALIZATION
WineData; % Load raw measurement dataset
interp_opts = struct('weights', 'orthogonalized');

% Determine input matrix type (Raw Data vs. Pre-computed Correlation Matrix)
if ismatrix(X) && size(X,1) == size(X,2) && isequal(X, X') && all(abs(diag(X) - 1) < 1e-5)
    R = X;
    N_obs = 500; % Fallback effective sample size for PA
else
    X_std = zscore(X); % Standardize data to correlation scale
    [N, J] = size(X_std);
    R = corr(X_std, 'Rows', 'pairwise');
    N_obs = N;
end

[~, J] = size(R);
Q_max = min(13, J - 1); % Retain up to J - 1 factors cleanly

%% 2. EIGENDECOMPOSITION & PARALLEL ANALYSIS (PA)
[V, D] = eig(R);
[d_sorted, idx] = sort(diag(D), 'descend');
d_sorted = max(d_sorted, 0); % Truncate machine-precision negative eigenvalues
eigenvals_data = d_sorted;

% Monte Carlo Parallel Analysis Simulation (Null Synthetic Spectrum)
n_sims = 50;
sim_eigs = zeros(J, n_sims);
rng(42, 'twister'); % Replicability seed
for s = 1:n_sims
    X_null = randn(N_obs, J);
    R_null = corr(X_null);
    sim_eigs(:, s) = sort(eig(R_null), 'descend');
end
eigenvals_pa = mean(sim_eigs, 2);

%% 3. FACTOR EXTRACTION & INTERPRETABILITY EVALUATION (via PAF)
ecv_vec   = zeros(Q_max, 1);
gsw_vec   = zeros(Q_max, 1);
index_vec = zeros(Q_max, 1);

for q = 1:Q_max
    Phi_q = eye(q); % Default orthogonal manifold assumption
    
    try
        % Extract true EFA loadings on correlation scale via PAF
        L_q = paf(R, q);
        
        rep = interp(L_q, Phi_q, interp_opts);
        ecv_vec(q)   = rep.CoreMetrics.ECV;
        gsw_vec(q)   = rep.CoreMetrics.GS_w;
        index_vec(q) = rep.CoreMetrics.Index;
    catch ME
        warning('PAF extraction/interp failed at q = %d: %s', q, ME.message);
        ecv_vec(q)   = NaN;
        gsw_vec(q)   = NaN;
        index_vec(q) = NaN;
    end
end

%% 4. NUMERICAL PROFILE REPORTING
ProfileTable = table((1:Q_max)', ecv_vec, gsw_vec, index_vec, ...
    'VariableNames', {'Q', 'ECV', 'GS_w', 'Interpretability'});
fprintf('\n=======================================================================\n');
fprintf('     INTERPRETABILITY & RETENTION PROFILE ACROSS Q\n');
fprintf('=======================================================================\n');
disp(ProfileTable);

%% 5. EXACT SPECIFICATION GRAPHICAL RENDER
% Color Palette Definitions
c_blue   = [0.12, 0.38, 0.68];
c_red    = [0.80, 0.25, 0.20];
c_green  = [0.18, 0.55, 0.28];
c_orange = [0.85, 0.50, 0.15];
c_purple = [0.45, 0.18, 0.62];
c_band   = [0.85, 0.90, 0.98]; % Soft blue highlight band

% Render Canvas
hFig = figure('Name', 'Factor Retention and Interpretability Analysis', ...
              'Color', [1 1 1], ...
              'Units', 'pixels', ...
              'Position', [100, 100, 1000, 450]);

x_band = [2 4 4 2]; % Candidate zone highlight (Q = 2..4)

% -------------------------------------------------------------------------
% PANEL A: SCREE PLOT & PARALLEL ANALYSIS
% -------------------------------------------------------------------------
ax1 = subplot(1, 2, 1);
hold(ax1, 'on'); grid(ax1, 'on'); box(ax1, 'on');
y_lim1 = [0, max(eigenvals_data) * 1.05];

% Candidate Domain Highlight Patch
p1 = patch(ax1, x_band, [y_lim1(1) y_lim1(1) y_lim1(2) y_lim1(2)], c_band, ...
    'EdgeColor', 'none', 'FaceAlpha', 0.65);

% Scree Curves
h_data = plot(ax1, 1:J, eigenvals_data, '-o', 'Color', c_blue, ...
    'LineWidth', 2.0, 'MarkerSize', 7, 'MarkerFaceColor', c_blue);

h_pa   = plot(ax1, 1:J, eigenvals_pa, '-.s', 'Color', c_red, ...
    'LineWidth', 1.5, 'MarkerSize', 6, 'MarkerFaceColor', c_red);

h_kaiser = line(ax1, [0.5, J + 0.5], [1 1], 'Color', [0.4 0.4 0.4], ...
    'LineStyle', ':', 'LineWidth', 1.2);

% Formatting
title(ax1, 'Scree Plot & Parallel Analysis', 'FontSize', 12, 'FontWeight', 'bold');
xlabel(ax1, 'Q', 'FontSize', 11);
ylabel(ax1, 'Eigenvalues', 'FontSize', 11);
xlim(ax1, [0.5, Q_max + 0.5]);
ylim(ax1, y_lim1);
set(ax1, 'FontName', 'Helvetica', 'FontSize', 10);

% Legend matching image exact labels and order
legend(ax1, [p1, h_data, h_pa, h_kaiser], ...
    {'Candidate Range (Q=2..4)', 'Data Eigenvalues', 'Parallel Analysis (Mean)', 'Kaiser Criterion (\lambda=1)'}, ...
    'Location', 'northeast', 'Box', 'on', 'FontSize', 9);

% -------------------------------------------------------------------------
% PANEL B: INTERPRETABILITY PROFILE
% -------------------------------------------------------------------------
ax2 = subplot(1, 2, 2);
hold(ax2, 'on'); grid(ax2, 'on'); box(ax2, 'on');

% Candidate Domain Highlight Patch
p2 = patch(ax2, x_band, [0 0 1.02 1.02], c_band, ...
    'EdgeColor', 'none', 'FaceAlpha', 0.65);

% Interpretability Metric Curves
h_ecv   = plot(ax2, 1:Q_max, ecv_vec, '-^', 'Color', c_green, 'LineWidth', 2.0, ...
    'MarkerSize', 7, 'MarkerFaceColor', c_green);

h_gsw   = plot(ax2, 1:Q_max, gsw_vec, '-s', 'Color', c_orange, 'LineWidth', 2.0, ...
    'MarkerSize', 6, 'MarkerFaceColor', c_orange);

h_interp= plot(ax2, 1:Q_max, index_vec, '-o', 'Color', c_purple, 'LineWidth', 2.2, ...
    'MarkerSize', 7, 'MarkerFaceColor', c_purple);

% Formatting
title(ax2, 'Interpretability Profile', 'FontSize', 12, 'FontWeight', 'bold');
xlabel(ax2, 'Q', 'FontSize', 11);
ylabel(ax2, 'Metric Score', 'FontSize', 11);
xlim(ax2, [0.5, Q_max + 0.5]);
ylim(ax2, [0, 1.02]);
set(ax2, 'FontName', 'Helvetica', 'FontSize', 10);

% Legend matching image exact labels and order
legend(ax2, [p2, h_ecv, h_gsw, h_interp], ...
    {'Candidate Range (Q=2..4)', 'ECV', 'GS_w', 'Interpretability'}, ...
    'Location', 'southeast', 'Box', 'on', 'FontSize', 9);

%% ========================================================================
%  HELPER FUNCTION: ITERATED PRINCIPAL AXIS FACTORING (PAF)
%  ========================================================================
function [L, h2] = paf(R, q, max_iter, tol)
    if nargin < 3, max_iter = 100; end
    if nargin < 4, tol = 1e-5; end

    J = size(R, 1);
    
    % Initial communality estimates: Squared Multiple Correlations (SMCs)
    if rank(R) == J
        h2 = 1 - 1 ./ diag(inv(R));
    else
        h2 = max(abs(R - eye(J)), [], 2);
    end

    R_adj = R;
    for iter = 1:max_iter
        h2_old = h2;
        
        % Update diagonal with communality estimates
        R_adj(1:J+1:end) = h2;
        
        % Eigendecomposition of reduced correlation matrix
        [V, D] = eig(R_adj);
        [d_sorted, idx] = sort(diag(D), 'descend');
        
        q_eigs = max(d_sorted(1:q), 0);
        V_q = V(:, idx(1:q));
        
        % Compute common factor loadings
        L = V_q * diag(sqrt(q_eigs));
        
        % Re-estimate communalities
        h2 = sum(L.^2, 2);
        
        % Convergence check
        if max(abs(h2 - h2_old)) < tol
            break;
        end
    end
end