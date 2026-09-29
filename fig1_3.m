%
% Numerically verifies Propositions 3.1-3.2 of Section 2.1 ("Asymptotic
% behaviour and choice of p1,p2") and produces three figures:
%
%   Figure 1 - Proposition 3.1 (L_p1 / L_infty limit):
%              Left panel: gs(v, p1, p2) vs p2 for fixed p1 values, showing
%              convergence to the theoretical limit gs_infty(v, p1) (dashed line).
%              Right panel: log-log plot of |gs(p2) - gs_infty| vs p2,
%              confirming the O(1/p2) polynomial convergence rate.
%              For each p1, only p2 > p1 is evaluated/plotted, since gs(.,p1,p2)
%              in Equation (7) is defined solely on 1 <= p1 < p2 < infty; points
%              with p2 <= p1 fall outside that domain and are excluded.
%
%   Figure 2 - Proposition 3.2 (Saturation trap):
%              1 - gs(v, p1, p1 + delta) vs p1 on a semi-log y-axis across fixed
%              gaps delta, confirming the exponential-in-p1 collapse to gs = 1
%              (a straight line on the log scale with slope determined by log(r2)).
%
%   Figure 3 - Global heatmap of gs(v, p1, p2):
%              2D intensity map over a (p1, p2) grid for a fixed test vector,
%              highlighting the "safe" evaluation plateau as p2 grows for
%              fixed p1 (Prop. 3.1) versus the saturation band along p1 ~ p2 (Prop. 3.2).
%
% Requirements:
%   Base MATLAB only (no extra toolboxes required).
%
% Usage:
%   asymptotics_plots            % Run with default benchmark vector (d = 10, r2 = 0.6)
%   asymptotics_plots(v)         % Run with a custom d-dimensional loading vector
%
function asymptotics_plots(v)
    if nargin < 1
        % Demo vector: one dominant entry, one "runner-up" cross-loading at
        % r2 = 0.6, plus small noise entries. d = 10, mimicking the J = 10
        % variables used in the paper's simulation (Section 3).
        % The RNG is seeded so the demo vector - and hence the figures - is
        % identical across runs, which is required for a reproducible
        % published result.
        rng(1);
        d = 10;
        v = zeros(1, d);
        v(1) = 1.00;                 % dominant loading
        v(2) = 0.60;                 % runner-up -> r2 = 0.6
        v(3:d) = 0.05 * randn(1, d-2); % small cross-loading noise
    end
    
    v = abs(v(:))';
    d = numel(v);
    [vs, ~] = sort(v, 'descend');
    vmax = vs(1);
    r2   = vs(2) / vmax;
    
    fprintf('Running Asymptotics Validation Script...\n');
    fprintf('Vector dimensions: d = %d | v_max = %.3f | Runner-up ratio (r2) = %.3f\n\n', ...
        d, vmax, r2);

    %% ---------- Figure 1: Proposition 3.1 (L_p1 / L_infty limit) ----------
    p1_list = [1, 2, 3, 4];
    p2_grid = 2:2:80;
    
    figure('Name', 'Figure 1: Proposition 3.1 - Convergence to L_p1 / L_infty Limit');
    
    tiledlayout_ok = exist('tiledlayout', 'file') == 2;
    if tiledlayout_ok
        tl = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    end

    % Left panel: gs(v, p1, p2) vs p2 (Linear Scale)
    if tiledlayout_ok, nexttile; else, subplot(1, 2, 1); end
    hold on;
    cmap = lines(numel(p1_list));
    
    for i = 1:numel(p1_list)
        p1 = p1_list(i);
        % gs(.,p1,p2) is only defined for p2 > p1 (Eq. 7); evaluating it at
        % p2 <= p1 either divides 0/0 (p2 == p1) or applies the formula
        % outside its valid domain (p2 < p1), producing spurious values.
        p2_valid = p2_grid(p2_grid > p1);
        gs_vals  = arrayfun(@(p2) gs_measure(v, p1, p2), p2_valid);
        gs_inf   = gs_infty(v, p1);
        
        plot(p2_valid, gs_vals, '-', 'Color', cmap(i,:), 'LineWidth', 1.8, ...
            'DisplayName', sprintf('p_1 = %d', p1));
        yline(gs_inf, '--', 'Color', cmap(i,:), 'LineWidth', 1.2, ...
            'HandleVisibility', 'off');
    end
    
    xlabel('p_2', 'FontSize', 14);
    ylabel('gs(v, p_1, p_2)', 'FontSize', 14);
    title('gs(v, p_1, p_2) \rightarrow gs_\infty(v, p_1) as p_2 \rightarrow \infty', 'FontSize', 12);
    legend('Location', 'southeast', 'FontSize', 12);
    grid on;
    set(gca, 'FontSize', 12);

    % Right panel: Log-Log plot of |gs(p2) - gs_infty| vs p2
    if tiledlayout_ok, nexttile; else, subplot(1, 2, 2); end
    hold on;
    
    for i = 1:numel(p1_list)
        p1 = p1_list(i);
        p2_valid = p2_grid(p2_grid > p1);   % same domain restriction as above
        gs_inf   = gs_infty(v, p1);
        gs_vals  = arrayfun(@(p2) gs_measure(v, p1, p2), p2_valid);
        err      = abs(gs_vals - gs_inf);
        err(err == 0) = eps;
        
        loglog(p2_valid, err, '-o', 'Color', cmap(i,:), 'MarkerSize', 4, ...
            'LineWidth', 1.2, 'DisplayName', sprintf('p_1 = %d', p1));
    end
    
    % Reference O(1/p2) slope, anchored to the p1 = 1 curve (whose domain
    % covers the full p2_grid, so its first valid point is p2_grid(1))
    p2_ref1 = p2_grid(p2_grid > 1);
    err1 = abs(gs_measure(v, 1, p2_ref1(1)) - gs_infty(v, 1));
    ref  = err1 * p2_ref1(1) ./ p2_grid;
    loglog(p2_grid, ref, 'k--', 'LineWidth', 1.5, 'DisplayName', 'O(1/p_2) reference');
    
    xlabel('p_2 (log scale)', 'FontSize', 14);
    ylabel('|gs(v, p_1, p_2) - gs_\infty(v, p_1)| (log scale)', 'FontSize', 14);
    title('Rate of Convergence: Polynomial O(1/p_2)', 'FontSize', 12);
    legend('Location', 'southwest', 'FontSize', 12);
    grid on;
    set(gca, 'XScale', 'log', 'YScale', 'log', 'FontSize', 12);

    %% ---------- Figure 2: Proposition 3.2 (Saturation trap) ----------
    delta_list = [1, 2, 3, 5];
    p1_grid    = 1:20;
    
    figure('Name', 'Figure 2: Proposition 3.2 - Saturation Trap as p1, p2 -> infinity Jointly');
    hold on;
    cmap2 = lines(numel(delta_list));
    
    for i = 1:numel(delta_list)
        delta = delta_list(i);
        one_minus_gs = arrayfun(@(p1) 1 - gs_measure(v, p1, p1 + delta), p1_grid);
        one_minus_gs(one_minus_gs <= 0) = eps;
        
        semilogy(p1_grid, one_minus_gs, '-o', 'Color', cmap2(i,:), ...
            'MarkerSize', 5, 'LineWidth', 1.5, ...
            'DisplayName', sprintf('\\delta = %d', delta));
    end
    
    xlabel('p_1 (linear scale)', 'FontSize', 14);
    ylabel('1 - gs(v, p_1, p_1 + \delta)  (log scale)', 'FontSize', 14);
    title(sprintf('Exponential Collapse to gs = 1 (r_2 = %.2f)', r2), 'FontSize', 12);
    legend('Location', 'northeast', 'FontSize', 12);
    grid on;
    set(gca, 'YScale', 'log', 'FontSize', 12);

    %% ---------- Figure 3: Heatmap of gs over (p1, p2) Grid ----------
    p1_range = 1:15;
    p2_range = 1:20;
    GS       = nan(numel(p1_range), numel(p2_range));
    
    for a = 1:numel(p1_range)
        for b = 1:numel(p2_range)
            if p2_range(b) > p1_range(a)
                GS(a,b) = gs_measure(v, p1_range(a), p2_range(b));
            end
        end
    end
    
    figure('Name', 'Figure 3: Global gs(v, p1, p2) Heatmap');
    imagesc(p2_range, p1_range, GS, 'AlphaData', ~isnan(GS));
    set(gca, 'YDir', 'normal', 'Color', [0.9 0.9 0.9], 'FontSize', 12);
    cb = colorbar;
    cb.Label.String = 'gs(v, p_1, p_2)';
    cb.Label.FontSize = 12;
    colormap(parula);
    
    xlabel('p_2', 'FontSize', 14);
    ylabel('p_1', 'FontSize', 14);
    title('gs(v, p_1, p_2) Heatmap: Safe Plateau (Prop. 3.1) vs Saturation Band (Prop. 3.2)', 'FontSize', 12);
    fprintf('Done. Successfully generated Figures 1, 2, and 3.\n');
end

%% ==================== LOCAL FUNCTIONS ====================
function s = gs_measure(v, p1, p2)
    v = abs(v(:));
    d = numel(v);
    
    if d <= 1
        s = 1;
        return;
    end
    if all(v == 0)
        s = 0;
        return;
    end
    
    nv1 = norm(v, p1);
    nv2 = norm(v, p2);
    c   = d^(1/p1 - 1/p2);
    ratio = nv1 / nv2;
    
    s = (c - ratio) / (c - 1);
    s = min(max(s, 0), 1);
end

function s = gs_infty(v, p1)
    v = abs(v(:));
    d = numel(v);
    vmax = max(v);
    
    if vmax == 0
        s = 0;
        return;
    end
    
    nv1   = norm(v, p1);
    c     = d^(1/p1);
    ratio = nv1 / vmax;
    
    s = (c - ratio) / (c - 1);
    s = min(max(s, 0), 1);
end