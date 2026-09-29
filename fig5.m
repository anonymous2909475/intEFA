% Prerequisites: Unrotated loading matrix 'Lambda_unrot' evaluated on 
% the PsyData dataset (Q = 4, raw covariance scale: proc_opts.data_type = 0).

% 1. Perform oblique factor rotation (Oblimin criterion, gamma = 0)
[Lambda_info, ~] = rotate(Lambda_unrot, 'infomax', 20);

% 2. Evaluate row-wise simple structure compliance (Figure 5)
rowcompliance(Lambda_info);

