function [gamma_lower, gamma_upper, gamma_resample_all] = ...
    bootstrap_uncertainty_aug(t, ik, V_sd, lambda, n, dt, dur, num_resamples)
% bootstrap_uncertainty_aug estimates the uncertainty of gamma (and R0) via bootstrap.
%
% Inputs:
%   t             - Time vector
%   ik            - Current vector
%   V_sd          - Measured voltage vector
%   lambda        - Regularization parameter
%   n             - Number of RC elements
%   dt            - Sampling time
%   dur           - Duration (tau_max)
%   num_resamples - Number of bootstrap resamples
%
% Outputs:
%   gamma_lower        - Lower bound (5th percentile) of gamma
%   gamma_upper        - Upper bound (95th percentile) of gamma
%   gamma_resample_all - All resampled gamma estimates (num_resamples x n)

    % (1) 원본 데이터로부터 한 번 gamma, R0를 추정(중앙값으로 삼아도 됨)
    [gamma_original, ~, ~, ~, ~, ~, ~] = ...
        DRT_estimation_aug(t, ik, V_sd, lambda, n, dt, dur);

    n_length = length(gamma_original);
    gamma_resample_all = zeros(num_resamples, n_length);

    % (2) bootstrap
    N = length(t);
    for b = 1:num_resamples
        % 복원추출
        resample_idx = randsample(N, N, true);

        % 중복/정렬 처리 (시간 순서가 중요하면 아래와 같이 처리)
        t_resampled     = t(resample_idx);
        ik_resampled    = ik(resample_idx);
        V_sd_resampled  = V_sd(resample_idx);

        [t_unique, unique_idx] = unique(t_resampled);
        ik_unique   = ik_resampled(unique_idx);
        V_unique    = V_sd_resampled(unique_idx);

        [t_sorted, sort_idx]   = sort(t_unique);
        ik_sorted = ik_unique(sort_idx);
        V_sorted  = V_unique(sort_idx);

        dt_resampled = [t_sorted(1); diff(t_sorted)];

        [gamma_b, ~, ~, ~, ~, ~, ~] = SD_DRT_estimation_aug(t_sorted, ik_sorted, V_sorted, lambda, n, dt_resampled, dur);

        gamma_resample_all(b, :) = gamma_b';
    end

    % (3) Percentile 계산
    gamma_resample_percentiles = prctile(gamma_resample_all, [5 95]);
    gamma_lower = gamma_resample_percentiles(1, :);
    gamma_upper = gamma_resample_percentiles(2, :);

    % (4) (참고) R0에 대한 bootstrap 필요 시, 추가로 R0_resample_all 배열을 만들어 동일하게 진행하면 됨.
end
