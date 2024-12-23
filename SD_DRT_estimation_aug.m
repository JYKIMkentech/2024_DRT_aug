function [gamma_est, R0_est, V_est, theta_discrete, tau_discrete, W_aug, y] = SD_DRT_estimation_aug(t, ik, V_sd, lambda_hat, n, dt, dur)
% DRT_estimation_aug estimates the gamma function and R0 using DRT with augmented variable.
%
% Inputs:
%   t           - Time vector (Nx1)
%   ik          - Current vector (Nx1)
%   V_sd        - Measured voltage vector (Nx1), where OCV=0 가정
%   lambda_hat  - Regularization parameter
%   n           - Number of RC elements
%   dt          - Sampling time vector (Nx1) 또는 scalar
%   dur         - Duration (tau_max) (현재 코드에서는 안 써도 무방, 또는 tau_max설정에 사용)
%
% Outputs:
%   gamma_est       - Estimated gamma vector (nx1)
%   R0_est          - Estimated R0 (scalar)
%   V_est           - Estimated voltage vector (Nx1)
%   theta_discrete  - Discrete theta values (nx1)
%   tau_discrete    - Discrete tau values (nx1)
%   W_aug           - Augmented W matrix (Nx(n+1)) = [W, ik]
%   y               - Adjusted y = V_sd (OCV=0)

    % 1) dt가 scalar라면 t와 길이 맞추기
    if isscalar(dt)
        dt = repmat(dt, length(t), 1);
    elseif length(dt) ~= length(t)
        error('Length of dt must be equal to length of t if dt is a vector.');
    end

    % 2) theta, tau 설정
    tau_min = 0.1;   % 실제 문제 상황에 맞게 조정
    tau_max = 1000;  
    theta_min = log(tau_min);
    theta_max = log(tau_max);
    theta_discrete = linspace(theta_min, theta_max, n)';
    delta_theta    = theta_discrete(2) - theta_discrete(1);
    tau_discrete   = exp(theta_discrete);

    % 3) W (RC 적분항)
    W = zeros(length(t), n);
    for k_idx = 1:length(t)
        if k_idx == 1
            for i = 1:n
                W(k_idx, i) = ik(k_idx) * (1 - exp(-dt(k_idx)/tau_discrete(i))) * delta_theta;
            end
        else
            for i = 1:n
                W(k_idx, i) = W(k_idx-1, i)*exp(-dt(k_idx)/tau_discrete(i)) + ...
                              ik(k_idx)*(1 - exp(-dt(k_idx)/tau_discrete(i)))*delta_theta;
            end
        end
    end

    % 4) Augmented W
    %    마지막 열에 i(k)를 붙여주고, 이는 R0와 곱해질 항이 됨
    W_aug = [W, ik(:)];  

    % 5) y = V_sd (OCV=0 가정)
    y = V_sd(:);

    % 6) 정규화(regularization) 위한 차분 행렬 (gamma에만 1차 차분, R0에는 없음)
    L = zeros(n-1, n);
    for i = 1:n-1
        L(i, i)   = -1;
        L(i, i+1) = 1;
    end
    L_aug = [L, zeros(n-1, 1)];  % R0에 대해서는 0

    % 7) QP: min (V_est - V_sd)^2 + lambda * || D(gamma) ||^2
    H = 2 * (W_aug'*W_aug + lambda_hat*(L_aug'*L_aug));
    f = -2 * W_aug' * y;

    % 8) gamma >=0, R0 >=0 라고 가정한다면 (물리적으로 음수 저항 불가능)
    A_ineq = -eye(n+1);
    b_ineq = zeros(n+1, 1);

    % 9) QP 문제 풀이
    options = optimoptions('quadprog','Display','off');
    params = quadprog(H, f, A_ineq, b_ineq, [], [], [], [], [], options);

    % 10) 결과 분리
    gamma_est = params(1:end-1);
    R0_est    = params(end);

    % 11) 추정 전압
    V_est = W_aug * params;  
end
