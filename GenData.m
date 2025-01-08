clc; clear; close all;

% Set random seed for reproducibility
rng(0);

%% (1) 사용자 지정 색상 팔레트 (총 10가지)
% 1:  [0.00000  0.45098  0.76078] (Blue)
% 2:  [0.93725  0.75294  0.00000]
% 3:  [0.80392  0.32549  0.29803] (Red)
% 4:  [0.12549  0.52157  0.30588]
% 5:  [0.57255  0.36863  0.62353]
% 6:  [0.88235  0.52941  0.15294]
% 7:  [0.30196  0.73333  0.83529]
% 8:  [0.93333  0.29803  0.59216]
% 9:  [0.49412  0.38039  0.28235]
% 10: [0.45490  0.46275  0.47059]
p_colors = [
    0.00000, 0.45098, 0.76078;  % #1 (Blue)
    0.93725, 0.75294, 0.00000;  % #2
    0.80392, 0.32549, 0.29803;  % #3 (Red)
    0.12549, 0.52157, 0.30588;  % #4
    0.57255, 0.36863, 0.62353;  % #5
    0.88235, 0.52941, 0.15294;  % #6
    0.30196, 0.73333, 0.83529;  % #7
    0.93333, 0.29803, 0.59216;  % #8
    0.49412, 0.38039, 0.28235;  % #9
    0.45490, 0.46275, 0.47059   % #10
];

%% (A) 전류/전압에 대한 고정 색상(Blue/Red) 지정
color_current = p_colors(1,:);  % Blue (#1)
color_voltage = p_colors(3,:);  % Red  (#3)

%% (2) Path
save_path = 'G:\공유 드라이브\Battery Software Lab\Projects\DRT\SD\';

%% (3) Parameters
num_scenarios = 10;  % Number of scenarios
num_waves = 3;       % Number of sine waves per scenario
t = linspace(0, 1000, 10001)';  % Time vector (0~1000 seconds, 10000 sample points)
dt = t(2) - t(1);
n = 201; % Number of RC elements

%% (4) Setting T (Period)
T_min = 15;           % Minimum period (seconds)
T_max = 250;          % Maximum period (seconds)

%% (5) Current Calculation
A = zeros(num_scenarios, num_waves);        
Tmat = zeros(num_scenarios, num_waves);     
ik_scenarios = zeros(num_scenarios, length(t)); 
R0 = 0.1;
OCV = 0;

% Generate random amplitudes and periods, and create current scenarios
for s = 1:num_scenarios
    % Generate amplitudes (normalized to sum to 3)
    temp_A = rand(1, num_waves);
    A(s, :) = 3 * temp_A / sum(temp_A);
    
    % Generate periods (random between T_min and T_max)
    Tmat(s, :) = T_min + (T_max - T_min) * rand(1, num_waves);
    
    % Create current scenario (sum of three sine waves)
    ik_scenarios(s, :) = A(s,1)*sin(2*pi*t / Tmat(s,1)) + ...
                         A(s,2)*sin(2*pi*t / Tmat(s,2)) + ...
                         A(s,3)*sin(2*pi*t / Tmat(s,3));
end

%% (6) True DRT Parameter Setting
% Unimodal
mu_theta_uni = log(10);
sigma_theta_uni = 1.53;
theta_min_uni = log(0.1);
theta_max_uni = log(1000);
theta_discrete_uni = linspace(theta_min_uni, theta_max_uni, n);
tau_discrete_uni = exp(theta_discrete_uni);
delta_theta_uni = theta_discrete_uni(2) - theta_discrete_uni(1);

gamma_discrete_true_unimodal = (1/(sigma_theta_uni * sqrt(2*pi))) ...
    * exp(- (theta_discrete_uni - mu_theta_uni).^2 / (2 * sigma_theta_uni^2));
gamma_discrete_true_unimodal = gamma_discrete_true_unimodal / max(gamma_discrete_true_unimodal);

% Bimodal
mu_theta1 = log(5);   sigma_theta1 = 1;
mu_theta2 = log(90);  sigma_theta2 = 0.7;

theta_min_bi = log(0.1);
theta_max_bi = log(1000);
theta_discrete_bi = linspace(theta_min_bi, theta_max_bi, n);
tau_discrete_bi = exp(theta_discrete_bi);
delta_theta_bi = theta_discrete_bi(2) - theta_discrete_bi(1);

gamma1 = (1 / (sigma_theta1 * sqrt(2 * pi))) ...
         * exp(- (theta_discrete_bi - mu_theta1).^2 / (2 * sigma_theta1^2));
gamma2 = (1 / (sigma_theta2 * sqrt(2 * pi))) ...
         * exp(- (theta_discrete_bi - mu_theta2).^2 / (2 * sigma_theta2^2));

gamma_discrete_true_bimodal = gamma1 + gamma2;
gamma_discrete_true_bimodal = gamma_discrete_true_bimodal / max(gamma_discrete_true_bimodal);

%% (7) Initialize Structs
AS1_1per = struct('SN', {}, 'V', {}, 'I', {}, 't', {});
AS1_2per = struct('SN', {}, 'V', {}, 'I', {}, 't', {});
AS2_1per = struct('SN', {}, 'V', {}, 'I', {}, 't', {});
AS2_2per = struct('SN', {}, 'V', {}, 'I', {}, 't', {});

%% (8) Voltage Calculation (Unimodal)
gamma_discrete_true = gamma_discrete_true_unimodal;
tau_discrete_current = tau_discrete_uni;
delta_theta_current = delta_theta_uni;

for s = 1:num_scenarios
    fprintf('Processing Unimodal Scenario %d/%d...\n', s, num_scenarios);
    
    ik = ik_scenarios(s, :)';  % Current data (column vector)
    
    V_est = zeros(length(t), 1);   
    V_RC = zeros(n, length(t));
    
    for k_idx = 1:length(t)
        if k_idx == 1
            for i = 1:n
                V_RC(i, k_idx) = gamma_discrete_true(i)*delta_theta_current*ik(k_idx) ...
                                 * (1 - exp(-dt / tau_discrete_current(i)));
            end
        else
            for i = 1:n
                V_RC(i, k_idx) = V_RC(i, k_idx-1)*exp(-dt / tau_discrete_current(i)) + ...
                                 gamma_discrete_true(i)*delta_theta_current*ik(k_idx) ...
                                 * (1 - exp(-dt / tau_discrete_current(i)));
            end
        end
        V_est(k_idx) = OCV + R0*ik(k_idx) + sum(V_RC(:, k_idx));
    end
    
    % Add noise (1%)
    noise_level = 0.01;
    V_sd_1per = V_est + noise_level .* V_est .* randn(size(V_est));

    % Add noise (2%)
    noise_level = 0.02;
    V_sd_2per = V_est + noise_level .* V_est .* randn(size(V_est));

    AS1_1per(s).SN = s;
    AS1_1per(s).V = V_sd_1per;
    AS1_1per(s).I = ik;
    AS1_1per(s).t = t;

    AS1_2per(s).SN = s;
    AS1_2per(s).V = V_sd_2per;
    AS1_2per(s).I = ik;
    AS1_2per(s).t = t;
end

%% (9) Voltage Calculation (Bimodal)
gamma_discrete_true = gamma_discrete_true_bimodal;
tau_discrete_current = tau_discrete_bi;
delta_theta_current = delta_theta_bi;

for s = 1:num_scenarios
    fprintf('Processing Bimodal Scenario %d/%d...\n', s, num_scenarios);
    
    ik = ik_scenarios(s, :)';  % Current data
    
    V_est = zeros(length(t), 1);
    V_RC = zeros(n, length(t));
    
    for k_idx = 1:length(t)
        if k_idx == 1
            for i = 1:n
                V_RC(i, k_idx) = gamma_discrete_true(i)*delta_theta_current*ik(k_idx) ...
                                 * (1 - exp(-dt / tau_discrete_current(i)));
            end
        else
            for i = 1:n
                V_RC(i, k_idx) = V_RC(i, k_idx-1)*exp(-dt / tau_discrete_current(i)) + ...
                                 gamma_discrete_true(i)*delta_theta_current*ik(k_idx) ...
                                 * (1 - exp(-dt / tau_discrete_current(i)));
            end
        end
        V_est(k_idx) = OCV + R0*ik(k_idx) + sum(V_RC(:, k_idx));
    end
    
    % Add noise (1%)
    noise_level = 0.01;
    V_sd_1per = V_est + noise_level .* V_est .* randn(size(V_est));
    
    % Add noise (2%)
    noise_level = 0.02;
    V_sd_2per = V_est + noise_level .* V_est .* randn(size(V_est));
    
    AS2_1per(s).SN = s;
    AS2_1per(s).V = V_sd_1per;
    AS2_1per(s).I = ik;
    AS2_1per(s).t = t;

    AS2_2per(s).SN = s;
    AS2_2per(s).V = V_sd_2per;
    AS2_2per(s).I = ik;
    AS2_2per(s).t = t;
end

%% (10) 시나리오별 Current & Voltage Plot
figure_names = {'AS1 1per', 'AS1 2per', ...
                'AS2 1per', 'AS2 2per'};
struct_cases = {AS1_1per, AS1_2per, AS2_1per, AS2_2per};

for case_idx = 1:length(struct_cases)
    current_case = struct_cases{case_idx};
    figure('Name', figure_names{case_idx}, 'NumberTitle', 'off');
    
    % Create a grid of subplots (5 rows x 2 columns) for 10 scenarios
    num_rows = 5;
    num_cols = 2;
    
    for s = 1:num_scenarios
        subplot(num_rows, num_cols, s);
        
        yyaxis left
        % 전류: 파란색 (Blue)
        plot(current_case(s).t, current_case(s).I, '-', ...
            'LineWidth', 3, 'Color', color_current);
        ylabel('Current (A)', 'FontSize', 12);
        hold on;
        
        yyaxis right
        % 전압: 빨간색 (Red)
        plot(current_case(s).t, current_case(s).V, '-', ...
            'LineWidth', 3, 'Color', color_voltage);
        ylabel('Voltage (V)', 'FontSize', 12);
        
        xlabel('Time (s)', 'FontSize', 12);
        title(['Scenario ', num2str(current_case(s).SN)], 'FontSize', 14);
        legend('Current', 'Voltage', 'Location', 'best');
        hold off;
    end
    
    sgtitle(figure_names{case_idx}, 'FontSize', 16);
end

%% (11) Gamma vs Theta Plot
Gamma_unimodal.theta = theta_discrete_uni';
Gamma_unimodal.gamma = gamma_discrete_true_unimodal';

Gamma_bimodal.theta = theta_discrete_bi';
Gamma_bimodal.gamma = gamma_discrete_true_bimodal';

% Unimodal
figure('Name','Gamma - Unimodal','NumberTitle','off');
plot(Gamma_unimodal.theta, Gamma_unimodal.gamma, 'k-', 'LineWidth', 3);
xlabel('\theta'); ylabel('\gamma');
title('Unimodal \gamma vs \theta');

% Bimodal
figure('Name','Gamma - Bimodal','NumberTitle','off');
plot(Gamma_bimodal.theta, Gamma_bimodal.gamma, 'k-', 'LineWidth', 3);
xlabel('\theta'); ylabel('\gamma');
title('Bimodal \gamma vs \theta');

%% (12) Plot Current Scenarios Only (Before Applying Resistance Distribution)
figure('Name','Current scenarios only','NumberTitle','off');
num_rows = 5;
num_cols = 2;

for s = 1:num_scenarios
    subplot(num_rows, num_cols, s);
    % 여기서는 전류만 표시 -> 파란색
    plot(t, ik_scenarios(s,:), '-', 'LineWidth', 3, 'Color', color_current);
    xlabel('Time (s)','FontSize',12);
    ylabel('Current (A)','FontSize',12);
    title(['Scenario ', num2str(s)], 'FontSize',14);
end
sgtitle('Current Scenarios','FontSize',16);

%% (13) 예: AS2_1per에서 시나리오 6,7,8,9만 2x2 subplot
scenarios_to_plot = [6, 7, 8, 9];
figure('Name','AS2_1per - Scenarios 6,7,8,9','NumberTitle','off');

for idx = 1:length(scenarios_to_plot)
    s = scenarios_to_plot(idx);
    subplot(2, 2, idx);
    
    yyaxis left
    % 전류: 파란색
    plot(AS2_1per(s).t, AS2_1per(s).I, '-', ...
        'LineWidth', 3, 'Color', color_current);
    ylabel('Current (A)', 'FontSize', 12);
    hold on;
    
    yyaxis right
    % 전압: 빨간색
    plot(AS2_1per(s).t, AS2_1per(s).V, '-', ...
        'LineWidth', 3, 'Color', color_voltage);
    ylabel('Voltage (V)', 'FontSize', 12);
    
    xlabel('Time (s)', 'FontSize', 12);
    title(['Scenario ', num2str(AS2_1per(s).SN)], 'FontSize', 14);
    legend('Current', 'Voltage', 'Location', 'best');
    hold off;
end
sgtitle('AS2\_1per: Scenarios 6,7,8,9', 'FontSize', 16);

%% (14) Save
save(fullfile(save_path, 'AS1_1per.mat'), 'AS1_1per');
save(fullfile(save_path, 'AS1_2per.mat'), 'AS1_2per');
save(fullfile(save_path, 'AS2_1per.mat'), 'AS2_1per');
save(fullfile(save_path, 'AS2_2per.mat'), 'AS2_2per');

save(fullfile(save_path, 'Gamma_unimodal.mat'), 'Gamma_unimodal');
save(fullfile(save_path, 'Gamma_bimodal.mat'),  'Gamma_bimodal');


