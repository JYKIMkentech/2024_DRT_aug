%% SD_Run_DRT_aug.m
clc; clear; close all;

%% (1) Description
% - 여러 시나리오(each AS_data)에 대해 DRT 추정 (gamma, R0)
% - Bootstrap으로 불확실성 평가 후 결과 플롯

%% (2) Graphic Parameters
axisFontSize   = 14;
titleFontSize  = 12;
legendFontSize = 12;
labelFontSize  = 12;

%% (3) Load Data
file_path = 'G:\공유 드라이브\Battery Software Lab\Projects\DRT\SD_lambda_aug\';
mat_files = dir(fullfile(file_path, '*.mat'));

for file = mat_files'
    load(fullfile(file_path, file.name));  % 예: AS1_1per_new, AS1_2per_new 등 로드
end

%% (4) Parameters
AS_structs = {AS1_1per_new, AS1_2per_new, AS2_1per_new, AS2_2per_new};
AS_names   = {'AS1_1per_new', 'AS1_2per_new', 'AS2_1per_new', 'AS2_2per_new'};
Gamma_structs = {Gamma_unimodal, Gamma_unimodal, Gamma_bimodal, Gamma_bimodal};

fprintf('Available datasets:\n');
for idx = 1:length(AS_names)
    fprintf('%d: %s\n', idx, AS_names{idx});
end
dataset_idx = input('Select a dataset to process (enter the number): ');
AS_data     = AS_structs{dataset_idx};
AS_name     = AS_names{dataset_idx};
Gamma_data  = Gamma_structs{dataset_idx};

% Type(실험 조건) 선택
types = unique({AS_data.type});
disp('Select a type:');
for i = 1:length(types)
    fprintf('%d. %s\n', i, types{i});
end
type_idx = input('Enter the type number: ');
selected_type = types{type_idx};

type_indices = find(strcmp({AS_data.type}, selected_type));
type_data    = AS_data(type_indices);

num_scenarios = length(type_data);
SN_list       = [type_data.SN];

fprintf('Selected dataset: %s\n', AS_name);
fprintf('Selected type: %s\n', selected_type);
fprintf('Scenario numbers: ');
disp(SN_list);

% Color for plotting
c_mat = lines(num_scenarios);

%% (5) True Gamma (for comparison)
% - (만약 ground truth가 있다고 가정하고, Gamma_data가 저장되어 있다고 했으므로)
gamma_discrete_true = Gamma_data.gamma'; 
theta_true          = Gamma_data.theta';  

%% (6) DRT + Bootstrap
% 결과 저장할 변수들
gamma_est_all                = cell(num_scenarios, 1);
R0_est_all                   = zeros(num_scenarios, 1);
theta_discrete_all           = cell(num_scenarios, 1);
gamma_lower_all              = cell(num_scenarios, 1);
gamma_upper_all              = cell(num_scenarios, 1);
gamma_resample_all_scenarios = cell(num_scenarios, 1);

num_resamples = 500;  % bootstrap 횟수

for s = 1:num_scenarios
    fprintf('Processing %s Type %s Scenario %d/%d...\n', ...
        AS_name, selected_type, s, num_scenarios);

    scenario_data = type_data(s);
    V_sd = scenario_data.V(:);
    ik   = scenario_data.I(:);
    t    = scenario_data.t(:);
    dt   = scenario_data.dt;
    dur  = scenario_data.dur;
    n    = scenario_data.n;
    
    % lambda(regularization)는 scenario_data에 저장되어 있다고 가정
    lambda = scenario_data.Lambda_hat; 
    
    % (6.1) DRT 추정 (R0 포함)
    [gamma_est, R0_est, V_est, theta_discrete, tau_discrete, W_aug, y] = ...
        SD_DRT_estimation_aug(t, ik, V_sd, lambda, n, dt, dur);
    
    % 결과 저장
    gamma_est_all{s}      = gamma_est(:);  % 열벡터화
    R0_est_all(s)         = R0_est;
    theta_discrete_all{s} = theta_discrete(:);

    % (6.2) Bootstrap (R0 포함)
    [gamma_lower, gamma_upper, gamma_resample_all] = ...
        bootstrap_uncertainty_aug(t, ik, V_sd, lambda, n, dt, dur, num_resamples);

    gamma_lower_all{s}              = gamma_lower(:);
    gamma_upper_all{s}              = gamma_upper(:);
    gamma_resample_all_scenarios{s} = gamma_resample_all;
end

%% (7) Plot Results

% (7.1) DRT comparison plot (all scenarios in one figure)
figure('Name', [AS_name, ' Type ', selected_type, ': DRT Comparison (All)'], 'NumberTitle', 'off');
hold on;
for s = 1:num_scenarios
    % theta, gamma
    theta_s = theta_discrete_all{s};
    gamma_s = gamma_est_all{s};
    
    % 에러바(errorbar) - bootstrap 결과 하한, 상한
    errorbar(theta_s, gamma_s, ...
        gamma_s - gamma_lower_all{s}, ...
        gamma_upper_all{s} - gamma_s, ...
        '--', 'LineWidth', 1.5, 'Color', c_mat(s,:), ...
        'DisplayName', ['Scenario ', num2str(SN_list(s))]);
end
% True gamma plot
plot(theta_true, gamma_discrete_true, 'k-', 'LineWidth', 2, 'DisplayName', 'True \gamma');
hold off;
xlabel('\theta = ln(\tau [s])', 'FontSize', labelFontSize);
ylabel('\gamma',               'FontSize', labelFontSize);
title([AS_name, ' Type ', selected_type, ': Estimated \gamma (All Scenarios)'], ...
    'FontSize', titleFontSize);
set(gca, 'FontSize', axisFontSize);
legend('Location', 'Best', 'FontSize', legendFontSize);
ylim([0 inf]);

% (7.2) 사용자 입력으로 시나리오 선택해서 plotting
disp('Available scenario numbers:');
disp(SN_list);
selected_scenarios = input('Enter scenario numbers to plot (e.g., [1,2,3]): ');

% (7.3) DRT comparison plot (selected scenarios)
figure('Name', [AS_name, ' Type ', selected_type, ': DRT Comparison (Selected)'], 'NumberTitle', 'off');
hold on;
for idx_s = 1:length(selected_scenarios)
    s_find = find(SN_list == selected_scenarios(idx_s));
    if ~isempty(s_find)
        theta_s = theta_discrete_all{s_find};
        gamma_s = gamma_est_all{s_find};
        
        errorbar(theta_s, gamma_s, ...
            gamma_s - gamma_lower_all{s_find}, ...
            gamma_upper_all{s_find} - gamma_s, ...
            '--', 'LineWidth', 1.5, 'Color', c_mat(s_find,:), ...
            'DisplayName', ['Scenario ', num2str(SN_list(s_find))]);
    else
        warning('Scenario %d not found in the data', selected_scenarios(idx_s));
    end
end
% True gamma plot
plot(theta_true, gamma_discrete_true, 'k-', 'LineWidth', 2, 'DisplayName', 'True \gamma');
hold off;
xlabel('\theta = ln(\tau [s])', 'FontSize', labelFontSize);
ylabel('\gamma',               'FontSize', labelFontSize);
title([AS_name, ' Type ', selected_type, ': Estimated \gamma (Selected Scenarios)'], ...
    'FontSize', titleFontSize);
set(gca, 'FontSize', axisFontSize);
legend('Location', 'Best', 'FontSize', legendFontSize);
ylim([0 inf]);

% (7.4) Individual scenario plots (subplot으로)
figure('Name', [AS_name, ' Type ', selected_type, ': Individual Scenario DRTs'], 'NumberTitle', 'off');
num_cols = 5;  
num_rows = ceil(num_scenarios / num_cols);

for s = 1:num_scenarios
    subplot(num_rows, num_cols, s);
    
    theta_s = theta_discrete_all{s};
    gamma_s = gamma_est_all{s};
    
    errorbar(theta_s, gamma_s, ...
        gamma_s - gamma_lower_all{s}, ...
        gamma_upper_all{s} - gamma_s, ...
        'LineWidth', 1.0, 'Color', c_mat(s,:));
    hold on;
    plot(theta_true, gamma_discrete_true, 'k-', 'LineWidth', 1.5);
    hold off;
    
    % -----------------------------
    %  R_0 값을 지수형식(Scientific Notation)으로 표시
    %  (예: 5e-13 -> 5.00000000e-13)
    % -----------------------------
    x_pos = min(theta_s) ;
    y_pos = max(gamma_s) ; 
    
    text(x_pos, y_pos, ...
         sprintf('R_0 = %.8e \\Omega', R0_est_all(s)), ...  % <-- 핵심 부분
         'FontSize', 10, 'Color', 'k');
    
    xlabel('\theta', 'FontSize', labelFontSize);
    ylabel('\gamma', 'FontSize', labelFontSize);
    title(['Scenario ', num2str(SN_list(s))], 'FontSize', titleFontSize);
    set(gca, 'FontSize', axisFontSize);
    ylim([0 inf]);
end

