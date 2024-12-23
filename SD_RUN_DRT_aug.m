%% SD_Run_DRT_aug.m
clc; clear; close all;

%% (1) Description
% - 여러 시나리오(each AS_data)에 대해 DRT 추정 (gamma, R0)
% - Bootstrap으로 불확실성 평가 후 결과 플롯

%% (2) Graphic Parameters
axisFontSize = 14;
titleFontSize = 12;
legendFontSize = 12;
labelFontSize = 12;

%% (3) Load Data
file_path = 'G:\공유 드라이브\Battery Software Lab\Projects\DRT\SD_lambda\';
mat_files = dir(fullfile(file_path, '*.mat'));
for file = mat_files'
    load(fullfile(file_path, file.name));  % 여러 변수(AS1_1per_new 등) 로드
end

%% (4) Parameters
% - 예시: AS_structs, AS_names, Gamma_structs 은 기존과 동일하게 구성
AS_structs = {AS1_1per_new, AS1_2per_new, AS2_1per_new, AS2_2per_new};
AS_names = {'AS1_1per_new', 'AS1_2per_new', 'AS2_1per_new', 'AS2_2per_new'};
Gamma_structs = {Gamma_unimodal, Gamma_unimodal, Gamma_bimodal, Gamma_bimodal};

fprintf('Available datasets:\n');
for idx = 1:length(AS_names)
    fprintf('%d: %s\n', idx, AS_names{idx});
end
dataset_idx = input('Select a dataset to process (enter the number): ');
AS_data = AS_structs{dataset_idx};
AS_name = AS_names{dataset_idx};
Gamma_data = Gamma_structs{dataset_idx};

% 추정해야 하는 파라미터로 R0를 포함
% (합성 데이터에서 OCV=0으로 생성했다 가정하므로, OCV=0은 상수로 두고 R0는 추정)
% lambda는 시나리오마다(또는 Cross Validation 통해) 구해둔 값이 있다고 가정.
% 예: scenario_data.Lambda_hat

% Type(실험 조건) 선택
types = unique({AS_data.type});
disp('Select a type:');
for i = 1:length(types)
    fprintf('%d. %s\n', i, types{i});
end
type_idx = input('Enter the type number: ');
selected_type = types{type_idx};

type_indices = find(strcmp({AS_data.type}, selected_type));
type_data = AS_data(type_indices);

num_scenarios = length(type_data);
SN_list = [type_data.SN];

fprintf('Selected dataset: %s\n', AS_name);
fprintf('Selected type: %s\n', selected_type);
fprintf('Scenario numbers: ');
disp(SN_list);

% Plot color matrix
c_mat = lines(num_scenarios);

%% (5) True Gamma (for comparison)
gamma_discrete_true = Gamma_data.gamma'; 
theta_true = Gamma_data.theta';  

%% (6) DRT 및 Bootstrap 수행
% 결과 저장용 cell
gamma_est_all   = cell(num_scenarios, 1);
R0_est_all      = zeros(num_scenarios, 1);
theta_discrete_all = cell(num_scenarios, 1);
gamma_lower_all = cell(num_scenarios, 1);
gamma_upper_all = cell(num_scenarios, 1);

num_resamples = 1000;  % bootstrap 횟수
gamma_resample_all_scenarios = cell(num_scenarios, 1);

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
    
    % lambda(regularization)는 scenario_data에 저장되었다고 가정
    lambda = scenario_data.Lambda_hat; 
    if isempty(lambda)
       lambda = 0.1;  % (임시) 값
    end

    % (6.1) DRT 추정 (R0 포함)
    [gamma_est, R0_est, V_est, theta_discrete, tau_discrete, W_aug, y] = SD_DRT_estimation_aug(t, ik, V_sd, lambda, n, dt, dur);

    gamma_est_all{s} = gamma_est;
    R0_est_all(s)    = R0_est;
    theta_discrete_all{s} = theta_discrete;

    % (6.2) Bootstrap (R0 포함)
    [gamma_lower, gamma_upper, gamma_resample_all] = bootstrap_uncertainty_aug(t, ik, V_sd, lambda, n, dt, dur, num_resamples);

    gamma_lower_all{s} = gamma_lower;
    gamma_upper_all{s} = gamma_upper;
    gamma_resample_all_scenarios{s} = gamma_resample_all;
end

%% (7) Plot: 모든 시나리오 비교
figure('Name', [AS_name, ' Type ', selected_type, ': DRT Comparison with Uncertainty'], ...
       'NumberTitle', 'off');
hold on;
for s = 1:num_scenarios
    theta_s = theta_discrete_all{s};
    gamma_s = gamma_est_all{s};
    errbar_lower = gamma_s - gamma_lower_all{s};
    errbar_upper = gamma_upper_all{s} - gamma_s;

    errorbar(theta_s, gamma_s, errbar_lower, errbar_upper, '--', ...
        'LineWidth', 1.5, 'Color', c_mat(s, :), ...
        'DisplayName', ['Scen ', num2str(SN_list(s)), ', R0=', num2str(R0_est_all(s),'%.3f')]);
end
plot(theta_true, gamma_discrete_true, 'k-', 'LineWidth', 2, 'DisplayName', 'True \gamma');
xlabel('\theta = ln(\tau [s])', 'FontSize', labelFontSize);
ylabel('\gamma', 'FontSize', labelFontSize);
title([AS_name, ' Type ', selected_type, ': Estimated \gamma with Uncertainty'], ...
       'FontSize', titleFontSize);
set(gca, 'FontSize', axisFontSize);
legend('Location', 'Best', 'FontSize', legendFontSize);
ylim([0 inf])
hold off;

%% (8) 원하는 시나리오만 선택해서 플롯
disp('Available scenario numbers:');
disp(SN_list);
selected_scenarios = input('Enter scenario numbers to plot (e.g., [1,2,3]): ');

figure('Name', [AS_name, ' Type ', selected_type, ': Selected Scenarios DRT'], ...
       'NumberTitle', 'off');
hold on;
for idx_s = 1:length(selected_scenarios)
    s = find(SN_list == selected_scenarios(idx_s));
    if ~isempty(s)
        theta_s = theta_discrete_all{s};
        gamma_s = gamma_est_all{s};
        errbar_lower = gamma_s - gamma_lower_all{s};
        errbar_upper = gamma_upper_all{s} - gamma_s;

        errorbar(theta_s, gamma_s, errbar_lower, errbar_upper, '--', ...
            'LineWidth', 1.5, 'Color', c_mat(s, :), ...
            'DisplayName', ['Scen ', num2str(SN_list(s)), ', R0=', num2str(R0_est_all(s),'%.3f')]);
    else
        warning('Scenario %d not found', selected_scenarios(idx_s));
    end
end
plot(theta_true, gamma_discrete_true, 'k-', 'LineWidth', 2, 'DisplayName', 'True \gamma');
xlabel('\theta = ln(\tau [s])');
ylabel('\gamma');
title([AS_name, ' Type ', selected_type, ': Selected Scenarios'], 'FontSize', titleFontSize);
legend('Location', 'Best');
ylim([0 inf])
hold off;

%% (9) 개별 시나리오 subplot
figure('Name', [AS_name, ' Type ', selected_type, ': Individual Scenarios'], ...
       'NumberTitle', 'off');
num_cols = 5;  
num_rows = ceil(num_scenarios / num_cols);

for s = 1:num_scenarios
    subplot(num_rows, num_cols, s);
    theta_s = theta_discrete_all{s};
    gamma_s = gamma_est_all{s};
    errbar_lower = gamma_s - gamma_lower_all{s};
    errbar_upper = gamma_upper_all{s} - gamma_s;

    errorbar(theta_s, gamma_s, errbar_lower, errbar_upper, 'LineWidth', 1.0, ...
        'Color', c_mat(s, :));
    hold on;
    plot(theta_true, gamma_discrete_true, 'k-', 'LineWidth', 1.5);
    hold off;
    xlabel('\theta', 'FontSize', labelFontSize);
    ylabel('\gamma', 'FontSize', labelFontSize);
    title(['Scenario ', num2str(SN_list(s)), ' (R0=', num2str(R0_est_all(s),'%.2f'), ')'], ...
          'FontSize', titleFontSize);
    set(gca, 'FontSize', axisFontSize);
    ylim([0 inf])
end
