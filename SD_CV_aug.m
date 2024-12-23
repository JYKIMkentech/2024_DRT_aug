%% SD_CV_aug.m
clc; clear; close all;

%% (1) 설정
axisFontSize = 14;
titleFontSize = 12;
legendFontSize = 12;
labelFontSize = 12;

lambda_grids = logspace(-8, 3, 5);
num_lambdas = length(lambda_grids);

OCV = 0; % 합성데이터 생성 시 0으로 고정 (이 스크립트에서는 사용 안 해도 무방)

%% (2) 데이터 로드
save_path = 'G:\공유 드라이브\Battery Software Lab\Projects\DRT\SD_lambda\';
file_path = 'G:\공유 드라이브\Battery Software Lab\Projects\DRT\SD_new\';
mat_files = dir(fullfile(file_path, '*.mat'));
if isempty(mat_files)
    error('데이터 파일이 존재하지 않습니다. 경로를 확인해주세요.');
end
for file = mat_files'
    load(fullfile(file_path, file.name));
end

%% (3) 데이터셋 선택
datasets = {'AS1_1per_new', 'AS1_2per_new', 'AS2_1per_new', 'AS2_2per_new'};
disp('데이터셋을 선택하세요:');
for i = 1:length(datasets)
    fprintf('%d. %s\n', i, datasets{i});
end
dataset_idx = input('데이터셋 번호를 입력하세요: ');
if isempty(dataset_idx) || dataset_idx < 1 || dataset_idx > length(datasets)
    error('유효한 데이터셋 번호를 입력해주세요.');
end
selected_dataset_name = datasets{dataset_idx};
if ~exist(selected_dataset_name, 'var')
    error('선택한 데이터셋이 로드되지 않았습니다.');
end
selected_dataset = eval(selected_dataset_name);

%% (4) 타입 선택
types = unique({selected_dataset.type});
disp('타입을 선택하세요:');
for i = 1:length(types)
    fprintf('%d. %s\n', i, types{i});
end
type_idx = input('타입 번호를 입력하세요: ');
if isempty(type_idx) || type_idx < 1 || type_idx > length(types)
    error('유효한 타입 번호를 입력해주세요.');
end
selected_type = types{type_idx};
type_indices = strcmp({selected_dataset.type}, selected_type);
type_data = selected_dataset(type_indices);
if isempty(type_data)
    error('선택한 타입에 해당하는 데이터가 없습니다.');
end
SN_list = [type_data.SN];

%% (5) CV에 필요한 새로운 필드 (Lambda_vec, CVE, Lambda_hat) 생성
new_fields = {'Lambda_vec', 'CVE', 'Lambda_hat'};
num_elements = length(selected_dataset);
empty_fields = repmat({[]}, 1, num_elements);

for nf = 1:length(new_fields)
    field_name = new_fields{nf};
    if ~isfield(selected_dataset, field_name)
        [selected_dataset.(field_name)] = empty_fields{:};
    end
end

%% (6) 람다 최적화 (교차 검증)
scenario_numbers = SN_list; 
validation_combinations = nchoosek(scenario_numbers, 2); % 10개라면 10C2=45
num_folds = size(validation_combinations, 1);

CVE_total = zeros(num_lambdas,1);

for m = 1 : num_lambdas
    lambda_test = lambda_grids(m);
    CVE = 0; 
    
    for f = 1 : num_folds
        val_trips = validation_combinations(f,:);
        train_trips = setdiff(1 : length(type_data), val_trips);
        
        % (6.1) Train 데이터 -> (W_aug_total, y_total) 생성
        W_aug_total = [];
        y_total     = [];
        for s = train_trips
            t   = type_data(s).t;
            dt  = [t(1); diff(t)];
            dur = type_data(s).dur;
            n   = type_data(s).n;
            I   = type_data(s).I;
            V   = type_data(s).V;
            
            % Augmented 형태로 W, y 생성
            [~, ~, ~, ~, ~, W_aug_s, y_s] = SD_DRT_estimation_aug(t, I, V, lambda_test, n, dt, dur); 
            % y_s = V (OCV=0), W_aug_s(:,1:n)=RC적분, W_aug_s(:,end)=I

            % 누적
            W_aug_total = [W_aug_total; W_aug_s];
            y_total     = [y_total; y_s];
        end
        
        % (6.2) 학습데이터를 이용해 gamma + R0 추정
        [params_train] = SD_DRT_estimation_with_Wy_aug(W_aug_total, y_total, lambda_test);
        % params_train = [gamma_est; R0_est]

        % (6.3) Validation 데이터 -> MSE 계산
        for s = val_trips
            t   = type_data(s).t;
            dt  = [t(1); diff(t)];
            dur = type_data(s).dur;
            n   = type_data(s).n;
            I   = type_data(s).I;
            V   = type_data(s).V;
            
            [~, ~, ~, ~, ~, W_aug_val, ~] = SD_DRT_estimation_aug(t, I, V, lambda_test, n, dt, dur);
            V_est_val = OCV + W_aug_val * params_train; 
            err = sum((V - V_est_val).^2);
            CVE = CVE + err;
        end
    end
    
    CVE_total(m) = CVE;
    fprintf('Lambda: %.2e, CVE: %.4f\n', lambda_test, CVE_total(m));
end

[~, optimal_idx] = min(CVE_total);
optimal_lambda = lambda_grids(optimal_idx);

%% (7) 결과 저장
for i = 1:length(type_data)
    type_data(i).Lambda_vec = lambda_grids;
    type_data(i).CVE        = CVE_total;
    type_data(i).Lambda_hat = optimal_lambda;
end
selected_dataset(type_indices) = type_data;
assignin('base', selected_dataset_name, selected_dataset);

%% (8) 데이터 저장
if ~exist(save_path, 'dir')
    mkdir(save_path);
end
save(fullfile(save_path, [selected_dataset_name, '.mat']), selected_dataset_name);
fprintf('Updated dataset saved to %s\n', ...
    fullfile(save_path, [selected_dataset_name, '.mat']));

%% (9) Plot (CVE vs lambda)
figure;
semilogx(lambda_grids, CVE_total, 'b-', 'LineWidth', 1.5); hold on;
semilogx(optimal_lambda, CVE_total(optimal_idx), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('\lambda', 'FontSize', labelFontSize);
ylabel('CVE', 'FontSize', labelFontSize);
title('CVE vs \lambda ', 'FontSize', titleFontSize);
grid on;
legend({'CVE', ['Optimal \lambda = ', num2str(optimal_lambda, '%.2e')]}, 'Location', 'best');
hold off;

%% SD_DRT_estimation_with_Wy_aug.m
function [params_train] = SD_DRT_estimation_with_Wy_aug(W_aug_total, y_total, lambda)
% SD_DRT_estimation_with_Wy_aug 
%   - Augmented matrix (W_aug_total)와 측정치(y_total)로부터
%     gamma(n개) + R0(1개) 파라미터를 동시에 추정한다.
%   - 1차 차분 규제(regularization)는 gamma에 대해서만 적용, R0에는 적용 X.
%
% Inputs:
%   W_aug_total : [W, i]가 이어붙여진 (Nx(n+1)) 행렬
%   y_total     : 전압 측정값 (Nx1, OCV=0 가정)
%   lambda      : 정규화 파라미터 (regularization strength)
%
% Output:
%   params_train: [ gamma_est (n x 1); R0_est (1 x 1) ] 크기의 추정 파라미터 벡터

    % (1) gamma 개수 확인
    n_params = size(W_aug_total, 2) - 1; 
    %   ex) W_aug_total의 열 = (n + 1)
    %       맨 마지막 열은 R0와 곱해질 i(k) 이므로, 나머지 n열이 gamma

    % (2) 1차 차분 행렬 (gamma에만 적용하기 위해 크기 n-1 x n)
    L = zeros(n_params - 1, n_params);
    for i = 1 : (n_params - 1)
        L(i, i)   = -1;
        L(i, i+1) =  1;
    end
    % R0 에 대해선 규제항 없음 -> L_aug 크기 (n-1 x (n+1)) 이고 마지막 열(=R0 부분)은 0
    L_aug = [L, zeros(n_params - 1, 1)];

    % (3) QP 용 행렬
    H = 2 * (W_aug_total' * W_aug_total + lambda * (L_aug' * L_aug));
    f = -2 * W_aug_total' * y_total;

    % (4) 제약조건 (gamma >= 0, R0 >= 0)
    A_ineq = -eye(n_params + 1); 
    b_ineq = zeros(n_params + 1, 1);

    % (5) quadprog 옵션
    options = optimoptions('quadprog','Display','off');

    % (6) quadprog 해
    params_train = quadprog(H, f, A_ineq, b_ineq, [], [], [], [], [], options);
    %   params_train(1:n_params) = gamma
    %   params_train(end)        = R0

end



