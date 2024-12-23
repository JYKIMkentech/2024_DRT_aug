clear; clc; close all;

%% Load specific .mat files from the directory
data_dir = 'G:\공유 드라이브\Battery Software Lab\Projects\DRT\SD';
output_dir = 'G:\공유 드라이브\Battery Software Lab\Projects\DRT\SD_new';

% 네 개의 특정 파일 목록 생성
file_names = {'AS1_1per.mat', 'AS1_2per.mat', 'AS2_1per.mat', 'AS2_2per.mat'};
mat_files = [];

% 네 개의 파일에 대해 반복하여 경로 정보를 얻기
for i = 1:length(file_names)
    file_path = fullfile(data_dir, file_names{i});
    mat_files = [mat_files; dir(file_path)];
end

% 모든 .mat 파일에 대해 반복
for fileIdx = 1:length(mat_files)
    % 파일 이름과 경로 설정
    data_file = fullfile(mat_files(fileIdx).folder, mat_files(fileIdx).name);
    [~, name, ~] = fileparts(data_file); % 파일명에서 이름 부분 추출 (예: 'AS2_1per')

    % 데이터 로드
    loaded_struct = load(data_file);
    data = loaded_struct.(name);

    %% 미리 할당할 총 조합 수 계산
    num_combinations = 0;

    % 조합 1: dt=0.1, duration=1000, n=[201, 101, 21]
    num_combinations = num_combinations + length([201, 101, 21]);

    % 조합 2: dt=[0.2, 1, 2], duration=1000, n=201
    num_combinations = num_combinations + length([0.2, 1, 2]);

    % 조합 3: dt=0.1, duration=[500, 250], n=201
    num_combinations = num_combinations + length([500, 250]);

    % combinations 구조체 배열 미리 할당 (모든 필드를 미리 정의)
    combinations = struct('dt', cell(1, num_combinations), ...
                          'duration', cell(1, num_combinations), ...
                          'n', cell(1, num_combinations), ...
                          'type', cell(1, num_combinations));

    idx = 1;

    % Define type list
    type_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];
    num_types = length(type_list);

    % 조합 1
    dt = 0.1;
    duration = 1000;
    n_list = [201, 101, 21];
    for i = 1:length(n_list)
        n = n_list(i);
        combinations(idx).dt = dt;
        combinations(idx).duration = duration;
        combinations(idx).n = n;
        combinations(idx).type = type_list(mod(idx-1, num_types) + 1);
        idx = idx + 1;
    end

    % 조합 2
    dt_list = [0.2, 1, 2];
    duration = 1000;
    n = 201;
    for i = 1:length(dt_list)
        dt = dt_list(i);
        combinations(idx).dt = dt;
        combinations(idx).duration = duration;
        combinations(idx).n = n;
        combinations(idx).type = type_list(mod(idx-1, num_types) + 1);
        idx = idx + 1;
    end

    % 조합 3
    dt = 0.1;
    duration_list = [500, 250];
    n = 201;
    for i = 1:length(duration_list)
        duration = duration_list(i);
        combinations(idx).dt = dt;
        combinations(idx).duration = duration;
        combinations(idx).n = n;
        combinations(idx).type = type_list(mod(idx-1, num_types) + 1);
        idx = idx + 1;
    end

    % 결과를 저장할 구조체 배열을 이름으로 할당
    new_data_name = [name '_new']; % 예: 'AS2_1per_new'
    eval([new_data_name ' = struct(''dt'', [], ''dur'', [], ''n'', [], ''SN'', [], ''V'', [], ''I'', [], ''t'', [], ''type'', []);']);
    total_results = num_combinations * 10;

    % 모든 조합에 대해 데이터 처리
    index = 1;
    for i = 1:length(combinations)

        dt = combinations(i).dt;
        duration = combinations(i).duration;
        n = combinations(i).n;
        type = combinations(i).type;

        for j = 1:10
            % Check if the fields 'SN', 'V', 'I', 't' exist in data(j)
            if isfield(data, 'SN') && isfield(data(j), 'V') && isfield(data(j), 'I') && isfield(data(j), 't')
                % 원본 데이터 가져오기
                SN = data(j).SN;
                V_orig = data(j).V;
                I_orig = data(j).I;
                t_orig = data(j).t;

                % duration에 따른 데이터 리샘플링
                dur = t_orig <= duration;
                V_dur = V_orig(dur);
                I_dur = I_orig(dur);
                t_dur = t_orig(dur);

                % dt에 따른 데이터 리샘플링
                step = round(dt/0.1);
                V_new = V_dur(1:step:end);
                I_new = I_dur(1:step:end);
                t_new = t_dur(1:step:end);

                % 결과 저장
                eval([new_data_name '(index).SN = SN;']);
                eval([new_data_name '(index).dt = dt;']);
                eval([new_data_name '(index).dur = duration;']);
                eval([new_data_name '(index).n = n;']);
                eval([new_data_name '(index).V = V_new;']);
                eval([new_data_name '(index).I = I_new;']);
                eval([new_data_name '(index).t = t_new;']);
                eval([new_data_name '(index).type = type;']);
                index = index + 1;
            else
                disp(['Field missing in data: ', mat_files(fileIdx).name, ', entry index: ', num2str(j)]);
            end
        end
    end

    % 동일한 이름으로 새 파일 저장 (확장자 변경)
    output_file = fullfile(output_dir, [name '_new.mat']);
    eval(['save(output_file, ''' new_data_name ''');']); % 자동으로 이름 매칭해서 저장

    disp(['파일 처리 완료: ', name]);
end
