clear;clc;


iter = 10000; % number of iteration of the SVM training process
conv_threshold = 5e-4; % convergence threshold for early stopping of the SVM training process
theta = 1.5; % weighting factor theta

size_gen_pool = 300000;
num_subsample = 3000;
num_fs = 300; % number of nearest samples to the SVM hyper-plane
num_ori_human = 753;
num_ori_spoof = 753;
dim_fea = 864*400;

load('.../tr_A.mat');
load('.../tr_B.mat');
load('.../generation_pool.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%  step(c)  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Prepare the training sets A and B
tr_A = cat(1,tr_A(1:num_ori_human,:,:),tr_A(num_ori_human+2:num_ori_human+2+num_ori_spoof-1,:,:));
trainingset_A = zeros(num_ori_human+num_ori_spoof,dim_fea);
for i = 1:size(trainingset_A,1)
trainingset_A(i,:) = reshape(tr_A(i,:,:), [1 dim_fea]);
end
trainingset_A = mapminmax(trainingset_A);
label_trA = [zeros(num_ori_human,1); ones(num_ori_spoof,1)]; % 0 human 1 spoof

trainingset_B = zeros(num_ori_human+num_ori_spoof,dim_fea);
for i = 1:size(trainingset_B,1)
trainingset_B(i,:) = reshape(tr_B(i,:,:), [1 dim_fea]);
end
trainingset_B = mapminmax(trainingset_B);
label_trB = [zeros(num_ori_human,1); ones(num_ori_spoof,1)]; % 0 human 1 spoof
% Train SVM model with tr_A
SVMModel = fitcsvm(trainingset_A,label_trA,'KernelFunction','linear');
% Predict tr_A and tr_B on trained SVM model
[test_label_trB,score_trB] = predict(SVMModel,trainingset_B);
[test_label_trA,score_trA] = predict(SVMModel,trainingset_A);
% Calculate the SVM score
CP_trA = classperf(label_trA, test_label_trA);
CP_trB = classperf(label_trB, test_label_trB);
svm_score = CP_trA.CorrectRate + theta.*CP_trB.CorrectRate;


score_diff = zeros(iter,1);
model_change_flag = zeros(iter,1); % 1 -- SVM model updated; 0 -- no update
sifted_samples_pool = [];
for cnt = 1:iter
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%  step(d)  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    idx = randi([1,size_gen_pool],num_subsample,1);
    gen_subsample = generation_pool(idx,:,:);
    testingset_gen_subsample = zeros(num_subsample,dim_fea);
    for i = 1:num_subsample
        testingset_gen_subsample(i,:) = reshape(gen_subsample(i,:,:), [1 dim_fea]);
    end
    [test_label_gen_sub,score_gen_sub] = predict(SVMModel,testingset_gen_subsample);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%  step(e)  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    idx_min = knnsearch(score_gen_sub(:,1),0,'K',num_fs);
    fake_samples = testingset_gen_subsample(idx_min,:);
    fake_samples_label = test_label_gen_sub(idx_min,:);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%  step(f)  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    for j = 1:num_fs
       fake_samples(j) =  fake_samples(j) + mean(fake_samples(j)).*randn(size(fake_samples(j)));
    end
    
    trainingset_A_fs = [trainingset_A; fake_samples];
    label_A_fs = [zeros(num_ori_human,1); ones(num_ori_spoof,1); zeros(size(fake_samples_label))];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%  step(g)  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    SVMModel_merge = fitcsvm(trainingset_A_fs,label_A_fs,'KernelFunction','linear');
    [test_label_trA_merge,score_trA_merge] = predict(SVMModel_merge,trainingset_A);
    [test_label_trB_merge,score_trB_merge] = predict(SVMModel_merge,trainingset_B);
    
    CP_trA_merge = classperf(label_trA, test_label_trA_merge);
    CP_trB_merge = classperf(label_trB, test_label_trB_merge);
    svm_score_merge = CP_trA_merge.CorrectRate + theta.*CP_trB_merge.CorrectRate;
    score_diff(cnt,1) = svm_score_merge - svm_score;
    
    if (abs(score_diff(cnt,1)) > conv_threshold) || (abs(score_diff(cnt,1)) == conv_threshold)
        if score_diff(cnt,1) > 0 
            SVMModel = SVMModel_merge;
            model_change_flag(cnt,1) = 1;
        else
            SVMModel = SVMModel;
            model_change_flag(cnt,1) = 0;
        end
    else
        
       continue
    end
    
    sv = SVMModel.SupportVectors;
    sv_labels = SVMModel.SupportVectorLabels;
    sifted_samples_idx = find(sv_labels==-1);
    sifted_samples_idx = sifted_samples_idx(sifted_samples_idx>=90);
    sifted_samples = sv(sifted_samples_idx,:);
    sifted_samples_pool = [sifted_samples_pool; sifted_samples];
    
    cnt
end














