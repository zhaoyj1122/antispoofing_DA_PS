clear; close all; clc;

% add required libraries to the path
addpath(genpath('utility'));
addpath(genpath('CQCC_v1.0'));
addpath(genpath('bosaris_toolkit'));

% set paths to the wave files and protocols
pathToDatabase = fullfile('.../ASVspoofing2017_v2');
trainProtocolFile = fullfile('.../ASVspoofing2017_v2/protocol_V2/ASVspoof2017_V2_train.trn.txt');
devProtocolFile = fullfile('.../ASVspoofing2017_v2/protocol_V2/ASVspoof2017_V2_dev.trl.txt');
evaProtocolFile = fullfile('.../ASVspoofing2017_v2/protocol_V2/ASVspoof2017_V2_eval.trl.txt');

% read protocol
fileID = fopen(trainProtocolFile);
protocol = textscan(fileID, '%s%s%s%s%s%s%s');
fclose(fileID);

% get file and label lists
filelist = protocol{1};
% labels = protocol{2};

parfor i=1:length(filelist)
    filePath = fullfile(pathToDatabase,'ASVspoof2017_V2_train',filelist{i});
    [x,fs] = audioread(filePath);
    % featrue extraction
    [~,cqtFeatureCell{i}] = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');

end
disp('Done!');

% read development protocol
fileID = fopen(devProtocolFile);
protocol = textscan(fileID, '%s%s%s%s%s%s%s');
fclose(fileID);

% get file and label lists
filelist = protocol{1};
% labels = protocol{2};

parfor i=1:length(filelist)
    filePath = fullfile(pathToDatabase,'ASVspoof2017_V2_dev',filelist{i});
    [x,fs] = audioread(filePath);
    % featrue extraction
    [~,cqtFeatureCell{i}] = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');

end
disp('Done!');

% read eval protocol
fileID = fopen(evaProtocolFile);
protocol = textscan(fileID, '%s%s%s%s%s%s%s');
fclose(fileID);

% get file and label lists
filelist = protocol{1};
% labels = protocol{2};


parfor i=1:length(filelist)
    filePath = fullfile(pathToDatabase,'ASVspoof2017_V2_eval',filelist{i});
    [x,fs] = audioread(filePath);
    % featrue extraction
    [~,cqtFeatureCell{i}] = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');

end
disp('Done!');

