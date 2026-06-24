clc;
clear;
close all;

% Define path to saliency generation repository

addpath(genpath("/path/to/repository"))

totalTime = 0;
totalSamples = 0;

% Define input and output directories

inputDir = '/path/to/imagefolders';

outputDir = '/path/to/saliencyfolders';

% Create output directory if it doesn't exist
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Get list of all image files in the subdirectory
imageFiles = dir(fullfile(inputDir, '*.jpg'));

fprintf('\n=== DEBUG START ===\n');
fprintf('Input directory: %s\n', inputDir);
fprintf('Images found: %d\n', length(imageFiles));

if isempty(imageFiles)
    disp('⚠️ No images detected. Check extension or path.');
else
    disp('Images detected:');
    disp({imageFiles.name});
end
fprintf('===================\n\n');

    % Process each image
for j = 1:length(imageFiles)
    imageName = imageFiles(j).name;

    inputImagePath = fullfile(inputDir, imageName);
        
    % Modify the output filename to include "ProtoObject"
    [~, name, ext] = fileparts(imageName); % Extract filename and extension
    newImageName = strcat(name, '_ProtoObject', ext); % Append "_ProtoObject" before the extension
    outputImagePath = fullfile(outputDir, newImageName); % Construct new output path

        % Check if saliency map already exists
    if exist(outputImagePath, 'file')
        fprintf('Skipped (already exists): %s\n', outputImagePath);
        continue;
    end

        % store image dimensions
    img = imread(inputImagePath);
    [rows, cols, channels] = size(img);

        % Start measuring time
    tic;

    result=runProtoSalTexMax(inputImagePath,'ICOR');
        
    salmap = result.data; % get saliency map
        
    normalizedSalMap = mat2gray(salmap);

        % resize image to original dimensions (since ProtoObject shrinks it)
        
    normalizedSalMap = imresize(normalizedSalMap, [rows, cols]);

        % Measure elapsed time
    elapsedTime = toc;
    totalTime = totalTime + elapsedTime;
    totalSamples = totalSamples + 1;

    imwrite(normalizedSalMap, outputImagePath) % save image

    fprintf('Processed and saved: %s\n', outputImagePath);

    close all;
end

% SAVE STATS

timePerSample = totalTime/totalSamples;

folderName = '/path/to/folder'; % Define folder name to save stats
% Check if the folder exists, if not, create it
if ~exist(folderName, 'dir')
    mkdir(folderName);
end
% Define file name and full path
fileName = 'stats.csv';
filePath = fullfile(folderName, fileName);
modelName = "ProtoObject_320_Val"; % This will be added as a column in the CSV
% Data to save
data = {modelName, totalTime, totalSamples, timePerSample};
% Check if file exists
if exist(filePath, 'file') == 2
    % Append new data to existing file
    writecell(data, filePath, 'WriteMode', 'append');
else
    % Write with header if the file does not exist
    header = ["Model", "TotalTime", "TotalSamples", "TimePerSample"];
    writematrix(header, filePath);
    writecell(data, filePath, 'WriteMode', 'append');
end

disp('Processing complete.');
