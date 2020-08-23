function [features_and_labels, population] = preprocess_data(data_cell_array, runStart)

% Input: Expecting eeg data to be a 1x9 cell array, where the last 6
% elements represent each run
% Output: features_and_labels will be an array of structs with the length
% equal to the total number of non-artifact samples per subject

features_and_labels(10000,1) = struct('features',[],'label',[]);
num_samples = 1;

% I need to standardize the feature set; to do that, I need to track every
% relative energy value for each band, each channel for each label
% for label 1
alphaPopulation1 = zeros(22,10000);
betaPopulation1 = zeros(22,10000);
tracker1 = 1;

% for label 2
alphaPopulation2 = zeros(22,10000);
betaPopulation2 = zeros(22,10000);
tracker2 = 1;

% for label 3
alphaPopulation3 = zeros(22,10000);
betaPopulation3 = zeros(22,10000);
tracker3 = 1;

% for label 4
alphaPopulation4 = zeros(22,10000);
betaPopulation4 = zeros(22,10000);
tracker4 = 1;

% Iterating over each run
for eachRunIdx = runStart:length(data_cell_array)
    eachRun = data_cell_array(eachRunIdx);
    eachRun = eachRun{1};
    
    % X will be a channel x timepoints matrix of each run's EEG data
    X = eachRun.X';
    
    % Only interested in the first 22 channels
    numChannels = 22;
    windowLength = 1750;
    
    % Each value in this vector represent the trial onset
    trials = eachRun.trial;
    
    % This represents the label for each trial
    y = eachRun.y;
    artifacts = eachRun.artifacts;
    
    for eachTrial = 1:length(trials)
        % if the trial has no artifacts
        if artifacts(eachTrial) == 0
            regionStart = trials(eachTrial);
            regionEnd = regionStart + windowLength - 1;
            interestedRegion = X(1:numChannels, regionStart:regionEnd);
            
            % Each struct will serve as an input to the classifier; each row in the
            % features field represents a beta and alpha band Relative Energy vector
            % for each sample or label window 
            eachInput = struct;
            eachFeature = zeros(2,numChannels);
            
            % iterate over each channel and compute Relative Energy for
            % Beta and Alpha frequency sub-bands
            for channel = 1:numChannels
                channelSignal = interestedRegion(channel,:);
                
                [gamma, beta, alpha, garbage] = getabs_power(channelSignal, 'db4');
                total = gamma + beta + alpha + garbage;
                
                alphaRE = alpha/total;
                betaRE = beta/total;
                
                % first row represents alpha band and second row represents
                % gamma band relative energies
                eachFeature(1,channel) = alphaRE;
                eachFeature(2,channel) = betaRE;              
            end
            
            trialLabel = y(eachTrial);
            eachInput.features = eachFeature;
            eachInput.label = trialLabel;
            
            % append to population
            if trialLabel == 1
                alphaPopulation1(:,tracker1) = eachFeature(1,:)';
                betaPopulation1(:,tracker1) = eachFeature(2,:)';
                tracker1 = tracker1 + 1;
            
            elseif trialLabel == 2
                alphaPopulation2(:,tracker2) = eachFeature(1,:)';
                betaPopulation2(:,tracker2) = eachFeature(2,:)';
                tracker2 = tracker2 + 1;   
                
            elseif trialLabel == 3
                alphaPopulation3(:,tracker3) = eachFeature(1,:)';
                betaPopulation3(:,tracker3) = eachFeature(2,:)';
                tracker3 = tracker3 + 1;
                
            elseif trialLabel == 4
                alphaPopulation4(:,tracker4) = eachFeature(1,:)';
                betaPopulation4(:,tracker4) = eachFeature(2,:)';
                tracker4 = tracker4 + 1;
                
            end
            
            % finally append eachInput to our features and labels array
            features_and_labels(num_samples) = eachInput;
            num_samples = num_samples + 1;
        end
    end
end

features_and_labels = features_and_labels(1:num_samples-1);

alphaPopulation1 = alphaPopulation1(:,1:tracker1-1);
betaPopulation1 = betaPopulation1(:,1:tracker1-1);

alphaPopulation2 = alphaPopulation2(:,1:tracker2-1);
betaPopulation2 = betaPopulation2(:,1:tracker2-1);

alphaPopulation3 = alphaPopulation3(:,1:tracker3-1);
betaPopulation3 = betaPopulation3(:,1:tracker3-1);

alphaPopulation4 = alphaPopulation4(:,1:tracker4-1);
betaPopulation4 = betaPopulation4(:,1:tracker4-1);

population(4,1) = struct('label',[],'alphaMean',[],'alphaVar',[],'betaMean',[],'betaVar',[]);

population(1).label = 1;
population(1).alphaMean = mean(alphaPopulation1,2); % column vector containing mean of each channel
population(1).alphaVar = var(alphaPopulation1,0,2); % column vector containing variance of each channel
population(1).betaMean = mean(betaPopulation1,2);
population(1).betaVar = var(betaPopulation1,0,2);

population(2).label = 2;
population(2).alphaMean = mean(alphaPopulation2,2); 
population(2).alphaVar = var(alphaPopulation2,0,2);
population(2).betaMean = mean(betaPopulation2,2);
population(2).betaVar = var(betaPopulation2,0,2);

population(3).label = 3;
population(3).alphaMean = mean(alphaPopulation3,2); 
population(3).alphaVar = var(alphaPopulation3,0,2); 
population(3).betaMean = mean(betaPopulation3,2);
population(3).betaVar = var(betaPopulation3,0,2);

population(4).label = 4;
population(4).alphaMean = mean(alphaPopulation4,2); 
population(4).alphaVar = var(alphaPopulation4,0,2); 
population(4).betaMean = mean(betaPopulation4,2);
population(4).betaVar = var(betaPopulation4,0,2);

end