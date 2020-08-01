function features_and_labels = preprocess_data(data_cell_array)

% Input: Expecting eeg data to be a 1x9 cell array, where the last 6
% elements represent each run
% Output: features_and_labels will be an array of structs with the length
% equal to the total number of non-artifact samples per subject

features_and_labels(10000,1) = struct('features',[],'label',[]);
num_samples = 1;

% Iterating over each run
for eachRunIdx = 4:length(data_cell_array)
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
            
            
            % finally append eachInput to our features and labels array
            features_and_labels(num_samples) = eachInput;
            num_samples = num_samples + 1;
        end
    end
end

features_and_labels = features_and_labels(1:num_samples-1);