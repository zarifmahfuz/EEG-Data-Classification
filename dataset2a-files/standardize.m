function stdFeatures = standardize(features, population)
    % Input:
    % features: expecting an array of structs containing the following
    % fields: features, label
    % population: expecting an array of structs containing the following
    % fields: label, alphaMean, alphaVar, betaMean, betaVar
    %
    % Returns: 
    % array of structs containing standardized feature values following the
    % formula, stdX = (x - populationMean)/populationSD
    
    numChannels = 22;
    
    for i = 1:length(features)
        eachSample = features(i);
        updatedFeatures = zeros(2, numChannels);
        
        % 2xnumChannels vector with alpha and beta relative energies for each channel 
        sampleFeatures = eachSample.features; 
        sampleLabel = eachSample.label;
        
        labelStats = population(sampleLabel);
        alphaFeat = sampleFeatures(1,:); % 1xnumChannels vector 
        betaFeat = sampleFeatures(2,:); % 1xnumChannels vector
        
        updatedFeatures(1,:) = ((alphaFeat' - labelStats.alphaMean)./sqrt(labelStats.alphaVar))';
        updatedFeatures(2,:) = ((betaFeat' - labelStats.betaMean)./sqrt(labelStats.betaVar))';
        
        features(i).features = updatedFeatures;
        
    end
    
    stdFeatures = features;
end