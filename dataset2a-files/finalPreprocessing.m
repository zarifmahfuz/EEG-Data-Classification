function finalPreprocessing(filesDir, numSubjects, trainOrEval)
    % Input: 
    % filesDir: directory containing the raw .mat files e.g. "data/"
    % numSubjects: number of raw data files that you want to process
    % trainOrEval: 1 for building training dataset and 2 for building
    % evaluation dataset
    
    % Returns:
    % Saves processed data for each subject in the subfolder
    % "processedData/" relative to the current path; 
    
    if isfolder(filesDir) == 1
        % cd(filesDir);
        for i = 1:numSubjects
            if trainOrEval == 1
                % need to process training data
                load([filesDir '/A0' num2str(i) 'T.mat']);
            elseif trainOrEval == 2
                load([filesDir '/A0' num2str(i) 'E.mat']);
            end
            
            runStart = 4;
            if trainOrEval == 1 && i == 4
                % the training dataset for subject 4 starts from run 2
                runStart = 2;
            end
            
            [features, pop] = preprocess_data(data, runStart);
            features = standardize(features, pop);
            
            % save the processed data
            if isfolder('processedData2a/') == 0
                mkdir('processedData2a/');
            end
            
            if trainOrEval == 1
                save(['processedData2a/' 'processed0' num2str(i) 'T.mat'], 'features');
            elseif trainOrEval == 2
                save(['processedData2a/' 'processed0' num2str(i) 'E.mat'], 'features');
            end
            
        end
    else
        errorMessage = sprintf('Error: The following folder does not exist:\n%s', filesDir);
        uiwait(warndlg(errorMessage));
        return;
    end
end