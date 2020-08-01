function [Gamma,Beta,Alpha,Garbage] = getabs_power(eeg,waveletFunction)
% Acknowledgements: Modified version of the code originally written by
% Hafeez Ullah Amin, Paper: Classification of EEG Signals Based on Pattern
% Recognition Approach

% Dataset 2a from BCI Competition IV is being used for this function; So
% the sampling rate of the input signal is 250Hz

% eeg: is the input EEG signal
% waveletFunction: is the mother wavelet, e.g. 'db4'
[C,L] = wavedec(eeg,4,waveletFunction); % Signal decomposition into approximation and detailed coefficients

% Calculation The coefficient vectors of every Band 
cD2 = detcoef(C,L,2); % Gamma: Coefficients of cD2 will contained 31.25-to-62.50Hz [cD1 will contained 62.60 to 125Hz, which is assumed as unwanted or noise]  
cD3 = detcoef(C,L,3); % Beta: Coefficients of cD3 will contained 15.62-to-31.25Hz 
cD4 = detcoef(C,L,4); % Alpha: Coefficients of cD4 will contained 7.81-to-15.62Hz 
cA4 = appcoef(C,L,waveletFunction,4); % Coefficients of cA5 will contained 0-to-7.81Hz

% Calculation of the Details Vectors of every band: 
D2 = wrcoef('d',C,L,waveletFunction,2); %Gamma 
D3 = wrcoef('d',C,L,waveletFunction,3); %Beta 
D4 = wrcoef('d',C,L,waveletFunction,4); %Alpha 
A4 = wrcoef('a',C,L,waveletFunction,4); %Garbage

% Gamma = D2; 
Gamma =(sum(D2.^2));

% Beta = D3; 
Beta=(sum(D3.^2));

% Alpha = D4; 
Alpha=(sum(D4.^2));

% Garbage = A4; 
Garbage = (sum(A4.^2));

end


