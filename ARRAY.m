function [PD,PF,Precision,F1,AUC,Accuracy,G_measure,MCC,Balance, probPos] = ARRAY(source, target, LOC)
%ARRAY Detailed explanation goes here
% INPUTS:
%   (1) source - a n1*(d+1) matrix, the last column is the label where 0/1
%   denotes the non-defective/defective module
%   (2) target - a n2*(d+1) matrix, the last column is the label where 0/1
%   denotes the non-defective/defective module 
%   (3) LOC - LOC metric
% OUTPUTS:
%   Performance measures

if ~all(unique(source(:,end))==[0,1]') || ~all(unique(target(:,end))==[0,1]')
    error('The label of each sample must be 0 or 1');
end

defRatio = sum(source(:,end)==1)/size(source, 1); % Calculate the defective ratio of source dataset

sourceCopy = source;
if defRatio > 0.45 || size(source,1)<90
    rand('seed',0);
    
    [F1,AUC,MCCprobPos] = ManualDown(target, LOC);
    
    return
end

[F1, AUC, MCC, probPos] = DFWTNB(source, target);
end

