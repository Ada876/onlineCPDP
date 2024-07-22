function [F1, AUC, MCC] = Performance(actual_label, probPos, threshold)
%PERFORMANCE Summary of this function goes here
%   Detailed explanation goes here
% INPUTS:
%   (1) actual_label - The actual label, a column vetor, each row is an instance's class label;
%   (2) predict_label - The predicted label, a column vetor, each row is an instance label;
%   (3) threshold - A number in [0,1], by default threshold=0.5.
% OUTPUTS:
%   Performance measures

% Default value
if ~exist('threshold','var')||isempty(threshold)
    threshold = 0.5;
end

% if numel(unique(actual_label)) < 1
%     error('Please make sure that the true label ''actual_label'' must has at least two different kinds of values.');
% end

assert(numel(unique(actual_label)) > 1, 'Please ensure that ''actual_label'' includes two or more different labels.'); % 
assert(length(actual_label)==length(probPos), 'Two input parameters must have the same size.');


predict_label = double(probPos>=threshold);

cf=confusionmat(actual_label,predict_label);
TP=cf(2,2);
TN=cf(1,1);
FP=cf(1,2);
FN=cf(2,1);

Accuracy = (TP+TN)/(FP+FN+TP+TN);
PD=TP/(TP+FN);
PF=FP/(FP+TN);
Precision=TP/(TP+FP);
F1=2*Precision*PD/(Precision+PD);
[X,Y,T,AUC]=perfcurve(actual_label, probPos, '1');% 
MCC = (TP*TN-FP*FN)/(sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)));


end

