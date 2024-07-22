function [F1, AUC, MCC, probPos] = ManualDown(target, LOC, thr)
% ManualDown - Summary of this function goes here: For a given target release, ManualDown considers a larger module as more
% defect-prone. Specifically, first rank the modules of target data descendly according to module size. Then, we classify the
% top x% modules in the ranking list as defective and the remaining modules as not defective. 
%   Detailed explanation goes here:
%   INPUTS:
%       (1) target - a n*(d+1) matrix where n and d separately denote the number of
%       samples and the number of features (i.e., metrics), the last column is the actual label;
%       (2) LOC - LOC metric in Target;
%       (3) thr (default value 0.5) - a classifcation threshold value belonging to (0,1);
%   OUTPUTS:
%       Perforamnce measures
% 

% Default value
if ~exist('thr', 'var')||isempty(thr)
    thr = 0.5; 
end

n = size(target, 1); % The number of target instances
%[~,idx] = sort(target(:,idxLOC),'descend'); % Sort target instances in descending order of LOC
[~,idx] = sort(LOC,'descend');
marker = floor(n * thr);
predLabel = zeros(n,1); 
for i=1:n 
    if i <= marker
        predLabel(i) = 1; % defective
    else
        predLabel(i) = 0; 
    end
end

[~,idxIdx] = sort(idx,'ascend');
predLabel = predLabel(idxIdx,:);

probPos = predLabel; % Label probability is needed to calculate AUC
[F1, AUC, MCC] = Performance(target(:,end), probPos);

end

