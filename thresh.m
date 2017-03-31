function y = thresh(X,para)
    %  To implement the soft, hard, SCAD thresholding rule
    % 
    % Input : 
    %   X    -  argument (a number or a vector )
    %
    %   para - a Matlab structure of input parameters
    %
    %       type: thresholding rule type
    %           1 = soft thresholding (default)
    %           2 = hard thresholding
    %           3 = SCAD
    %
    %       lambda: thresholding level (default = min(X)/2)
    %
    %       a: parameter for SCAD penalty (default = 3.7)
    %
    
    

% default values
type = 1;
lambda = min(X)/2;
a = 3.7; 

if nargin > 1; % then para has been added
    if isfield(para,'type');
        type = getfield(para, 'type');
    end
    
    if isfield(para, 'lambda');
        lambda = getfield(para, 'lambda');
    end
    
    if isfield(para, 'a');
        a = getfield(para, 'a');
    end
    
else
    para = struct('nothing',[]);
end

if type==1;
    y = sign(X).*(abs(X)>=lambda).*(abs(X)-lambda);
elseif type==2;
    y = X.*(abs(X)>lambda);
else y = sign(X).*(abs(X)>=lambda).*(abs(X)-lambda).*(abs(X)<=2*lambda)...
        +((a-1).*X-sign(X)*a*lambda)/(a-2).*(2*lambda<abs(X)).*(abs(X)<=a*lambda)...
        +X.*(abs(X)>a*lambda);
end

 