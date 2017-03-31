% Implementation of the sparse SVD
% Matlab elapsed time: 89.169 seconds

%readtable('data.txt')
%X = ans
function [u v iter] = ssvd(X,param)
%  Requirement: thresh.m
% 
%  Input variables:
%       X - argument (n x d matrix)
%       param - a Matlab structure of input parameters
%
%               threu = type of penalty (thresholding rule) for the left 
%                       singular vector,
%                       1 = (Adaptive) LASSO (default)
%                       2 = hard thresholding
%                       3 = SCAD
%
%               threv = type of penalty (thresholding rule) for the right 
%                       singular vector,
%                       1 = (Adaptive) LASSO (default)
%                       2 = hard thresholding
%                       3 = SCAD
%
%               gamu = weight parameter in Adaptive LASSO for the 
%                      left singular vector, 
%                      nonnegative constant (default = 0, LASSO)
%
%               gamv = weight parameter in Adaptive LASSO for the  
%                      right singular vector,
%                      nonnegative constant (default = 0, LASSO)
%
%               u0,  v0 = initial values of left/right singular vectors                
%                         (default = the classical SVs)
% 
%               merr = threshold to decide convergence (default = 10^(-4))
%
%               niter = maximum number of iterations (default = 100)
%
%   Output: 
%       u = left sparse singular vector
%       v = right sparse singaulr vector
%       iter = number of iterations to achieve the convergence


[n d]= size(X);

threu = 1;
threv = 1;

gamu = 0;
gamv = 0;

[t1 t2 t3] = svd(X,'econ');
u0 = t1(:,1);
v0 = t3(:,1);

merr = 10^(-4);
niter = 100;

if nargin > 1 ;
    if isfield(param,'gamu')    %  then change to input value
        gamu = getfield(param,'gamu');
    end
    
    if isfield(param,'gamv')    
        gamv = getfield(param,'gamv');
    end
      
    if isfield(param,'threu')    %  then change to input value
        threu = getfield(param,'threu');
    end
    
    if isfield(param,'threv')    %  then change to input value
        threv = getfield(param,'threv');
    end

    if isfield(param,'u0')    %  then change to input value
        u0 = getfield(param,'u0');
    end
    
    if isfield(param,'v0')    %  then change to input value
        v0 = getfield(param,'v0');
    end
    
    if isfield(param,'merr')    %  then change to input value
        merr = getfield(param,'merr');
    end
    
    if isfield(param,'niter')    %  then change to input value
        niter = getfield(param,'niter');
    end
end

ud = 1;
vd = 1;
iter = 0;

%Bv = zeros(d,1);
%Bu = zeros(n,1);

SST = sum(sum(X.^2));
while (ud > merr || vd > merr)
    iter = iter+1;
    %% Updating v
	z =  X'*u0;
	winv = abs(z).^gamv;
	sigsq = (SST - sum(z.^2))/(n*d-d);

	tv = sort([0; abs(z.*winv)]);
	rv = sum(tv>0);
	Bv = ones(d+1, 1)*Inf;
    for i = 1:rv;
        lvc  =  tv(d+1-i);
        temp1 = find(winv~=0);
        para = struct('type', threv, 'lambda',lvc./winv(temp1));
        temp2 = thresh(z(temp1), para);
		vc = zeros(d,1);
		vc(temp1) = temp2;
		Bv(i) = sum(sum((X - u0*vc').^2))/sigsq + i*log(n*d);
    end
    Iv = find(Bv==min(Bv), 1, 'first');
%    Iv = min(find(Bv==min(Bv)));
    temp = sort([0; abs(z.*winv)]);
    lv = temp(d+1-Iv);
    para = struct('lambda',lv./winv(temp1));
	temp2 = thresh(z(temp1),para);
	v1 = zeros(d,1);
	v1(temp1) = temp2;
	v1 = v1/sqrt(sum(v1.^2)); %v_new
	
	%% Updating u
	z = X*v1;
	winu = abs(z).^gamu;
	sigsq = (SST - sum(z.^2))/(n*d-n);

	tu = sort([0; abs(z.*winu)]);
	ru = sum(tu>0);
	Bu = ones(n+1, 1)*Inf;
    
    for i = 1:ru;
        luc  =  tu(n+1-i);
        temp1 = find(winu~=0);
        para = struct('type', threu,'lambda',luc./winu(temp1));
        temp2 = thresh(z(temp1), para);
		uc = zeros(n,1);
		uc(temp1) = temp2;
		Bu(i) = sum(sum((X - uc*v1').^2))/sigsq + i*log(n*d);
    end
    Iu = find(Bu==min(Bu), 1, 'first');
%    Iu = min(find(Bu==min(Bu)));
	temp = sort([0; abs(z.*winu)]);
    lu = temp(n+1-Iu);
    para = struct('lambda',lu./winu(temp1));
	temp2 = thresh(z(temp1),para);
	u1 = zeros(n,1);
	u1(temp1) = temp2;
	u1 = u1/sqrt(sum(u1.^2)); %u_new
    
    
	ud = sqrt(sum((u0-u1).^2));
    vd = sqrt(sum((v0-v1).^2));

    if iter > niter
        disp('Fail to converge! Increase the niter!')
        break
    end
    
	u0 = u1;
	v0 = v1;
end

u = u1;
v = v1;

