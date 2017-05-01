## Implementation of the sparse SVD
#
#	work with thresh.R 
#
#  	Input variables:
#       X - argument (n x d matrix)
#       threu = type of penalty (thresholding rule) for the left 
#       	    singular vector,
#			1 = (Adaptive) LASSO (default)
#                 2 = hard thresholding
#                 3 = SCAD
#
#       threv = type of penalty (thresholding rule) for the right 
#               singular vector,
#                 1 = (Adaptive) LASSO (default)
#                 2 = hard thresholding
#                 3 = SCAD
#
#       gamu = weight parameter in Adaptive LASSO for the 
#              left singular vector, nonnegative constant (default = 0, LASSO)
#
#       gamv = weight parameter in Adaptive LASSO for the  
#              right singular vector, nonnegative constant (default = 0, LASSO)
#
#       u0,  v0 = initial values of left/right singular vectors                
#                 (default = the classical SVs)
# 
#       merr = threshold to decide convergence (default = 10^(-4))
#
#       niter = maximum number of iterations (default = 100)
#
#   Output: 
#       u = left sparse singular vector
#       v = right sparse singaulr vector
#       iter = number of iterations to achieve the convergence


ssvd = function(X,threu = 1, threv = 1, gamu = 2, gamv =2,  u0 = svd(X)$u[,1], v0 = svd(X)$v[,1], merr = 10^(-4), niter = 100){
    n = dim(X)[1]
    d = dim(X)[2]

    ud = 1;
    vd = 1;
    iter = 0;
    SST = sum(X^2);

    while (ud > merr || vd > merr) {
        iter = iter+1;
	

        # Updating v
        z =  t(X)%*% u0; 
        winv = abs(z)^gamv;
        sigsq = abs(SST - sum(z^2))/(n*d-d);
        
    	  cand = z*winv;
	  delt = sort(c(0,abs(cand)))
	  delt.uniq = unique(delt)
        Bv = rep(1,length(delt.uniq)-1)*Inf;
  
	  #ind = which(winv!=0)
	  ind = which(winv>10^(-8))
	  cand1 = cand[ind]
        winv1 = winv[ind]
        for (i in 1:length(Bv)){
            temp2 = thresh(cand1, type = 1, delta = delt.uniq[i]);
		temp2 = temp2/winv1;
            temp3 = rep(0,d);
            temp3[ind] = temp2
            Bv[i] = sum((X - u0%*%t(temp3))^2)/sigsq + sum(temp2!=0)*log(n*d)
	 }
    
        Iv = min(which(Bv==min(Bv)))
        th = delt.uniq[Iv]
        temp2 = thresh(cand1,type = 1, delta = th);
	  temp2 = temp2/winv1;
	  v1 = rep(0,d)
        v1[ind] = temp2;
        v1 = v1/sqrt(sum(v1^2)) #v_new
	
        # Updating u
        z = X%*%v1;
        winu = abs(z)^gamu;
        sigsq = abs(SST - sum(z^2))/(n*d-n);
    	  
	  cand = z*winu;
	  delt =sort(c(0,abs(cand)))
	  delt.uniq = unique(delt)
        Bu = rep(1,length(delt.uniq)-1)*Inf;
         
       #ind = which(winu!=0)
	 ind = which(winu > 10^(-8))

	 cand1 = cand[ind]
	 winu1 = winu[ind]
  
	 for (i in 1:length(Bu)){
            temp2 = thresh(cand1, type = threu, delta = delt.uniq[i]);
		temp2 = temp2/winu1;
		temp3 = rep(0,n);
		temp3[ind] = temp2;
		Bu[i] = sum((X - temp3%*%t(v1))^2)/sigsq + sum(temp2!=0)*log(n*d)
        }
        
        Iu = min(which(Bu==min(Bu)));
	  th = delt.uniq[Iu];
	  temp2 = thresh(cand1,type = 1, delta = th);
	  temp2 = temp2/winu1;
	  u1 = rep(0,n);
        u1[ind] =  temp2;
        u1 = u1/sqrt(sum(u1^2));

    
        ud = sqrt(sum((u0-u1)^2));
        vd = sqrt(sum((v0-v1)^2));

        if (iter > niter){
        print("Fail to converge! Increase the niter!")
	  break
        }
        
	u0 = u1;
	v0 = v1;
    }

return(list(u = u1, v = v1, iter = iter))
}
