# Biclustering-SSVD
Implementation of the sparse singular value decomposition algorithm (Lee et al., 2010) to be used for Biclustering; Duke STA663 Final Project

The `Sparse SVD Optimized` notebook contains the final, optimized version of the algorithm from Lee et al., 2010, whose code is publically available in matlab and R (see `ssvd.R`, `ssvd.m`, and `thresh.m`). This SSVD algorithm is meant to be used for biclustering applications and analysis. The `Sparse SVD First Pass` notebook is a direct portiing of the code from matlab to python, but has not been optimized, and such runs much slower. The First Pass algorithm runs at approximately 18 minutes, whereas the optimized version run at approximately 40 seconds. `Data.txt` is the application example dataset (the genes example) provided by the authors of the paper. The paper, Lee et al., 2010, can be found at `ssvd.pdf`, and the supplemental to the paper can be found at `LeeShenHuangMarron09-sup.pdf`. 

# Installation Instructions

1) Either git clone this repo, or 
2) Execute the following in terminal `pip install Biclustering-SSVD`
