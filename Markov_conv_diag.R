### Geweke convergence diagnostic for Markov processes.
### Run this code for different post-burn-in runs of the Markov process.
### Convergence to the invariant distribution will be reflected in a high p-value.
# install.packages("R.matlab")
# install.packages("coda")

rm(list=ls())
library(R.matlab)
library(coda)



sample_size = 1000000
num_traj = 64



mat <- readMat("stdist_draws_v6.mat")
traj_all <- as.vector(mat$stdist.draws)

geweke_vec = rep(NA, num_traj)
heidel_vec = rep(NA, num_traj)
for (traj_idx in 1:num_traj) {
  start <- (traj_idx-1)*sample_size+1
  end <- traj_idx*sample_size
  traj <- as.mcmc(traj_all[start:end])
  
  geweke_vec[traj_idx] = pnorm(geweke.diag(traj, frac1=0.499, frac2=0.499)$z)  # Results insensitive to fractions
  heidel_vec[traj_idx] = heidel.diag(traj)[3]
}

cat(mean(geweke_vec))  # p-value of Geweke test
cat(mean(heidel_vec))  # p-value of Heidelberger-Welch test