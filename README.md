# pisa
Probabilistic Inference-time Scaling Algorithms

Refer to blog for motivation and initial results: https://roshanwariar.github.io/2025/12/08/pita/

To run power sampling with autoregressive MCMC run power_sampling_mcmc.py. 

To run my modified version of "power" Sequential Monte Carlo (or stochastic beam search) then run power_sampling_smc.py

TLDR is that power sampling with MCMC can extend reasoning abilities beyond simply low/high temp sampling of base models, but takes forever. Sequential Monte Carlo over multiple chains with a "power coefficient" is much faster and achieves better intelligence. 

*put chart here*
