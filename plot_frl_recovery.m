function [] = plot_stan_flat_rl_recovery(fitRL,show_traces)
%PLOT_STAN_FLAT_RL_RECOVERY Summary of this function goes here
%   Detailed explanation goes here

% Show User Fit Outcomes
print(fitRL)
if show_traces
   fitRL.traceplot
end

% Extract Parameter Fits
mu_ag = fitRL.extract('permuted',false).mu_ag;
mu_al = fitRL.extract('permuted',false).mu_al;
mu_b  = fitRL.extract('permuted',false).mu_b;

% Plot 
figure;
subplot(3,3,1);
   hist(mu_ag)

subplot(3,3,2); 
   scatter(mu_ag,mu_al);
   lsline;
   [rho, pval] = corr(mu_ag,mu_al);
   disp(['Rho, pval: ' sprintf('%5.3f   ',[rho pval])])

subplot(3,3,3);
   scatter(mu_ag,mu_b);
   lsline;
   [rho, pval] = corr(mu_ag,mu_b);
   disp(['Rho, pval: ' sprintf('%5.3f   ',[rho pval])])

subplot(3,3,5);
   hist(mu_al)
   
subplot(3,3,6);
   scatter(mu_al,mu_b);
   lsline;
   [rho, pval] = corr(mu_al,mu_b);
   disp(['Rho, pval: ' sprintf('%5.3f   ',[rho pval])])

subplot(3,3,9);
   hist(mu_b)

subplot(3,3,7)
   hist(mu_ag-mu_al)
   [~,p,~,stats] = ttest(mu_ag-mu_al);
   disp(['p  ,tstat: ' sprintf('%5.3f   ',[p stats.tstat])])


end

