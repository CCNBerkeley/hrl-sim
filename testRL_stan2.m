function [] = testRL_stan2()

%% Simulate Presentation
alpha_gain = .2;                                            % Alpha gain
alpha_loss = .1;                                            % Alpha loss

ns    = 24;                                                 % Number of Subjects
betas = sort(7 + randn(1,25));                              % ?

nt = [repmat([150 300],1,ns/2)];                            % Number of trials each subject sees
nt = [nt sum(nt)];                                          % is either 150, 300

stims = repmat([1 3 5],1,50);                               % Stimulus presented, for each 
stims = stims(randperm(length(stims)));                     % ...
stims = repmat(stims,2*ns);

reward_prob = [.8 .2 .7 .3 .6 .4];                          % Probability of being rewarded 
smfn = @(x) 1/(1 + exp(x));                                 % Define softmax function

lin_ind = 0;                                                % Linear index for ns-by-nt matrix
for subj_ind = 1:ns                                         % 
   Q = 0.5 * ones(1,6);                                     % ?

   for trial = 1:nt(subj_ind)                               % 
      lin_ind  = lin_ind + 1;                               % Update counter
      cur_stim = stims(trial);                              % Current stimulus
      pair     = Q(cur_stim + [0 1]);                       % (?)
      softmax  = smfn(betas(subj_ind)*(pair(1) - pair(2))); %  
      success  = 1+(rand < softmax);                        % Correctness as a logical
      reward   = rand < reward_prob(cur_stim + (success - 1));     % Gets rewarded? (Boolean) 
      LR       = alpha_gain *     success ...               % Learning Rate (?)
               + alpha_loss *(1 - success);                 % Calculate the learning rate

      Q(cur_stim + (success - 1)) =      LR  * reward ...
                                  + (1 - LR) * Q(cur_stim + (success-1));

      choices  (lin_ind) = cur_stim;
      outcomes (lin_ind) = success;
      rewards  (lin_ind) = reward;
      subj_nums(lin_ind) = subj_ind;
      inits    (lin_ind) = trial == 1;
   end
   disp( ['Q: ' sprintf('%5.3f   ',Q)] )
end

%% Save Stan Input
% data{
%   int<lower=1> n_s;                               // number of subjects
%   int<lower=1> n_t;                               // number of trials for that subject
%   int<lower=1,upper=5> Choice[n_t ,n_s];          // choice options trial n_t (choice is 1,3 of 5) for subject n_s
%   int<lower=1,upper=2> Correct[n_t,n_s];          // correct (=1, yes-correct)? trial n_t for subject n_s
%   int<lower=0,upper=1> Reward[n_t,n_s];           // reward (=1, yes)? trial n_t for subject n_s
% }// end data

RL_data = struct('n_s',ns,'n_t',nt,'Choice',choices,'Correct',outcomes,...
                 'Reward',rewards,'Subject',subj_nums,'Init',inits);
%%
tic
fitRL = stan('file','./opzet_stan_kort_anne_2.stan','data',RL_data,'iter',1000,...
             'chains',4,'refresh',100,'warmup',500,'thin',10);
fitRL.verbose = false;
%fitRL.check();% show progress
fitRL.block();% block further instructions

toc
save fitRL2 fitRL
%%
print(fitRL)
fitRL.traceplot
%%
mu_ag = fitRL.extract('permuted',false).mu_ag;
mu_al = fitRL.extract('permuted',false).mu_al;
mu_b  = fitRL.extract('permuted',false).mu_b;

figure;
subplot(3,3,1);hist(mu_ag)
subplot(3,3,2);scatter(mu_ag,mu_al);lsline;[rho pval] = corr(mu_ag,mu_al);[rho pval]
subplot(3,3,3);scatter(mu_ag,mu_b);lsline;[rho pval] = corr(mu_ag,mu_b);[rho pval]
subplot(3,3,5);hist(mu_al)
subplot(3,3,6);scatter(mu_al,mu_b);lsline;[rho pval] = corr(mu_al,mu_b);[rho pval]
subplot(3,3,9);hist(mu_b)

subplot(3,3,7)
hist(mu_ag-mu_al)
[h p ci stats] = ttest(mu_ag-mu_al);
[p stats.tstat]

end
