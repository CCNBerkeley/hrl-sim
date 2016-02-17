function [] = test_STAN_flat_RL_recovery()
% [] = test_STAN_flat_RL_recovery simulates human experimental data for a stacked one-step 
% selection paradigm, i.e. presentation of AB, CD, EF pairs for which each choice (e.g. A in AB)
% has a fixed probability of resulting in a reward, and then uses STAN to recover the parameters
% governing the simulated agents' choices.

%% Simulate Human Data for Experimental Paradigm 
% Stimuli (i.e. AB, CD, EF pairs) are coded 1,3, or 5, with selection reward probabilities coded
% serially in the variable "reward_prob".
%
% Notes:
%    1. Indexing is all crazy
%    2. This whole thing can probably be vectorized

alpha_gain = .2;                                            % Importance of gains to agents
alpha_loss = .1;                                            % Importance of losses to agents

ns    = 24;                                                 % Number of Subjects
betas = sort(7 + randn(1,25));                              % Inverse temp. param ie sensitivity 
                                                            % ... to Reward Prediction Error
trial_counts = repmat([150 300],1,ns/2);                    % Number of trials each subject
ntrials      = sum(trial_counts);                           % ... sees, either 150 or 300
trial_counts = [trial_counts ntrials];                      % Save total for whole-group analy.

stims = repmat([1 3 5],1,50);                               % Stimulus presented for each
stims = stims(randperm(length(stims)));                     % ... trial (randomized)
stims = repmat(stims,2*ns);                                 % Resultant size 2*ns-by-300*ns

reward_prob = [.8 .2 .7 .3 .6 .4];                          % Reward prob.s for A/B, C/D, E/F
smfn = @(x) 1./(1 + exp(x));                                % SoftMax FuNction for squashing
                                                            % ... RPE response
% Pre-initialize large matrices
outcomes = NaN(1,ntrials);                                  % Outcomes for simulated choices
rewards  = NaN(1,ntrials);                                  % Whether each outcome is rewarded
subj_ids = NaN(1,ntrials);                                  % The id. number of each subject
inits    = NaN(1,ntrials);                                  % Boolean, initialize learning?

% Simulate Experiment
lin_ind = 0;                                                % Linear index for vectors as lists
for subj_ind = 1:ns                                         % of vars by trial, e.g. "choices"
   % Initialize prob.s for current agent's choices
   choice_Ps = 0.5 * ones(1,6);                             % Choice probabilities

   % Simulate this subject's experimental data
   subj_ntrials = trial_counts(subj_ind);                   % Extract subject's epoch length
   for trial = 1:subj_ntrials                               % Loop over this subjects trials.
      lin_ind  = lin_ind + 1;                               % Increment loop counter
      
      cur_stim  = stims(trial);                             % Current stimulus
      stim_inds = cur_stim + [0, 1];                        % Index of choice reward probs.
      
      pair     = choice_Ps(stim_inds);                      % Prob. of eg picking A in AB choice
      pre_sqsh = betas(subj_ind)*(pair(1) - pair(2));       % Pre-squashed RPE response
      
      softmax  = smfn(pre_sqsh);                            % Squashed RPE response
      success  = 1+(rand < softmax);                        % Correctness as 1 or 2. 1=Corr.
      
      rwrd_ind = cur_stim + success - 1;                    % Eg. stim 3 correct => 3, inc. => 4
      reward   = rand < reward_prob(rwrd_ind);              % Boolean indication of reward 
      LR       = alpha_gain*reward + alpha_loss*(~reward);  % Calculate the learning rate

      choice_Ps(rwrd_ind) =      LR *reward  ...            % Update choice probabilities
                          + (1 - LR)*choice_Ps(rwrd_ind);   % based on outcome of this choice.

      outcomes(lin_ind) = success;                          % Record simulated agent success
      rewards (lin_ind) = reward;                           % Record feedback given to agent
      subj_ids(lin_ind) = subj_ind;                         % Record the agent's ID number
      inits   (lin_ind) = trial == 1;                       % Record if init. must happen
   end
   
   % Display the simulated agent's learned probabilities of making various selections. 
   disp( ['Choice Probs: ' sprintf('%5.3f   ',choice_Ps)] )
end

%% Save Stan Input
%
% STAN data interface defined as follows:
% data{
%    int<lower=1> n_s;                       // number of subjects
%    int<lower=1> n_t;                       // number of trials for that subject
%    int<lower=1,upper=5> Choice[n_t ,n_s];  // choice options 
%    int<lower=1,upper=2> Correct[n_t,n_s];  // correct (=1 yes-correct, =2 no-incorrect)
%    int<lower=0,upper=1> Reward[n_t,n_s];   // reward  (=0 no reward  , =1 yes rewarded)
% }// end data

RL_data = struct('n_s',ns,'n_t',trial_counts,'Choice',stims,'Correct',outcomes,...
                 'Reward',rewards,'Subject',subj_ids,'Init',inits);

%% Perform the STAN fitting
tic
fitRL = stan('file','./opzet_stan_kort_anne_2.stan','data',RL_data,'iter',1000,...
             'chains',4,'refresh',100,'warmup',500,'thin',10);
fitRL.verbose = false;
%fitRL.check();% show progress
fitRL.block();% block further instructions

toc
save fitRL2 fitRL

%% Show User Fit Outcomes
print(fitRL)
fitRL.traceplot

%% Extract Parameter Fits and Plot Stuff

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
