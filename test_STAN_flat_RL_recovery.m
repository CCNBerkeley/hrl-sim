function [] = test_stan_flat_rl_recovery()
% [] = test_stan_flat_rl_recovery simulates human experimental data for a stacked one-step 
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

stims = repmat([1 2 3],1,50);                               % Stimulus presented for each
stims = stims(randperm(length(stims)));                     % ... trial (randomized)
stims = repmat(stims,2*ns);                                 % Resultant size 2*ns-by-300*ns

reward_prob = [0.8,0.2; 0.7,0.3; 0.6,0.4];                  % Reward prob.s for A/B, C/D, E/F
smfn = @(x) 1./(1 + exp(x));                                % SoftMax FuNction for squashing
                                                            % ... RPE response
% Pre-initialize large matrices
outcomes = NaN(1,ntrials);                                  % Outcomes for simulated choices
rewards  = NaN(1,ntrials);                                  % Whether each outcome is rewarded
subj_ids = NaN(1,ntrials);                                  % The id. number of each subject
inits    = NaN(1,ntrials);                                  % Boolean, initialize learning?
choice   = NaN(1,ntrials);                                  % Vector of stimulus presentations

% Simulate Experiment
lin_ind = 0;                                                % Linear index for vectors as lists
for subj_ind = 1:ns                                         % of vars by trial, e.g. "choices"
   % Initialize prob.s for current agent's choices
   action_vals = 0.5 * ones(3,2);                             % Choice probabilities

   % Simulate this subject's experimental data
   subj_ntrials = trial_counts(subj_ind);                   % Extract subject's epoch length
   for trial = 1:subj_ntrials                               % Loop over this subjects trials.
      lin_ind  = lin_ind + 1;                               % Increment loop counter
      
      cur_stim = stims(trial);                              % Current stimulus
      cur_vals = action_vals(cur_stim,:);                   % Extract the learned action values
      cur_diff = cur_vals(1) - cur_vals(2);                 %
      
      threshold = smfn(betas(subj_ind)*cur_diff);           % Squashed RPE response
      success   = rand > threshold;                         % Correctness as boolean
       
      rwrd_ind = success + 1;                               % Col. index into reward prob. matrix
      reward   = rand < reward_prob(cur_stim,rwrd_ind);     % Boolean indication of reward 
      alpha    = alpha_gain*reward + alpha_loss*(~reward);  % Calculate the learning rate

      new_action_vals =      alpha *reward  ...             % Update action values based on the
                      + (1 - alpha)*cur_vals(rwrd_ind);     % outcome of this choice.

      action_vals(cur_stim,rwrd_ind) = new_action_vals;     % 
      
      outcomes(lin_ind) = success;                          % Record simulated agent success
      rewards (lin_ind) = reward;                           % Record feedback given to agent
      subj_ids(lin_ind) = subj_ind;                         % Record the agent's ID number
      inits   (lin_ind) = trial == 1;                       % Record if init. must happen
      choice  (lin_ind) = cur_stim;                         % Record the current stimulus
   end
   
   % Display the simulated agent's learned probabilities of making various selections. 
   disp( ['Choice Probs: ' sprintf('%5.3f   ',reshape(action_vals',1,6))] )
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

RL_data = struct('n_s',ns,'n_t',trial_counts,'Choice',choice,'Correct',outcomes,...
                 'Reward',rewards,'Subject',subj_ids,'Init',inits);

%% Perform the STAN fitting
tic
fitRL = stan('file','./flat_rl_recovery.stan','data',RL_data,'iter',1000,...
             'chains',4,'refresh',100,'warmup',500,'thin',10);
fitRL.verbose = false;
%fitRL.check();% show progress
fitRL.block();% block further instructions

toc
save('fitRL.mat',fitRL)

plot_stan_flat_rl_recovery(fitRL)
end
