function [] = test_stan_flat_rl_recovery()
% [] = test_stan_flat_rl_recovery simulates human experimental data for a stacked one-step 
% selection paradigm, i.e. presentation of AB, CD, EF pairs for which each choice (e.g. A in AB)
% has a fixed probability of resulting in a reward, and then uses STAN to recover the parameters
% governing the simulated agents' choices.

%% Simulate Human Data for Experimental Paradigm 
% Stimuli (i.e. AB, CD, EF pairs) are coded 1,2, or 3.

alpha_gain = .2;                                            % Importance of gains to agents
alpha_loss = .1;                                            % Importance of losses to agents

ns    = 2;                                                  % Number of Subjects
betas = sort(7 + randn(1,ns+1));                            % Inverse temp. param ie sensitivity 
                                                            % ... to Reward Prediction Error
trial_counts = repmat([150 300],1,ns/2);                    % Number of trials each subject
ntrials      = sum(trial_counts);                           % ... sees, either 150 or 300
trial_counts = [trial_counts ntrials];                      % Save total for whole-group analy.

stims = repmat([1 2 3],1,50);                               % Stimulus presented for each
stims = stims(randperm(length(stims)));                     % ... trial (randomized)
stims = repmat(stims,2*ns);                                 % Resultant size 2*ns-by-300*ns

reward_prob = [0.8, 0.2;                                    % Reward prob.s. Row 1 is for AB
               0.7, 0.3;                                    % probs for CD
               0.6, 0.4 ];                                  % probs for EF

smfn = @(x) 1./(1 + exp(x));                                % SoftMax FuNction for squashing ...
                                                            % RPE response, which 
% Pre-initialize large matrices
stanin_correct  = NaN(1,ntrials);                           % Outcomes for simulated choices
stanin_rewards  = NaN(1,ntrials);                           % Whether each outcome is rewarded
stanin_subj_ids = NaN(1,ntrials);                           % The id. number of each subject
stanin_inits    = NaN(1,ntrials);                           % Boolean, initialize learning?
stanin_stimuli  = NaN(1,ntrials);                           % Vector of stimulus presentations

% Simulate Experiment
lin_ind = 0;                                                % Linear index for vectors as lists
for subj_ind = 1:ns                                         % of vars by trial, e.g. "choices"

   % Simulate this subject's experimental data
   action_vals  = 0.5 * ones(3,2);                          % Initialize action values
   subj_ntrials = trial_counts(subj_ind);                   % Extract subject's epoch length

   for trial = 1:subj_ntrials                               % Loop over this subjects trials.
      lin_ind  = lin_ind + 1;                               % Increment loop counter
      
      cur_stim = stims(trial);                              % Current stimulus
      cur_vals = action_vals(cur_stim,:);                   % Extract the learned action values
      cur_diff = cur_vals(1) - cur_vals(2);                 %
      
      threshold = smfn(betas(subj_ind)*cur_diff);           % Squashed RPE response
      success   = rand > threshold;                         % Successfully chose option 1?
       
      rwrd_ind = success + 1;                               % Column index in reward prob. matrix
      reward   = rand < reward_prob(cur_stim,rwrd_ind);     % Boolean indication of reward 
      alpha    = alpha_gain*reward + alpha_loss*(~reward);  % Calculate the learning rate

      new_action_vals =      alpha *reward  ...             % Update action values based on the
                      + (1 - alpha)*cur_vals(rwrd_ind);     % outcome of this choice.

      action_vals(cur_stim,rwrd_ind) = new_action_vals;     % 
      
      % Record calculations as vectors, these go to STAN.
      stanin_correct (lin_ind) = success + 1;               %
      stanin_rewards (lin_ind) = reward;                    %
      stanin_subj_ids(lin_ind) = subj_ind;                  %
      stanin_inits   (lin_ind) = trial == 1;                %
      stanin_stimuli (lin_ind) = cur_stim;                  %
   end
   
   % Display the simulated agent's learned probabilities of making various selections. 
   disp( ['Action Vals: ' sprintf('%5.3f   ',reshape(action_vals',1,6))] )
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
RL_data = struct('n_s'    ,ns              ...
                ,'n_t'    ,trial_counts    ...
                ,'stimuli',stanin_stimuli  ...
                ,'correct',stanin_correct  ...
                ,'reward' ,stanin_rewards  ...
                ,'subject',stanin_subj_ids ...
                ,'init'   ,stanin_inits    );
              
save('./output/sim_data.mat')             

%% Perform the STAN fitting
tic
fitRL = stan('file'       ,'./flat_rl_recovery.stan' ...
            ,'data'       ,RL_data ...
            ,'iter'       ,10000   ...
            ,'chains'     ,4       ...
            ,'refresh'    ,100     ...
            ,'warmup'     ,1000    ...
            ,'thin'       ,10      ...
            ,'working_dir','./output');

fitRL.verbose = false;
%fitRL.check();% show progress
fitRL.block();% block further instructions
toc

save('./output/fitRL.mat','fitRL')

plot_stan_flat_rl_recovery(fitRL,0)
end
