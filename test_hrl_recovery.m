function [] = test_hrl_recovery()
% [] = test_stan_flat_rl_recovery simulates human experimental data for a stacked one-step 
% selection paradigm, i.e. presentation of AB, CD, EF pairs for which each choice (e.g. A in AB)
% has a fixed probability of resulting in a reward, and then uses STAN to recover the parameters
% governing the simulated agents' choices.

%% Simulate Human Data for Experimental Paradigm 
% Stimuli (i.e. AB, CD, EF pairs) are coded 1,2, or 3.

mbwgt       = 0.5;                                          % Fraction model-based learning
alpha1_gain = 0.2;                                          % Importance of gains to agents
alpha2_gain = 0.2;                                          % Importance of gains to agents
alpha1_loss = 0.1;                                          % Importance of losses to agents
alpha2_loss = 0.1;                                          % Importance of losses to agents
fwrd_lr     = 0.2;                                          % Forward learning rate
lambda      = 0.2;                                          %

ns    = 2;                                                  % Number of Subjects
betas = sort(7 + randn(1,ns+1));                            % Inverse temp. param ie sensitivity 
                                                            % ... to Reward Prediction Error
trial_counts = repmat([150 300],1,ns/2);                    % Number of trials each subject
ntrials      = sum(trial_counts);                           % ... sees, either 150 or 300
trial_counts = [trial_counts ntrials];                      % Save total for whole-group analy.

stims = repmat([1 2 3],1,50);                               % Stimulus presented for each
stims = stims(randperm(length(stims)));                     % ... trial (randomized)
stims = repmat(stims,2*ns);                                 % Resultant size 2*ns-by-300*ns

trans_prob  = [0.8, 0.2;                                    % Transition probabilities, for 
               0.7, 0.3;                                    % moving from AB, CD, and EF to 
               0.6, 0.4 ];                                  % XY (Good) and ZW (Less Good)

trans_learn = repmat(0.5,3,2,2);                            % Transition probs learned, as matrix
                                                            % T(prev_state, action, cur_state)

reward_prob = [0.8, 0.2;                                    % Reward probs for second stages,
               0.6, 0.4];                                   % row 1 is XY, row 2 is ZY.

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
   mbavals{1} = 0.5 * ones(size(trans_prob));               % Step 1 model-based action values
   mbavals{2} = 0.5 * ones(size(reward_prob));              % Step 2 ...

   mfavals{1} = 0.5 * ones(size(trans_prob));               % Model-free action values
   mfavals{2} = 0.5 * ones(size(reward_prob));              % ...
   
   havals{1} = mbwgt*mbavals{1} + (1-mbwgt)*mfavals{1};     % Mixture (hybrid) model values
   havals{2} = mbwgt*mbavals{2} + (1-mbwgt)*mfavals{2};     % ...

   subj_ntrials   = trial_counts(subj_ind);                 % Extract subject's epoch length
   for trial = 1:subj_ntrials                               % Loop over this subjects trials.
      
      %---- Stage 1 stimulus and choice. ----%
      lin_ind  = lin_ind + 1;                               % Increment loop counter
      stim1    = stims(trial);                              % Current stimulus, a volatile var.
      
      cur_mbavals{1} = mbavals{1}(stim1,:);                 % How much is an MBA worth...?
      cur_mfavals{1} = mfavals{1}(stim1,:);                 % Compared to a masters of fine arts?
      cur_havals {1} =    mbwgt *cur_mbavals{1} ...         %
                     + (1-mbwgt)*cur_mfavals{1};            %
      
      cur_diff  = cur_havals{1}(1) - cur_havals{1}(2);      %
      threshold = smfn(betas(subj_ind)*cur_diff);           %
      success1  = rand > threshold;                         % Successfully chose option 1?
      act1      = 2 - success1;                             % Chose opt. 1 if smart, otherwise 2.

      % Record stage 1 info as vecs for STAN
      stanin_correct (lin_ind,1) = success1 + 1;            %
      stanin_stimuli (lin_ind,1) = stim1;                   %
      
      %---- Stage 2 stimulus and choice ----%      
      stim2 = 1 + (rand > trans_prob(stim1,act1));          % Do we move to XY (1) or ZW (2)?
      
      delta_spe = 1 - trans_learn(stim1,act1,stim2);        % Transition pred. update parameter.
      trans_learn(stim1,act1,stim2) = ...                   % Update the learned transition rate
      trans_learn(stim1,act1,stim2) + fwrd_lr *delta_spe;   % matrix.
      
      pred = pred_state(stim1,act1,trans_learn);            % Predicted state, i.e. S(s1t,a1t)

      cur_mbavals{2} = mbavals{2}(stim2,:);                 %
      cur_mfavals{2} = mfavals{2}(stim2,:);                 %
      cur_havals {2} =    mbwgt *cur_mbavals{2} ...         %
                     + (1-mbwgt)*cur_mfavals{2};            %
      
      cur_diff  = cur_havals{2}(1) - cur_havals{2}(2);      %
      threshold = smfn(betas(subj_ind)*cur_diff);           %
      success2  = rand > threshold;                         % Successfully chose option 1?
      act2      = 2 - success2;                             % Chose opt. 1 if smart, otherwise 2.
      reward    = rand < reward_prob(stim2,act2);           % Boolean indication of reward 

      % Record stage 2 info as vecs for STAN
      stanin_correct (lin_ind,2) = success2 + 1;            %
      stanin_stimuli (lin_ind,2) = stim2;                   %
      
      %---- Action-value Updates ----%
      alpha1 = alpha1_gain*reward + alpha1_loss*(~reward);  % Calculate the stage 1 learning rate
      alpha2 = alpha2_gain*reward + alpha2_loss*(~reward);  % Calculate the stage 2 learning rate

      delta1 = mfavals{2}(stim2,act2) - mfavals{1}(stim1,act1);
      delta2 = reward                 - mfavals{2}(stim2,act2);

      mfavals{2}(stim2,act2) = mfavals{2}(stim2,act2) + alpha2*delta2;
      mfavals{1}(stim1,act1) = mfavals{1}(stim1,act1) + alpha1*delta1 ...
                                                      + lambda*alpha2*delta2;
      
      mbavals{2}(stim2,act2) = mbavals{2}(stim2,act2) + alpha2*delta2;
      mbavals{1}(stim1,act1) = max(mbavals{2}(pred,:));

      havals{1} = mbwgt*mbavals{1} + (1-mbwgt)*mfavals{1};  %
      havals{2} = mbwgt*mbavals{2} + (1-mbwgt)*mfavals{2};  %
 
      % Record general trial info as vecs for STAN
      stanin_rewards (lin_ind) = reward;                    %
      stanin_subj_ids(lin_ind) = subj_ind;                  %
      stanin_inits   (lin_ind) = trial == 1;                %
   end
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

plot_frl_recovery(fitRL,0)
end

function [prediction] = pred_state(cur_state,action,transitions)
   % Returns 1 or 2 for getting to XY vs ZW respectively.
   
   % Get the greatest expected transition value
   most_likely = max(transitions(cur_state,action,:));
   
   % 
   prediction = find(transitions(cur_state,action,:) == most_likely);
   if numel(prediction) > 1
      prediction = prediction((rand > 0.5)+1);
   end
   % Insert handling for equal transition probabilities?
   
end
