data{
  int<lower=1          > n_s;                         // number of subjects
  int<lower=1          > n_t         [n_s+1] ;        // number of trials per subject, plus total number of trials
  int<lower=1,upper=5  > stimuli [n_t[n_s+1]];        // 
  int<lower=1,upper=2  > correct [n_t[n_s+1]];        // correct (=1, yes-correct)? trial n_t
  int<lower=0,upper=1  > reward  [n_t[n_s+1]];        // reward (=1, yes)? trial n_t
  int<lower=1,upper=n_s> subject [n_t[n_s+1]];        // subject number
  int<lower=0,upper=1  > init    [n_t[n_s+1]];        // is this first trial of a subject? Should RL be initialized?
}

parameters{
  // Parameter means over whole group
	real group_mean_beta_pr; 				                  // Inverse temperature parameter
	real group_mean_alpha_gain_pr;                        // Sensitivity to gains
	real group_mean_alpha_loss_pr;                        // Sensitivity to losses
 
  // Parameter standard deviations over whole group
   real<lower=0> group_sdev_beta_pr;
	real<lower=0> group_sdev_alpha_gain_pr;
	real<lower=0> group_sdev_alpha_loss_pr;
  
  // Parameters for individuals
   real indiv_beta_pr      [n_s];
	real indiv_alpha_gain_pr[n_s];
	real indiv_alpha_loss_pr[n_s];
}
	
transformed parameters{
  // Parameter means over whole group
   real<lower=0,upper=100> group_mean_beta;
   real<lower=0,upper=1  > group_mean_alpha_gain;
   real<lower=0,upper=1  > group_mean_alpha_loss;

  // Parameter standard deviations over whole group
   real<lower=0,upper=100> indiv_beta      [n_s]; 
   real<lower=0,upper=1  > indiv_alpha_gain[n_s];
   real<lower=0,upper=1  > indiv_alpha_loss[n_s];

  // Parameters for individuals (probit)
   group_mean_beta       <- Phi(group_mean_beta_pr      )*100;
   group_mean_alpha_gain <- Phi(group_mean_alpha_gain_pr);
   group_mean_alpha_loss <- Phi(group_mean_alpha_loss_pr);

  // Parameters for individuals (probit)
   for (subj in 1:n_s){
      indiv_beta      [subj] <- Phi(indiv_beta_pr      [subj])*100;
      indiv_alpha_gain[subj] <- Phi(indiv_alpha_gain_pr[subj]);
      indiv_alpha_loss[subj] <- Phi(indiv_alpha_loss_pr[subj]);
   }
}
	
model{
   // Define vars needed for subject loop
   int       sid;                                     // subject ID
   real      action_vals_set [3,2];                   // A/B, C/D, E/F choice probabilities.
   real      action_vals_init[3,2];                   // Initial probabilities.
   real      action_vals_cur [2];                     // Choice probabilities for this set, e.g. C/D
   real      epsilon;                                 // 
   int       success;                                 //
   real      alpha;                                   //
   vector[2] pchoice;                                 //
   real      threshold;                               //
   int       index;                                   //
   int       cur_stim;                                //
   real      cur_diff;                                //

   // Set prior on group level mean parameters
   group_mean_beta_pr       ~ normal(0,1);
   group_mean_alpha_gain_pr ~ normal(0,1);
   group_mean_alpha_loss_pr ~ normal(0,1);

   // Set prior on group level standard deviations
   group_sdev_beta_pr       ~ uniform(0,1.5);
   group_sdev_alpha_gain_pr ~ uniform(0,1.5);
   group_sdev_alpha_loss_pr ~ uniform(0,1.5);

   // Set prior for individual level parameters
   for (subj in 1:n_s){
      indiv_beta_pr      [subj] ~ normal(group_mean_beta_pr      , group_sdev_beta_pr      );
      indiv_alpha_gain_pr[subj] ~ normal(group_mean_alpha_gain_pr, group_sdev_alpha_gain_pr);
      indiv_alpha_loss_pr[subj] ~ normal(group_mean_alpha_loss_pr, group_sdev_alpha_loss_pr);
   }

   // What is this parameter for?
   epsilon <- 0.00001; 

  // now start looping over subjects
   for (trial in 1:n_t[n_s+1]){

      // set initial values (i.e. values for trial 1) for the subject
      if (init[trial] == 1){
         sid <- subject[trial];                                   // Alias the subject id
         for (row in 1:3) {
            for (col in 1:2) {
               action_vals_set [row,col] <- 0.5;
               action_vals_init[row,col] <- 0.5;
            }
         }
         pchoice[1] <- 0.5;                                       // Prob. of picking choice 1 on trial 1
         pchoice[2] <- 0.5;                                       // Prob. of picking choice 2 on trial 1
      }

      cur_stim <- stimuli[trial];
      
      action_vals_cur[1] <- action_vals_set[cur_stim,1];
      action_vals_cur[2] <- action_vals_set[cur_stim,2];

      cur_diff   <- action_vals_cur[1] - action_vals_cur[2];
      threshold  <- indiv_beta[sid]*(cur_diff);
      
      pchoice[1] <- 1 / ( 1 + exp(threshold) );                   // Probability of picking choice 1
      pchoice[2] <- 1 - pchoice[1];                               // Probability of picking choice 2

      pchoice[1] <- epsilon/2+(1-epsilon)*pchoice[1];             //
      pchoice[2] <- epsilon/2+(1-epsilon)*pchoice[2];             //

      correct[trial] ~ categorical(pchoice);
      success <- correct[trial];                                 // Success of choice, boolean	
      index   <- success;

      // Reinforcement
      alpha <- reward[trial] *indiv_alpha_gain[sid] + (1-reward[trial]) *indiv_alpha_loss[sid];
      action_vals_set[cur_stim,index] <- action_vals_set[cur_stim,index] + alpha *(reward[trial] - action_vals_set[cur_stim,index]);
   }
}