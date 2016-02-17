data{
  int<lower=1          > n_s;                         // number of subjects
  int<lower=1          > n_t         [n_s+1] ;        // number of trials per subject, plus total number of trials
  int<lower=1,upper=5  > Choice  [n_t[n_s+1]];        // choice options trial n_t (choice is 1,3 of 5). (All subjects stacked)
  int<lower=1,upper=2  > Correct [n_t[n_s+1]];        // correct (=1, yes-correct)? trial n_t
  int<lower=0,upper=1  > Reward  [n_t[n_s+1]];        // reward (=1, yes)? trial n_t
  int<lower=1,upper=n_s> Subject [n_t[n_s+1]];        // subject number
  int<lower=0,upper=1  > Init    [n_t[n_s+1]];        // is this first trial of a subject? Should RL be initialized?
}

parameters{
  // Parameter means over whole group
	real group_mean_beta; 				                  // Inverse temperature parameter
	real group_mean_alpha_gain;                        // Sensitivity to gains
	real group_mean_alpha_loss;                        // Sensitivity to losses
 
  // Parameter standard deviations over whole group
   real<lower=0> group_sdev_beta;
	real<lower=0> group_sdev_alpha_gain;
	real<lower=0> group_sdev_alpha_loss;
  
  // Parameters for individuals
   real indiv_beta      [n_s];
	real indiv_alpha_gain[n_s];
	real indiv_alpha_loss[n_s];
}
	
transformed parameters{
  // Parameter means over whole group
   real<lower=0,upper=100> group_mean_beta_tr;
   real<lower=0,upper=1  > group_mean_alpha_gain_tr;
   real<lower=0,upper=1  > group_mean_alpha_loss_tr;

  // Parameter standard deviations over whole group
   real<lower=0,upper=100> indiv_beta_tr      [n_s]; 
   real<lower=0,upper=1  > indiv_alpha_gain_tr[n_s];
   real<lower=0,upper=1  > indiv_alpha_loss_tr[n_s];

  // Parameters for individuals (probit)
   group_mean_beta_tr       <- Phi(group_mean_beta )*100;
   group_mean_alpha_gain_tr <- Phi(group_mean_alpha_gain);
   group_mean_alpha_loss_tr <- Phi(group_mean_alpha_loss);

  // Parameters for individuals (probit)
   for (subj in 1:n_s){
      indiv_beta_tr      [subj] <- Phi(indiv_beta      [subj])*100;
      indiv_alpha_gain_tr[subj] <- Phi(indiv_alpha_gain[subj]);
      indiv_alpha_loss_tr[subj] <- Phi(indiv_alpha_loss[subj]);
   }
}
	
model{
   // Define vars needed for subject loop
   int       sid;                                     // Subject ID
   real      choice_probs_set [6];                    // A/B, C/D, E/F choice probabilities.
   real      choice_probs_init[6];                    // Initial probabilities.
   real      choice_probs_cur [2];                    // Choice probabilities for this set, e.g. C/D
   real      epsilon;                                 // 
   int       success;                                 //
   real      alpha;                                   //
   vector[2] pchoice;                                 //

   // Set prior on group level mean parameters
   group_mean_beta       ~ normal(0,1);
   group_mean_alpha_gain ~ normal(0,1);
   group_mean_alpha_loss ~ normal(0,1);

   // Set prior on group level standard deviations
   group_sdev_beta       ~ uniform(0,1.5);
   group_sdev_alpha_gain ~ uniform(0,1.5);
   group_sdev_alpha_loss ~ uniform(0,1.5);

   // Set prior for individual level parameters
   for (subj in 1:n_s){
      indiv_beta      [subj] ~ normal(group_mean_beta      , group_sdev_beta      );
      indiv_alpha_gain[subj] ~ normal(group_mean_alpha_gain, group_sdev_alpha_gain);
      indiv_alpha_loss[subj] ~ normal(group_mean_alpha_loss, group_sdev_alpha_loss);
   }

   // Define epsilon (?) why this value
   epsilon <- 0.00001;

  // now start looping over subjects
   for (trial in 1:n_t[n_s+1]){

      // set initial values (i.e. values for trial 1) for the subject
      if (Init[trial] == 1){
         si <- Subject[trial];                                    // Alias the subject id
         for (v in 1:6) {
            choice_probs     [v] <- 0.5;
            choice_probs_init[v] <- 0.5;
         }
         pchoice[1] <- 0.5;                                       // Prob. of picking choice 1 on trial 1
         pchoice[2] <- 0.5;                                       // Prob. of picking choice 2 on trial 1
      }

      choice_probs_cur[1] <- choice_probs[Choice[trial]  ];
      choice_probs_cur[2] <- choice_probs[Choice[trial]+1];

      pre_squash <- indiv_beta_tr[si]*(choice_probs_cur[2] - choice_probs_cur[1]);
      
      pchoice[1] <- 1 / ( 1 + exp(pre_squash) );                  // Probability of picking choice 1
      pchoice[2] <- 1 - pchoice[1];                               // Probability of picking choice 2

      pchoice[1] <- epsilon/2+(1-epsilon)*pchoice[1];             // (?)
      pchoice[2] <- epsilon/2+(1-epsilon)*pchoice[2];             // (?)

      Correct[trial] ~ categorical(pchoice);
      success <- Correct[trial] - 1;                              // Success of choice, boolean	
      index   <- Choice[trial] + success;

      // Reinforcement
      alpha <- Reward[trial] *indiv_alpha_gain_tr[si] + (1-Reward[trial]) *indiv_alpha_loss_tr[si];
      choice_probs[index] <- choice_probs[index] + alpha *(Reward[trial] - choice_probs[index]);
   }
}