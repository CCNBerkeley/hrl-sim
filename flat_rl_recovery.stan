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
	real mu_b_pr; 				                           //inverse gain parameter
	real mu_ag_pr;   				                        //alphaG
	real mu_al_pr;   				                        //alphaL
 
  // Parameter standard deviations over whole group
   real<lower=0> sd_b;   		  	                     //inverse gain parameter
	real<lower=0> sd_ag;   				                  //alphaG
	real<lower=0> sd_al;   				                  //alphaL
  
  // Parameters for individuals
   real b_ind_pr [n_s];   			                     //inverse gain parameter
	real ag_ind_pr[n_s];   				                  //alphaG
	real al_ind_pr[n_s];   				                  //alphaL
}
	
transformed parameters{
  // Parameter means over whole group
   real<lower=0,upper=100> mu_b; 		 		         //inverse gain parameter
   real<lower=0,upper=1  > mu_ag;   				      //alphaG
   real<lower=0,upper=1  > mu_al;   				      //alphaL

  // Parameter standard deviations over whole group
   real<lower=0,upper=100> b_ind [n_s];     		      //inverse gain parameter
   real<lower=0,upper=1  > ag_ind[n_s];   				//alphaG
   real<lower=0,upper=1  > al_ind[n_s];   	         //alphaL

  // Parameters for individuals (probit)
   mu_b  <- Phi(mu_b_pr )*100;   	                  //inverse gain parameter
   mu_ag <- Phi(mu_ag_pr);   				               //alphaG
   mu_al <- Phi(mu_al_pr);   				               //alphaL

  // Parameters for individuals (probit)
   for (subj in 1:n_s){
      b_ind [subj] <- Phi(b_ind_pr [subj])*100;
      ag_ind[subj] <- Phi(ag_ind_pr[subj]);
      al_ind[subj] <- Phi(al_ind_pr[subj]);
   }
}
	
model{
   // define vars needed for subject loop
   int       sid;                                     // Subject ID
   real      prQ0[6];                                 //
   real      prQ [6];                                 //
   real      Qchoice[2];                              //
   real      epsilon;                                 //
   int       success;                                 //
   real      alpha;                                   //
   vector[2] pchoice;                                 //

   // set prior on group level mean parameters
   mu_b_pr  ~ normal(0,1);   			                  //inverse gain parameter
   mu_ag_pr ~ normal(0,1);   				               //alphaG
   mu_al_pr ~ normal(0,1);   				               //alphaL

   // set prior on group level standard deviations
   sd_b  ~ uniform(0,1.5);     		                  //inverse gain parameter
   sd_ag ~ uniform(0,1.5);   				               //alphaG
   sd_al ~ uniform(0,1.5);   				               //alphaL

   // set prior for individual level parameters
   for (subj in 1:n_s){
      b_ind_pr [subj] ~ normal(mu_b_pr ,  sd_b);     	//inverse gain parameter
      ag_ind_pr[subj] ~ normal(mu_ag_pr, sd_ag);      //alphaG
      al_ind_pr[subj] ~ normal(mu_al_pr, sd_al);   	//alphaL
   }

   // Define epsilon (?) why this value
   epsilon <- 0.00001;

  // now start looping over subjects
   for (trial in 1:n_t[n_s+1]){

      // set initial values (i.e. values for trial 1) for the subject
      if (Init[trial] == 1){
         si <- Subject[trial];                              // Alias the subject id
         for (v in 1:6) {
            prQ0[v] <- 0.5;                                 // (?)
            prQ [v] <- 0.5;                                 // (?)
         }
         pchoice[1] <- 0.5;                                 // Prob. of picking choice 1 on trial 1
         pchoice[2] <- 0.5;                                 // Prob. of picking choice 2 on trial 1
      }

      Qchoice[1] <- prQ[Choice[trial]  ];                   // (?)
      Qchoice[2] <- prQ[Choice[trial]+1];                   // (?)

      pre_squash <- b_ind[si]*(Qchoice[2] - Qchoice[1]);
      pchoice[1] <- 1 / ( 1 + exp(pre_squash) );            // Probability of picking choice 1
      pchoice[2] <- 1 - pchoice[1];                         // Probability of picking choice 2

      pchoice[1] <- epsilon/2+(1-epsilon)*pchoice[1];       // (?)
      pchoice[2] <- epsilon/2+(1-epsilon)*pchoice[2];       // (?)

      Correct[trial] ~ categorical(pchoice);
      success <- Correct[trial] - 1;                        // Success of choice, boolean	
      index   <- Choice[trial] + success;                   // (?)

      // reinforcement
      alpha      <- Reward[trial] *ag_ind[si] + (1-Reward[trial]) *al_ind[si];
      prQ[index] <- prQ[index] + alpha *(Reward[trial] - prQ[index]);
   }
}