data{
  int<lower=1> n_s;                           // number of subjects
  int<lower=1> n_t[n_s+1];                    // number of trials per subject, plus total number of trials
  int<lower=1,upper=5> Choice[n_t[n_s+1]];    // choice options trial n_t (choice is 1,3 of 5). (All subjects stacked)
  int<lower=1,upper=2> Correct[n_t[n_s+1]];   // correct (=1, yes-correct)? trial n_t
  int<lower=0,upper=1> Reward[n_t[n_s+1]];    // reward (=1, yes)? trial n_t
  int<lower=1,upper=n_s> Subject[n_t[n_s+1]]; // subject number
  int<lower=0,upper=1> Init[n_t[n_s+1]];      // is this first trial of a subject? Should RL be initialized?
}                                             // end data

parameters{
  // group level mean parameters
	real mu_b_pr; 				   //inverse gain parameter
	real mu_ag_pr;   				//alphaG
	real mu_al_pr;   				//alphaL
 

  // group level standard deviation
   real<lower=0> sd_b;   		  	   //inverse gain parameter
	real<lower=0> sd_ag;   				//alphaG
	real<lower=0> sd_al;   				//alphaL
  

  // individual level parameters
   real b_ind_pr[n_s];   			   //inverse gain parameter
	real ag_ind_pr[n_s];   				//alphaG
	real al_ind_pr[n_s];   				//alphaL
	}//end paramters
	
   transformed parameters{
      // group level mean parameters
      real<lower=0,upper=100> mu_b; 		 		//inverse gain parameter
      real<lower=0,upper=1> mu_ag;   				//alphaG
      real<lower=0,upper=1> mu_al;   				//alphaL


      // individual level parameters
      real<lower=0,upper=100> b_ind[n_s];     		//inverse gain parameter
      real<lower=0,upper=1> ag_ind[n_s];   				//alphaG
      real<lower=0,upper=1> al_ind[n_s];   				//alphaL


      // group level mean parameters (probit)
      mu_b  <-Phi(mu_b_pr)*100;   		//inverse gain parameter
      mu_ag <-Phi(mu_ag_pr);   				//alphaG
      mu_al <-Phi(mu_al_pr);   				//alphaL


      // individual level parameters (probit)
      for (s in 1:n_s)
      {
         b_ind[s]  <- Phi(b_ind_pr[s])*100;
         ag_ind[s] <- Phi(ag_ind_pr[s]);
         al_ind[s] <- Phi(al_ind_pr[s]);

      }// end for loop
   }// end transformed parameters
	


model{
  // define general variables needed for subject loop
  int si;
  real prQ0[6];
  real prQ[6];
  real Qchoice[2];
  real epsilon;
  int a;
  real alpha;
  vector[2] pchoice;

  // set prior on group level mean parameters
   mu_b_pr ~  normal(0,1);   			  //inverse gain parameter
	mu_ag_pr ~ normal(0,1);   				//alphaG
	mu_al_pr ~ normal(0,1);   				//alphaL
 

  // set prior on group level standard deviations
   sd_b ~  uniform(0,1.5);     		  //inverse gain parameter
	sd_ag ~ uniform(0,1.5);   				//alphaG
	sd_al ~ uniform(0,1.5);   				//alphaL
  

  // set prior for individual level parameters
  for (s in 1:n_s)
  {
    b_ind_pr[s] ~ normal(mu_b_pr,   sd_b);     		    //inverse gain parameter
	  ag_ind_pr[s]~ normal(mu_ag_pr,  sd_ag);   				//alphaG
	  al_ind_pr[s]~ normal(mu_al_pr,  sd_al);   				//alphaL
   
  }

  
  
  // defineer epsilon
  epsilon <- 0.00001;

  // now start looping over subjects
  for (t in 1:n_t[n_s+1])
  {

      // set initial values subject
      if (Init[t]==1){
            si <- Subject[t];
            for (v in 1:6)
              {
                prQ0[v] <- 0.5;
                prQ[v] <- 0.5;

              }// end inital values loop
          // trial 1
          pchoice[1]<-0.5;
          pchoice[2]<-0.5;
        }

          

          
              
              
          Qchoice[1]    <- prQ[Choice[t]]; 
          Qchoice[2]    <- prQ[(Choice[t]+1)];
            pchoice[1]    <- 1/(1+exp(b_ind[si]*(Qchoice[2]-Qchoice[1])));
                pchoice[2]    <- 1-pchoice[1];
                pchoice[1]    <- epsilon/2+(1-epsilon)*pchoice[1];
          pchoice[2]    <- epsilon/2+(1-epsilon)*pchoice[2];

          Correct[t]~categorical(pchoice);
          a <- Correct[t]-1; //0=correct,1=incorrect	

                // reinforcement
          alpha <- Reward[t]*ag_ind[si]+(1-Reward[t])*al_ind[si];
                prQ[(Choice[t]+a)] <- prQ[(Choice[t]+a)] + alpha*(Reward[t]-prQ[(Choice[t]+a)]);
      
   }// end subject loop
}// end of model loop
