clear all
dMatlabProcessManager = 'C:\Users\Dan\code\MatlabProcessManager-master';
dMatlabProcessStanMaster = '/Users/Anne/Dropbox/MesDocuments/WorkGeneral/Softwarelibraries/MatlabStan-master';
addpath(genpath(dMatlabProcessManager));
addpath(genpath(dMatlabProcessStanMaster));

%% simulate data

alphaG = .2;
alphaL = .1;
ns = 24;
betas = sort(7+randn(1,25));
nt = [repmat([150 300],1,ns/2)];nt = [nt sum(nt)];
stims = repmat([1 3 5],1,50);
stims = stims(randperm(length(stims)));
stims = repmat(stims,2*ns);
rewP = [.8 .2 .7 .3 .6 .4];

j = 0;
for si = 1:ns
    Q = .5*ones(1,6);
    for t = 1:nt(si)
        j = j+1;
        s = stims(t);
        pair = Q(s+[0 1]);
        softmax = 1/(1+exp(betas(si)*(pair(1)-pair(2))));% p(1)
        a = 1+(rand<softmax);
        r = rand<rewP(s+(a-1));
        LR = alphaG*r+alphaL*(1-r);
        Q(s+(a-1)) = (1-LR)*Q(s+(a-1))+LR*r;
        Choice(j) = s;
        Correct(j) = a;
        Reward(j)=r;
        Subject(j) = si;
        Init(j) = t==1;
    end
    Q
end

%%

% data{
%   int<lower=1> n_s;                               // number of subjects
%   int<lower=1> n_t;                               // number of trials for that subject
%   int<lower=1,upper=5> Choice[n_t ,n_s];          // choice options trial n_t (choice is 1,3 of 5) for subject n_s
%   int<lower=1,upper=2> Correct[n_t,n_s];          // correct (=1, yes-correct)? trial n_t for subject n_s
%   int<lower=0,upper=1> Reward[n_t,n_s];           // reward (=1, yes)? trial n_t for subject n_s
% }// end data
RL_data = struct('n_s',ns,'n_t',nt,'Choice',Choice,'Correct',Correct,...
    'Reward',Reward,'Subject',Subject,'Init',Init);
%%
tic
fitRL = stan('file','opzet_stan_kort_anne_2.stan','data',RL_data,'iter',1000,...
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
mu_ag =fitRL.extract('permuted',false).mu_ag;
mu_al =fitRL.extract('permuted',false).mu_al;
mu_b =fitRL.extract('permuted',false).mu_b;

figure;
subplot(3,3,1);hist(mu_ag)
subplot(3,3,2);scatter(mu_ag,mu_al);lsline;[rho pval] = corr(mu_ag,mu_al);[rho pval]
subplot(3,3,3);scatter(mu_ag,mu_b);lsline;[rho pval] = corr(mu_ag,mu_b);[rho pval]
subplot(3,3,5);hist(mu_al)
subplot(3,3,6);scatter(mu_al,mu_b);lsline;[rho pval] = corr(mu_al,mu_b);[rho pval]
subplot(3,3,9);hist(mu_b)

subplot(3,3,7)
hist(mu_ag-mu_al)
[h p ci stats] = ttest(mu_ag-mu_al);[p stats.tstat]