library(rstan)
library(ggplot2)
library(bayesplot)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())


#Loading the data
CBC_data <- read.csv(".../Conjoint_Cleaned_Dataset.csv")[,2:11]
Demographic_data <- read.csv(".../Demographic_Cleaned_Dataset.csv")[,2:5]



#Bayesian Logit Model
XX <- cbind(CBC_data$Product_1,
            CBC_data$Product_2,
            CBC_data$Feature_1,
            CBC_data$Feature_2,
            CBC_data$Feature_3,
            CBC_data$Price)

Temp_list <- list(N=nrow(CBC_data),Y=CBC_data$Choice,A=ncol(XX), X=XX)


write("
      
data {
  int<lower=0> N;
  int<lower=0> A;
  matrix[N,A] X;
  int<lower=0,upper=1> Y[N];
}
parameters {
  vector[A] Beta;
}

model {
  Beta ~ normal(0,10);
  Y ~ bernoulli_logit(X*Beta);
  
}","CBC_Bayesian_logistic_regression.stan")


CBC_Bayesian_Logistic_regression <- stan(
  file = "CBC_Bayesian_logistic_regression.stan",
  data = Temp_list,
  iter = 100000,
  chains = 4
)




#Plotting the traceplots
mcmc_trace(CBC_Bayesian_Logistic_regression,pars=c("Beta[1]","Beta[2]"))
mcmc_trace(CBC_Bayesian_Logistic_regression,pars=c("Beta[3]","Beta[4]"))
mcmc_trace(CBC_Bayesian_Logistic_regression,pars=c("Beta[5]","Beta[6]"))



#Plotting the posterior distributions
color_scheme_set("red")
mcmc_areas(CBC_Bayesian_Logistic_regression,
           pars = c("Beta[1]","Beta[2]","Beta[3]","Beta[4]"),
           prob = 0.75)

color_scheme_set("red")
mcmc_areas(CBC_Bayesian_Logistic_regression,
           pars = c("Beta[5]"),
           prob = 0.75)

color_scheme_set("yellow")
mcmc_areas(CBC_Bayesian_Logistic_regression,
           pars = c("Beta[6]"),
           prob = 0.75)




#Hierarchical Model[7]:

R <- length(unique(CBC_data$ID)) # number of respondents
C <- length(unique(CBC_data$Card)) #number of choices in each round of questions
S <- length(unique(CBC_data$Set)) #number of round of questions
A <- length(colnames(CBC_data)) - 4 #number of attributes in all the choices
D <- length(colnames(Demographic_data))-1 #number of demographic attributes



Y <- array(NaN,dim=c(R,S)) #Array of respondents x sets of choices containing the choice made in each round
X <- array(NaN,dim=c(R,S,C,A)) #Array of possible choices under each set and card
Z <- array(NaN,dim=c(D,R)) #Array of demographic features x respondents


for (r in 1:R) {#For each respondent
  
  #Temporary variable for storing the entire respondents' card
  Respondent <- CBC_data[CBC_data$ID == unique(CBC_data$ID)[r],]
  
  #Respondents' demographic features
  Z[,r] <- t(Demographic_data[,c(2:4)][r,])
  
  for (s in 1:S) {#We're creating a 4d matrix of: for each round of questions S, each possible choice C, all possibilities of K attributes
    
    #Taking out the the given question 
    Question <- Respondent[Respondent$Set == unique(Respondent$Set)[s],]
    X[r,s,,] <- data.matrix(Question[,c(4:9)])
    Y[r,s] <- Question$Card[Question$Choice == 1]
    
    
  }
  
}




write("
data{
  int<lower=1> R; // number of respondents
  int<lower=2> C; // number of choices in each round of questions
  int<lower=1> S; // number of round of questions
  int<lower=1> A; // number of attributes in all the choices
  int<lower=1> D; // number of demographic attributes
  
  int<lower=1,upper=C> Y[R,S]; // Array of respondents x sets of choices containing the choice made in each round
  matrix[C,A] X[R,S]; // Array of possible choices under each set and card
  matrix[D,R] Z; // Array of demographic features x respondents
}

parameters {
  matrix[A,R] Beta; // A x R matrix of coefficients for each respondent and attribute
  matrix[A,D] Gamma; // A x D matrix of coefficients for each demographic feature
  

  corr_matrix[A] Omega; // Prior correlation
  vector<lower=0>[A] tau; // Prior scale
  
}
      
model {

// Priors[1][7][8]
  to_vector(Gamma) ~ normal(0,10);
  tau ~ cauchy(0, 2.5);
  Omega ~ lkj_corr(2);



// Likelihood
    for (r in 1:R) {
      Beta[,r] ~ multi_normal(Gamma*Z[,r],quad_form_diag(Omega, tau));
  
        for (s in 1:S) {
          Y[r,s] ~ categorical_logit(X[r,s]*Beta[,r]);
    }
  }
}","CBC_Hierarchical_multinomial_conditional_logit.stan")



Temp_list <- list(R=R, C=C, S=S, A=A, D=D, Y=Y, X=X, Z=Z)

CBC_Hierarchical_multinomial_conditional_logit <- stan(
  file = "CBC_Hierarchical_multinomial_conditional_logit.stan",
  data = Temp_list,
  iter = 100000,
  chains = 6,
  warmup = 20000
)



#Plotting Rhat values
mcmc_rhat(rhat(CBC_Hierarchical_multinomial_conditional_logit))

#Plotting posterior distribution of group-level coefficients for Gender
mcmc_areas(CBC_Hierarchical_multinomial_conditional_logit,
           pars = c("Gamma[1,1]","Gamma[2,1]","Gamma[3,1]"),
           prob = 0.8)


#Plotting attribute coefficients of respondent 1
plot(CBC_Hierarchical_multinomial_conditional_logit,pars=paste("Beta[",1:6,",1]",sep=""))

#Plotting attribute coefficients of respondent 2
plot(CBC_Hierarchical_multinomial_conditional_logit,pars=paste("Beta[",1:6,",2]",sep=""))

#Plotting Price coefficients for the first 50 respondents
plot(CBC_Hierarchical_multinomial_conditional_logit,pars=paste("Beta[6,",1:50,"]",sep=""))


#References same as in a PDF file