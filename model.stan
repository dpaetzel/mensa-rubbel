data {
  int<lower=1> n_tickets;

  int<lower=1> n_wins;

  real<lower=0> alpha;

  real<lower=0> beta;
}

parameters {
  real<lower=0, upper=1> theta;
}


model {
  theta ~ beta(alpha, beta);
  n_wins ~ binomial(n_tickets, theta);
}
