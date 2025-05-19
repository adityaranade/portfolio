data {
  int<lower = 1> K; // number of (variables) columns in the data
  int<lower = 1> N1; // number of observations in training data
  matrix[N1,K] X1;
  vector[N1] Y1;
  int<lower = 1> N2; // number of observations in testing data
  matrix[N2,K] X2;
  vector[N2] Y2;
}

transformed data {
  vector[N1] mu = rep_vector(0, N1);
  matrix[N1, N1] D2;

// combined matrix
  int<lower=1> N = N1+N2;
  matrix[N,K] Xc;
  
  for (n1 in 1:N1) Xc[n1,] = X1[n1,];
  for (n2 in 1:N2) Xc[(N1 + n2),] = X2[n2,];
  
  matrix[N, N] D20;

  for (i in 1:N){
  for (j in i:N){
    real s = 0;
  for (k in 1:K){  
        s +=  (Xc[i,k] - Xc[j,k])^2;
      }
  D20[i, j] = s;
  D20[j, i] = D20[i, j];
    }
  }
  D2 = D20[1:N1,1:N1];
}

parameters {
  real<lower = 0> lambda;
  real<lower = 0> sigma;
  real<lower = 0> tau;
}

transformed parameters {
}

model {
  matrix[N1, N1] S;
  matrix[N1, N1] L_S;

  // Populate covariance matrix
  for (i in 1:N1) {
    for (j in i:N1) {
      S[i, j] = sigma^2 * exp(-D2[i, j]/(2*lambda^2));
      S[j, i] = S[i, j];
    }
  }

  for (i in 1:N1)
    S[i, i] += tau^2 + (1.490116e-08);

  L_S = cholesky_decompose(S);

  // Set prior density
  // https://mc-stan.org/docs/2_27/stan-users-guide/fit-gp-section.html#priors-for-length-scale
  lambda  ~ std_normal();
  sigma  ~ std_normal();
  tau  ~ std_normal();

  // Set model density
  Y1 ~ multi_normal_cholesky(mu, L_S);
}
generated quantities {
  vector[N2] y_pred ;
  vector[N2] var_pred;
  // vector[N1] y_rep ;
  real log_p;
  {
  matrix[N, N] S;

  // Populate covariance matrix
  for (i in 1:N) {
    for (j in i:N) {
      S[i, j] = sigma^2 * exp(-D20[i, j]/(2*lambda^2));
      S[j, i] = S[i, j];
    }
  }

  for (i in 1:N)
    S[i, i] += tau^2 + (1.490116e-08);

  // for prediction
  matrix[N2, N2] S_nn = S[(N1+1):N,(N1+1):N]; // N2 x N2
  matrix[N1, N1] S_oo = S[1:N1,1:N1];      // N1 x N1
  matrix[N2, N1] S_no = S[(N1+1):N,1:N1];   // N2 x N1
  matrix[N1, N2] S_on = S[1:N1,(N1+1):N];   // N1 x N2
  matrix[N1, N1] S_inv = inverse(S_oo);
  
  vector[N2] mu_pred ;
  matrix[N2, N2] sigma_pred;
  mu_pred = S_no * S_inv * Y1;    // (N2 x N1) * (N1 x N1) * (N1 x 1) = (N2 x 1)
  sigma_pred = S_nn - S_no * S_inv * S_on;   // (N2 x N2) - (N2 x N1) * (N1 x N1) * (N1 x N2) = (N2 x N2)
  //y_pred = multi_normal_rng(mu_pred,sigma_pred);
  y_pred = mu_pred;
  var_pred = diagonal(sigma_pred);

  // for replication
  matrix[N1, N1] L_Sr = cholesky_decompose(S_oo);
  // y_rep = multi_normal_cholesky_rng(mu, L_Sr);
  log_p = multi_normal_lpdf(Y2 | mu_pred , sigma_pred);
  }
}
