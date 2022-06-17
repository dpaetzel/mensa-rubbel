import arviz as az
import IPython
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import stan

# Different groups went to the cafeteria and received tickets/wins as follows.
DATA = np.array([
    # “one of two was a win”
    [1, 2],
    # “one of four was a win”
    [1, 4],
    [0, 1],
    # Data from the informal poll in the Discord channel (blubber).
    [1, 5],
    [2, 3],
    [2, 6],
    [0, 3],
    [0, 5],
    [1, 6],
    # Data from the informal poll in the Discord channel (thread in blubber).
    [2, 3],
    [1, 1],
    [2, 3],
    [1, 2],
    [0, 5],
    [0, 6],
    [0, 4],
    [1, 4],
])

plt.style.use("seaborn")


def fit(n_tickets, n_wins, alpha, beta):
    with open("model.stan") as f:
        program_code = f.read()

    data = {
        "n_tickets": n_tickets,
        "n_wins": n_wins,
        "alpha": alpha,
        "beta": beta,
    }

    # We fix the random seed so models are cached.
    random_seed = 1

    model_: stan.model.Model = stan.build(program_code,
                                          data=data,
                                          random_seed=random_seed)

    fit_: stan.fit.Fit = model_.sample(num_samples=10000)

    data_: az.InferenceData = az.from_pystan(
        posterior=fit_,
        # posterior_predictive
        # predictions
        # prior
        # prior_predictive
        # observed_data
        # constant_data
        # predictions_constant_data
        # log_likelihood
        # coords
        # dims
        posterior_model=model_,
        # prior_model
    )

    return model_, fit_, data_


# 1/10 to 1/20 is our prior belief: Somebody™ said that 10-15% of revenue can be
# used for marketing, 4 € revenue per meal is a rough estimate. This means that
# 40-60 ct per meal can be used for this lottery. Coffee costs 2 €, i.e. to be
# even, each 4th to 5th ticket or so may be a win. We set the lower of this as
# our expected value (due to non-profit, may be even lower). Thus:
#
# a / (a + b) = 1 / 5
#
# Some tinkering gave me this prior.
alpha = 2
beta = 8

# For demonstrational purposes, we train incrementally and plot each updated
# distribution.
n_inc_plots = 6
fig, ax = plt.subplots(1 + n_inc_plots, 1)
prior = st.beta(alpha, beta)
theta = np.linspace(0, 1, 1000)
ax[0].plot(theta, prior.pdf(theta))

# For demonstrational purposes, we train incrementally a few times and plot each
# updated distribution.
for i in range(1, n_inc_plots + 1):
    n_wins = DATA[0:i, 0].sum()
    n_tickets = DATA[0:i, 1].sum()
    model_, fit_, data_ = fit(n_tickets=n_tickets,
                              n_wins=n_wins,
                              alpha=alpha,
                              beta=beta)
    az.plot_posterior(data_, ax=ax[i])

# Now plot the final distribution.
n_wins = DATA[:, 0].sum()
n_tickets = DATA[:, 1].sum()
model_, fit_, data_ = fit(n_tickets=n_tickets,
                          n_wins=n_wins,
                          alpha=alpha,
                          beta=beta)
az.plot_posterior(data_)

print()
print(f"Data mean: {n_wins}/{n_tickets} =", n_wins / n_tickets)
print()
print(az.summary(data_))

plt.show()

IPython.embed(banner1="")
# consider running `globals().update(locals())` in the shell to fix not being
# able to put scopes around variables
