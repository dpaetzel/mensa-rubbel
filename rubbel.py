import arviz as az
import IPython
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import stan

# Different groups went to the cafeteria and received tickets/wins as follows.
DATA = np.array([
    # Our initial data.
    # “one of two was a win”
    [1, 2],
    # “one of four was a win”
    [1, 4],
    [0, 1],
    [0, 3],

    # Data from the informal poll in the Discord channel (blubber).
    # [2, 3], # duplicate, also in thread
    [2, 6],
    [0, 3],
    [0, 4],

    # Data from the informal poll in the Discord channel (thread in blubber).
    [0, 5],
    [2, 3],
    [1, 1],
    [2, 3],
    [1, 2],
    [0, 5],
    [0, 6],
    [0, 4],
    [1, 4],
    [0, 4],
    [0, 7],
    [1, 8],
    [2, 5],  # 2022-06-24 23:19

    # Data from the +/- poll in the Discord channel (we removed all people that
    # were already accounted for in the previous data points).
    [10, (10 + 32)],
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
# (a + b) / a = 5
#
# (a + b) = 5 * a
#
# b = 5 * a - a
#
# b = 4 * a
#
# Prior candidates.
alpha = np.linspace(1, 7, 10)
beta = 4 * alpha
fig, ax = plt.subplots(len(alpha))
theta = np.linspace(0, 1, 1000)
for i, ab in enumerate(zip(alpha, beta)):
    a, b = ab
    prior = st.beta(a, b)
    print()
    print(f"Prior candidate {i}: Beta({a}, {b}).")
    prob_mass = prior.cdf(0.5) - prior.cdf(0.05)
    print(f"Probability mass between in [0.05, 0.5]: {prob_mass}")
    print(f"Mean, var:", prior.stats())
    print()
    ax[i].plot(theta, prior.pdf(theta))

plt.show()

# Some tinkering gave me this prior.
alpha = 7
beta = 28

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
