import matplotlib.pyplot as plt
import pandas as pd


def estimate_eps(n_dist_values, current_guess, title=None, n=4):
    # Iterate until value is accepted
    while True:
        # Draw plot
        plt.close()
        f, ax = plt.subplots()

        plot_n_dist(n_dist_values, guess=current_guess, ax=ax, title=title, n=n)
        plt.show()

        # Get new user input
        x_new = input('new estimate?' + (' [Empty to accept current value]' if current_guess is not None else ''))
        if x_new == '':
            if current_guess is not None:
                # Accepted
                return current_guess
        else:
            # New value
            try:
                current_guess = float(x_new)
            except ValueError:
                print('Please provide a float value')

        assert isinstance(f, plt.Figure)
        plt.close(f)


def plot_n_dist(n_dist_values, ax, title=None, n=4, guess=None):
    if not isinstance(n_dist_values, pd.Series):
        data = sorted(n_dist_values, reverse=True)
    else:
        data = n_dist_values.sort_values(ascending=False).to_numpy()

    ax.plot(data)

    if title is not None:
        ax.set_title(title)
    ax.set_xlabel('index')
    ax.set_xticks([])
    ax.set_ylabel(f'{n}-dist')

    # Draw current eps if possible
    if guess is not None:
        ax.plot([0, len(n_dist_values)], [guess] * 2, 'k:')
