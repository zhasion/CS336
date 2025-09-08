import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit


def curve_template(x, a, b):
    return a * x ** b

def fit_optimal_point(optimal_points, ax, type):
    compute_budget, optimal_variable, optimal_final_loss = zip(*optimal_points)

    ax.plot(compute_budget, optimal_variable, marker='o', markersize=8, label=f'{type} optimal')

    hyper_param, _ = curve_fit(curve_template, compute_budget, optimal_variable)
    
    x_fit = np.linspace(min(compute_budget), max(compute_budget), 100)
    y_fit = curve_template(x_fit, *hyper_param)
    
    ax.plot(x_fit, y_fit, 'b--', linewidth=2, label=f'{type} = {hyper_param[0]:.3f}*compute_budget^({hyper_param[1]:.3f})')
    ax.set_xlabel('compute_budget')
    ax.set_ylabel(type)
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')


def plot_optimal(budget_data, type):
    assert type in ['params', 'tokens']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(budget_data.keys())))

    optimal_points = []
    for i, (compute_budget, item) in enumerate(budget_data.items()):
        variable = [a[type] for a in item]
        final_loss = [a['final_loss'] for a in item]

        # find the optimal.
        optimal_index = final_loss.index(min(final_loss))
        optimal_variable = variable[optimal_index]
        optimal_final_loss = final_loss[optimal_index]
        optimal_points.append((compute_budget, optimal_variable, optimal_final_loss))

        # sort.
        variable, final_loss = zip(*sorted(zip(variable, final_loss)))

        # plot
        ax1.scatter(variable, final_loss, alpha=0.7, s=50, color=colors[i])
        ax1.plot(variable, final_loss, alpha=0.5, linewidth=2, color=colors[i], label=f'{compute_budget}')
        ax1.scatter(optimal_variable, optimal_final_loss, color=colors[i], marker='*', s=100, edgecolors='black', linewidths=1)

    if len(optimal_points) > 2:
        fit_optimal_point(optimal_points, ax2, type)

    ax1.legend()
    ax1.set_xlabel(type)
    ax1.set_ylabel('final loss')
    ax1.set_xscale('log')
    plt.show()


def main():
    with open('data/isoflops_curves.json', 'r', encoding='utf8') as f:
        data = json.load(f)

    budget_data = {}
    for item in data:
        parameters = item['parameters']
        compute_budget = item['compute_budget']
        final_loss = item['final_loss']
        if compute_budget not in budget_data.keys():
            budget_data[compute_budget] = []
        budget_data[compute_budget].append({
            'params': parameters,
            'tokens': compute_budget / parameters / 6,
            'final_loss': final_loss
        })
    plot_optimal(budget_data, 'params')
    plot_optimal(budget_data, 'tokens')


if __name__ == '__main__':
    main()