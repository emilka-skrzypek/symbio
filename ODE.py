import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
import math
import warnings
import contextlib

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Parametry nominalne ---
params_nominal = {
    'p1': 8.8,
    'p2': 440,
    'p3': 100,
    'd1': 1.375e-14,
    'd2': 1.375e-1,
    'd3': 3e-5,
    'k1': 1.925e-4,
    'k2': 1e5,
    'k3': 1.5e5
}

# --- Model ODE ---
def model(t, y, p):
    p53, MDMcyt, MDMn, PTEN = y
    dp53 = p['p1'] - p['d1'] * p53 * MDMn**2
    dMDMcyt = (
        p['p2'] * (p53**4 / (p53**4 + p['k2']**4)) -
        p['k1'] * (p['k3']**2 / (p['k3']**2 + PTEN**2)) * MDMcyt -
        p['d2'] * MDMcyt
    )
    dMDMn = (
        p['k1'] * (p['k3']**2 / (p['k3']**2 + PTEN**2)) * MDMcyt -
        p['d2'] * MDMn
    )
    dPTEN = p['p3'] * (p53**4 / (p53**4 + p['k2']**4)) - p['d3'] * PTEN
    return [dp53, dMDMcyt, dMDMn, dPTEN]

def get_scenario_params(scenario_id):
    base = params_nominal.copy()
    if scenario_id == 1:
        return base
    elif scenario_id == 3:
        base['p1'] *= 1.5
        base['d1'] *= 0.7
        return base
    else:
        raise ValueError("Nieznany scenariusz. Wybierz 1 (zdrowe) lub 3 (nowotworowe).")

def simulate(params, t_end=48, method="RK45"):
    t_eval = np.linspace(0, t_end, 1000)
    y0 = [1e-6, 1e-6, 1e-6, 0]
    sol = solve_ivp(model, [0, 48], y0, args=(params,), t_eval=t_eval)
    return sol.t, sol.y

def local_sensitivity(param_key, variation=0.2):
    base = params_nominal.copy()
    t, y_base = simulate(base)
    p53_base = y_base[0]

    up = base.copy()
    up[param_key] *= (1 + variation)
    _, y_up = simulate(up)

    down = base.copy()
    down[param_key] *= (1 - variation)
    _, y_down = simulate(down)

    return t, p53_base, y_up[0], y_down[0]

def global_sensitivity():
    problem = {
        'num_vars': len(params_nominal),
        'names': list(params_nominal.keys()),
        'bounds': [[v * 0.8, v * 1.2] for v in params_nominal.values()]
    }

    param_values = sobol_sample.sample(problem, 1024)
    Y = []

    for row in param_values:
        p = dict(zip(problem['names'], row))
        _, y = simulate(p)
        Y.append(y[0][-1])

    Si = sobol_analyze.analyze(problem, np.array(Y), print_to_console=True)
    return Si

def plot_local(param):
    t, base, up, down = local_sensitivity(param)
    plt.plot(t, base, label="Base")
    plt.plot(t, up, label=f"{param} +20%")
    plt.plot(t, down, label=f"{param} -20%")
    plt.yscale("linear")
    plt.xlabel("Czas (h)")
    plt.ylabel("Poziom p53")
    plt.title(f"Lokalna wrażliwość: {param}")
    plt.legend()
    plt.grid()
    plt.show()

def plot_sobol(Si):
    keys = list(params_nominal.keys())
    s1 = Si['S1']
    plt.bar(keys, s1)
    plt.ylabel("Indeks Sobola S1")
    plt.title("Globalna analiza wrażliwości – p53 @ t=200h")
    plt.yscale("linear")
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

def local_sensitivity_ranking():
    rankings = []
    for key in params_nominal.keys():
        t, base, up, down = local_sensitivity(key)
        delta_up = np.abs(up - base) / (base + 1e-8)
        delta_down = np.abs(down - base) / (base + 1e-8)
        delta_mean = (delta_up + delta_down) / 2
        score = np.mean(delta_mean)
        rankings.append((key, score))
    rankings.sort(key=lambda x: -x[1])
    return rankings

def plot_rankings(local_rank, global_sobol):
    local_keys = [k for k, _ in local_rank]
    local_vals = [v for _, v in local_rank]
    global_keys = list(params_nominal.keys())
    global_vals = list(global_sobol['S1'])

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    axs[0].bar(local_keys, local_vals, color="skyblue")
    axs[0].set_title("Ranking lokalnej wrażliwości")
    axs[0].set_ylabel("Średnia względna zmiana p53")
    axs[0].tick_params(axis='x', rotation=45)
    axs[0].grid(True)
    axs[0].set_ylim(0, 1)

    axs[1].bar(global_keys, global_vals, color="salmon")
    axs[1].set_title("Ranking globalnej wrażliwości (Sobol S1)")
    axs[1].set_ylabel("Indeks Sobola S1")
    axs[1].tick_params(axis='x', rotation=45)
    axs[1].grid(True)
    axs[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

def plot_all_sensitivity_curves():
    n_params = len(params_nominal)
    cols = 3
    rows = math.ceil(n_params / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axs = axs.flatten()

    for idx, param in enumerate(params_nominal.keys()):
        t, base, up, down = local_sensitivity(param)
        delta_up = np.abs(up - base) / (base + 1e-8)
        delta_down = np.abs(down - base) / (base + 1e-8)
        sensitivity_time = (delta_up + delta_down) / 2
        axs[idx].plot(t, sensitivity_time, label=f"Sens({param})")
        axs[idx].set_yscale("linear")
        axs[idx].set_title(f"{param}")
        axs[idx].set_xlabel("Czas [h]")
        axs[idx].set_ylabel("Zmiana względna p53")
        axs[idx].grid(True)
        axs[idx].legend()

    for ax in axs[n_params:]:
        ax.axis("off")

    plt.tight_layout()
    plt.suptitle("Zadanie 5: Funkcje wrażliwości lokalnej w czasie", y=1.02, fontsize=16)
    plt.savefig("zad5_sensitivity_curves.png")
    plt.show()

def plot_all_p53_changes():
    n_params = len(params_nominal)
    cols = 3
    rows = math.ceil(n_params / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axs = axs.flatten()

    for idx, param in enumerate(params_nominal.keys()):
        t, base, up, down = local_sensitivity(param)
        base = np.where(base <= 0, 1e-8, base)
        up = np.where(up <= 0, 1e-8, up)
        down = np.where(down <= 0, 1e-8, down)

        axs[idx].plot(t, base, label="Base")
        axs[idx].plot(t, up, label="+20%")
        axs[idx].plot(t, down, label="-20%")
        axs[idx].set_yscale("linear")

        axs[idx].set_title(f"{param}")
        axs[idx].set_xlabel("Czas [h]")
        axs[idx].set_ylabel("Poziom p53")
        axs[idx].grid(True)
        axs[idx].legend()

    for ax in axs[n_params:]:
        ax.axis("off")

    plt.tight_layout()
    plt.suptitle("Zadanie 6: Zmiana poziomu p53 dla ±20% każdego parametru", y=1.02, fontsize=16)
    plt.savefig("zad6_p53_change.png")
    plt.show()

def plot_task6_p53_change(top_param, bottom_param):
    for param in [top_param, bottom_param]:
        t, base, up, down = local_sensitivity(param)
        plt.plot(t, base, label="Base")
        plt.plot(t, up, label=f"{param} +20%")
        plt.plot(t, down, label=f"{param} -20%")
        plt.xlabel("Czas [h]")
        plt.ylabel("Poziom p53")
        plt.title(f"Zmiana poziomu p53 dla ±20% parametru {param}")
        plt.grid()
        plt.legend()
        plt.yscale("linear")
        plt.show()

def run_all_analysis_plots():
    local_rank = local_sensitivity_ranking()
    top_param = local_rank[0][0]
    bottom_param = local_rank[-1][0]

    print(f"\nTop parameter (największy wpływ): {top_param}")
    print(f"Bottom parameter (najmniejszy wpływ): {bottom_param}")

    plot_all_sensitivity_curves()
    plot_all_p53_changes()
    plot_task6_p53_change(top_param, bottom_param)

def compare_scenarios():
    for scen in [1, 3]:
        print(f"\n=== Scenariusz {scen} ===")
        print("Lokalna analiza wrażliwości (ranking):")
        ranking = []
        for key in params_nominal.keys():
            t, base, up, down = local_sensitivity(key)
            delta_up = np.abs(up - base) / (base + 1e-8)
            delta_down = np.abs(down - base) / (base + 1e-8)
            delta_mean = (delta_up + delta_down) / 2
            score = np.mean(delta_mean)
            ranking.append((key, score))
        ranking.sort(key=lambda x: -x[1])
        for k, v in ranking:
            print(f"{k}: {v:.4f}")

        print("\nGlobalna analiza Sobola:")
        Si = global_sensitivity()
        for k, v in zip(params_nominal.keys(), Si['S1']):
            print(f"{k}: {v:.4f}")

# === WYKONANIE ===
with open("analiza_wyniki.txt", "w") as f:
    with contextlib.redirect_stdout(f):
        t, y = simulate(params_nominal, method="RK45")
        plt.plot(t, y[0], label="p53")
        plt.plot(t, y[1], label="MDMcyt")
        plt.plot(t, y[2], label="MDMn")
        plt.plot(t, y[3], label="PTEN")
        plt.xlabel("Czas [h]")
        plt.ylabel("Stężenie")
        plt.legend()
        plt.title("Symulacja układu (RK4)")
        plt.grid()
        plt.show()

        plot_local("p1")
        plot_local("d3")

        local_rank = local_sensitivity_ranking()
        Si = global_sensitivity()
        plot_rankings(local_rank, Si)

        print("\nLocal sensitivity ranking:")
        for k, v in local_rank:
            print(f"{k}: {v:.4f}")

        print("\nGlobal Sobol S1:")
        for k, v in zip(params_nominal.keys(), Si['S1']):
            print(f"{k}: {v:.4f}")

        run_all_analysis_plots()
        compare_scenarios()
