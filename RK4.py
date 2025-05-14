# Automatyczna instalacja bibliotek (aby program kompilował się u wszystkich użytkowników)
import subprocess
import sys

def install_if_missing(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Instalowanie pakietu: {package} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_if_missing('numpy')
install_if_missing('matplotlib')

# Importowanie bibliotek
import numpy as np
import matplotlib.pyplot as plt

# Parametry biologiczne modelu
params_bio = {
    'p1': 8.8,              # produkcja białka p53
    'p2': 440,              # produkcja MDM2 w cytoplazmie
    'p3': 100,              # produkcja PTEN
    'd1': 1.375e-14,        # degradacja p53
    'd2': 1.375e-4,         # degradacja MDM2 jądrowego i cytoplazmatycznego
    'd3': 3e-5,             # degradacja PTEN
    'k1': 1.925e-4,         # szybkość transportu MDM2 z cytoplazmy do jądra
    'k2': 1e5,              # próg aktywacji produkcji MDM2
    'k3': 1.5e5,            # próg regulacji przez PTEN
}

# Funkcja modelująca równania ODE- zmiany parametrów w czasie
def model(time, concentration, params):
    p1, p2, p3 = params['p1'], params['p2'], params['p3']       # pobranie parametrów biologicznych z lokalnej kopii params_bio
    d1, d2, d3 = params['d1'], params['d2'], params['d3']
    k1, k2, k3 = params['k1'], params['k2'], params['k3']

    p53, mdm2_c, mdm2_n, pten = concentration                   # wektor aktualnych stężeń białek

    # Dodatkowe regulacje zależne od stanu
    if params.get('sirna', True):
        d1 += 0.02  # siRNA zwiększa degradację p53
    if params.get('pten_active', False):
        p3 = 0  # brak produkcji PTEN
    if params.get('dna_damage', False):
        d1 -= 0.1  # mniej degradacji gdy brak uszkodzenia

    dp53 = p1 - d1 * p53 * (mdm2_n ** 2)
    dmdm2_c = p2 * (p53 ** 4) / (p53 ** 4 + k2 ** 4) - k1 * (k3 ** 2 / (k3 ** 2 + pten ** 2)) * mdm2_c - d2 * mdm2_c
    dmdm2_n = k1 * (k3 ** 2 / (k3 ** 2 + pten ** 2)) * mdm2_c - d2 * mdm2_n
    dpten = p3 * (p53 ** 4) / (p53 ** 4 + k2 ** 4) - d3 * pten

    if time < 0.5:  # tylko dla pierwszych kroków, żeby nie zasypać terminala
        print(f"[t={time:.2f}] dp53={dp53:.3e}, dmdm2_c={dmdm2_c:.3e}, dmdm2_n={dmdm2_n:.3e}, dpten={dpten:.3e}")
    return np.array([dp53, dmdm2_c, dmdm2_n, dpten])            # wektor szybkości zmian dla każdedo białka

# RK4 ze zmiennym krokiem dopasowującym się po każdej iteracji
def rk4_step(f, time, concentration, time_step, params):
    k1 = f(time, concentration, params)
    k2 = f(time + time_step/2, concentration + time_step*k1/2, params)
    k3 = f(time + time_step/2, concentration + time_step*k2/2, params)
    k4 = f(time + time_step, concentration + time_step*k3, params)
    return concentration + time_step * (k1 + 2*k2 + 2*k3 + k4) / 6

def integrate_rk4_adaptive(f, concentration0, time0, t_end, time_step0, tol, params):   # tol- tolerancja błędu (maksymalny dopuszczalny błąd w obliczeniach)
    time, concentration, time_step = time0, concentration0.copy(), time_step0
    result = [(time, concentration.copy())]

    while time < t_end:
        y_full = rk4_step(f, time, concentration, time_step, params)                    # pełny krok (time_step) klasyczną metodą RK4
        y_half1 = rk4_step(f, time, concentration, time_step/2, params)                 # dzieli krok na dwa mniejsze, aby oszacować błąd numeryczny— pierwszy mały krok
        y_half2 = rk4_step(f, time + time_step/2, y_half1, time_step/2, params)         # drugi mały krok, ale zaczynający się po pierwszym kroku
        error = np.linalg.norm(y_half2 - y_full) / (np.linalg.norm(y_half2) + 1e-10)    # obliczenie i normalizacja błędu

        if error < tol:                                                                 # adaptacja kroku czasowego: jeśli błąd jest akceptowalny- aktualizuje czas 
            time += time_step                                                           # jeśli błąd jest zbyt duży- zmniejsza krok czasowy o połowę
            concentration = y_half2
            result.append((time, concentration.copy()))
            time_step *= 2 if error < tol/10 else 1
        else:
            time_step /= 2
    return result

# Scenariusze biologiczne
scenarios = {
    'A': {'pten_active': True, 'dna_damage': False, 'sirna': False},
    'B': {'pten_active': True, 'dna_damage': True, 'sirna': False},
    'C': {'pten_active': False, 'dna_damage': True, 'sirna': False},
    'D': {'pten_active': False, 'dna_damage': True, 'sirna': True}
}

# Symulacja i wykresy
def simulate_and_plot():
    t0, t_end = 0, 48                    # czas początkowy i końcowy
    h0 = 0.1                             # krok czasowy (6 min)
    y0 = np.array([0.0, 0.0, 0.0, 0.0])  # [p53, MDM2_c, MDM2_n, PTEN]
    tol = 1e-4                           # tolerancja błędu w RK4
    labels = ['p53', 'MDM2_c', 'MDM2_n', 'PTEN']

    for scenario, conditions in scenarios.items():
        sim_params = params_bio.copy()      # tworzymy lokalną kopię params_bio dla symulacji każdego scenariusza 
        sim_params.update(conditions)       # nadpisuje ustawienia specyficzne dla danego scenariusza
        print(f"\n=== Scenariusz {scenario} ===")
        print("Parametry symulacji:")
        for key in ['d1', 'p3', 'pten_active', 'sirna', 'dna_damage']:
            print(f"  {key} = {sim_params.get(key)}")
        results = integrate_rk4_adaptive(model, y0, t0, t_end, h0, tol, sim_params)     # wykonanie symulacji
        # Pokazuje pierwsze 5 kroków symulacji:
        print("Pierwsze 5 kroków:")
        for t, y in results[:5]:
            print(f"t={t:.2f} -> p53={y[0]:.4f}, MDM2_c={y[1]:.4f}, MDM2_n={y[2]:.4f}, PTEN={y[3]:.4f}")


        ts, ys = zip(*results)          # skompresowanie wyników (grupuje wszystkie pierwsze elementy (czasy) i drugie elementy (wektory stężeń) razem)
        ys = np.array(ys)               # konwersja ys z krotki do tablicy NumPy

        plt.figure(figsize=(10, 6))         # generowanie wykresu
        for i in range(4):
            plt.semilogy(ts, ys[:, i], label=labels[i])
        plt.title(f"Scenariusz {scenario}")
        plt.xlabel("Czas [time_step]")
        plt.ylabel("Stężenie / aktywność")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

simulate_and_plot()
