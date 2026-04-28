import random
import time
import os
import numpy as np
from numba import njit, prange
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEFAULT_K = 20
K_MAX     = 200

# ---------------------------------------------------------------------------
# Wczytywanie danych
# ---------------------------------------------------------------------------

def read_tsp_file(filename):
    cities = []
    if not os.path.exists(filename):
        print(f"Brak pliku {filename} w folderze. Zignorowano.")
        return cities
    with open(filename, 'r') as file:
        reading_nodes = False
        for line in file:
            line = line.strip()
            if line == "EOF":
                break
            if line == "NODE_COORD_SECTION":
                reading_nodes = True
                continue
            if reading_nodes:
                parts = line.split()
                if len(parts) >= 3:
                    cities.append((float(parts[1]), float(parts[2])))
    return cities

def create_distance_matrix(cities):
    coords = np.array(cities)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.round(np.sqrt((diff**2).sum(axis=2))).astype(np.int32)

def compute_neighbor_lists(dist_matrix, k):
    n = dist_matrix.shape[0]
    k = min(k, n - 1)
    neighbors = np.empty((n, k), dtype=np.int32)
    if k + 1 >= n:
        for v in range(n):
            idx = np.argsort(dist_matrix[v])
            neighbors[v] = idx[idx != v][:k]
    else:
        for v in range(n):
            idx = np.argpartition(dist_matrix[v], k + 1)[:k + 1]
            idx = idx[np.argsort(dist_matrix[v, idx])]
            neighbors[v] = idx[idx != v][:k]
    return neighbors

@njit(cache=True, fastmath=True)
def _nn_route_nb(start, dist_matrix):
    """Greedy nearest-neighbor route starting from `start`."""
    n = dist_matrix.shape[0]
    visited = np.zeros(n, dtype=np.bool_)
    route = np.empty(n, dtype=np.int32)
    route[0] = start
    visited[start] = True
    for step in range(1, n):
        cur = route[step - 1]
        best_d = 2**30
        best_v = -1
        for v in range(n):
            if not visited[v] and dist_matrix[cur, v] < best_d:
                best_d = dist_matrix[cur, v]
                best_v = v
        route[step] = best_v
        visited[best_v] = True
    return route

@njit(parallel=True, cache=True, fastmath=True)
def generate_nn_starts(dist_matrix, start_cities):
    """Równoległa generacja tras greedy NN dla każdego miasta startowego."""
    n_starts = len(start_cities)
    n = dist_matrix.shape[0]
    starts = np.empty((n_starts, n), dtype=np.int32)
    for s in prange(n_starts):
        starts[s] = _nn_route_nb(start_cities[s], dist_matrix)
    return starts

@njit(cache=True, fastmath=True)
def _invert_cl_nb(route, dist_matrix, neighbors):
    """2-opt z candidate lists"""
    n = len(route)
    k = neighbors.shape[1]

    position = np.empty(n, dtype=np.int32)
    for idx in range(n):
        position[route[idx]] = idx

    current_cost = 0
    for idx in range(n):
        current_cost += dist_matrix[route[idx], route[(idx + 1) % n]]

    steps = 0
    while True:
        best_delta = 0
        best_i = -1
        best_j = -1

        for i in range(n):
            A = route[i - 1] if i > 0 else route[n - 1]
            B = route[i]
            d_AB = dist_matrix[A, B]

            for ki in range(k):
                C = neighbors[A, ki]
                d_AC = dist_matrix[A, C]
                if d_AC >= d_AB:
                    break
                if C == B:
                    continue
                j = position[C]
                if i < j:
                    if i == 0 and j == n - 1:
                        continue
                    D = route[j + 1] if j < n - 1 else route[0]
                    delta = d_AC + dist_matrix[B, D] - d_AB - dist_matrix[C, D]
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j

        if best_delta < 0:
            left, right = best_i, best_j
            while left < right:
                tmp = route[left]
                route[left] = route[right]
                route[right] = tmp
                position[route[left]] = left
                position[route[right]] = right
                left += 1
                right -= 1
            current_cost += best_delta
            steps += 1
        else:
            break

    return current_cost, steps


@njit(cache=True, fastmath=True)
def _random_invert_nb(route, dist_matrix):
    """Zadanie 2: n losowych sąsiadów (bez DLB — losowe próbkowanie)."""
    n = len(route)
    current_cost = 0
    for k in range(n):
        current_cost += dist_matrix[route[k], route[(k + 1) % n]]

    steps = 0
    while True:
        best_delta = 0
        best_i = -1
        best_j = -1
        for _ in range(n):
            a = np.random.randint(0, n)
            b = np.random.randint(0, n)
            while b == a:
                b = np.random.randint(0, n)
            i = a if a < b else b
            j = b if a < b else a
            if i == 0 and j == n - 1:
                continue
            A = route[i - 1] if i > 0 else route[n - 1]
            B = route[i]
            C = route[j]
            D = route[j + 1] if j < n - 1 else route[0]
            delta = (dist_matrix[A, C] + dist_matrix[B, D]
                     - dist_matrix[A, B] - dist_matrix[C, D])
            if delta < best_delta:
                best_delta = delta
                best_i = i
                best_j = j
        if best_delta < 0:
            left, right = best_i, best_j
            while left < right:
                tmp = route[left]
                route[left] = route[right]
                route[right] = tmp
                left += 1
                right -= 1
            current_cost += best_delta
            steps += 1
        else:
            break
    return current_cost, steps


@njit(cache=True, fastmath=True)
def _transpose_cl_nb(route, dist_matrix, neighbors):
    """Transpose z candidate lists (best-improvement, bez DLB)."""
    n = len(route)
    k = neighbors.shape[1]

    position = np.empty(n, dtype=np.int32)
    for idx in range(n):
        position[route[idx]] = idx

    current_cost = 0
    for idx in range(n):
        current_cost += dist_matrix[route[idx], route[(idx + 1) % n]]

    steps = 0
    while True:
        best_delta = 0
        best_i = -1
        best_j = -1

        for i in range(n):
            A = route[i - 1] if i > 0 else route[n - 1]
            B = route[i]
            d_AB = dist_matrix[A, B]

            for ki in range(k):
                C = neighbors[A, ki]
                d_AC = dist_matrix[A, C]
                if d_AC >= d_AB:
                    break
                if C == B:
                    continue
                j = position[C]
                if i == j:
                    continue
                ii_ = i if i < j else j
                jj_ = j if i < j else i

                if jj_ == ii_ + 1:
                    A2 = route[ii_ - 1] if ii_ > 0 else route[n - 1]
                    B2 = route[ii_]
                    C2 = route[jj_]
                    D2 = route[jj_ + 1] if jj_ < n - 1 else route[0]
                    delta = (dist_matrix[A2, C2] + dist_matrix[B2, D2]
                             - dist_matrix[A2, B2] - dist_matrix[C2, D2])
                elif ii_ == 0 and jj_ == n - 1:
                    B2 = route[0]
                    C2 = route[n - 1]
                    delta = (dist_matrix[C2, route[1]] + dist_matrix[route[n - 2], B2]
                             - dist_matrix[B2, route[1]] - dist_matrix[route[n - 2], C2])
                else:
                    A2 = route[ii_ - 1] if ii_ > 0 else route[n - 1]
                    B2 = route[ii_]
                    C2 = route[ii_ + 1]
                    X2 = route[jj_ - 1]
                    Y2 = route[jj_]
                    Z2 = route[jj_ + 1] if jj_ < n - 1 else route[0]
                    old_c = (dist_matrix[A2, B2] + dist_matrix[B2, C2]
                             + dist_matrix[X2, Y2] + dist_matrix[Y2, Z2])
                    new_c = (dist_matrix[A2, Y2] + dist_matrix[Y2, C2]
                             + dist_matrix[X2, B2] + dist_matrix[B2, Z2])
                    delta = new_c - old_c

                if delta < best_delta:
                    best_delta = delta
                    best_i = ii_
                    best_j = jj_

        if best_delta < 0:
            i2, j2 = best_i, best_j
            tmp = route[i2]
            route[i2] = route[j2]
            route[j2] = tmp
            position[route[i2]] = i2
            position[route[j2]] = j2
            current_cost += best_delta
            steps += 1
        else:
            break

    return current_cost, steps

# ---------------------------------------------------------------------------
# Numba prange — równoległa obsługa wszystkich startów
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True, fastmath=True)
def run_invert_parallel(starts_2d, dist_matrix, neighbors):
    n_starts = starts_2d.shape[0]
    n = starts_2d.shape[1]
    costs  = np.empty(n_starts, dtype=np.int64)
    steps  = np.empty(n_starts, dtype=np.int64)
    routes = np.empty((n_starts, n), dtype=np.int32)
    for s in prange(n_starts):
        route = starts_2d[s].copy()
        cost, st = _invert_cl_nb(route, dist_matrix, neighbors)
        costs[s] = cost
        steps[s] = st
        routes[s] = route
    return costs, steps, routes

@njit(parallel=True, cache=True, fastmath=True)
def run_random_invert_parallel(starts_2d, dist_matrix):
    n_starts = starts_2d.shape[0]
    n = starts_2d.shape[1]
    costs  = np.empty(n_starts, dtype=np.int64)
    steps  = np.empty(n_starts, dtype=np.int64)
    routes = np.empty((n_starts, n), dtype=np.int32)
    for s in prange(n_starts):
        route = starts_2d[s].copy()
        cost, st = _random_invert_nb(route, dist_matrix)
        costs[s] = cost
        steps[s] = st
        routes[s] = route
    return costs, steps, routes

@njit(parallel=True, cache=True, fastmath=True)
def run_transpose_parallel(starts_2d, dist_matrix, neighbors):
    n_starts = starts_2d.shape[0]
    n = starts_2d.shape[1]
    costs  = np.empty(n_starts, dtype=np.int64)
    steps  = np.empty(n_starts, dtype=np.int64)
    routes = np.empty((n_starts, n), dtype=np.int32)
    for s in prange(n_starts):
        route = starts_2d[s].copy()
        cost, st = _transpose_cl_nb(route, dist_matrix, neighbors)
        costs[s] = cost
        steps[s] = st
        routes[s] = route
    return costs, steps, routes

# ---------------------------------------------------------------------------
# Wizualizacja
# ---------------------------------------------------------------------------

def plot_route(route, cities, title, filepath):
    n = len(route)
    x = [cities[i][0] for i in route] + [cities[route[0]][0]]
    y = [cities[i][1] for i in route] + [cities[route[0]][1]]

    line_width = max(0.8, min(2.2, 240 / max(n, 1)))
    point_size = max(6, min(60, int(1800 / max(n, 1))))

    fig, ax = plt.subplots(figsize=(10, 7), dpi=180)
    ax.set_facecolor('#f7f7f7')
    ax.plot(x, y, linestyle='-', color='#1f5aa6', linewidth=line_width, alpha=0.9, zorder=1)
    ax.scatter(
        x[:-1],
        y[:-1],
        color='#1f5aa6',
        s=point_size,
        edgecolor='white',
        linewidth=0.4,
        alpha=0.95,
        zorder=2,
    )
    ax.scatter(
        [x[0]],
        [y[0]],
        marker='s',
        color='#d62728',
        s=point_size * 2.2,
        edgecolor='white',
        linewidth=0.6,
        label='Start',
        zorder=3,
    )
    ax.set_title(title)
    ax.set_xlabel('Koordynata X')
    ax.set_ylabel('Koordynata Y')
    ax.legend(frameon=True, framealpha=0.9)
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.4)
    ax.set_aspect('equal', adjustable='box')

    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    x_pad = (xmax - xmin) * 0.03 if xmax > xmin else 1.0
    y_pad = (ymax - ymin) * 0.03 if ymax > ymin else 1.0
    ax.set_xlim(xmin - x_pad, xmax + x_pad)
    ax.set_ylim(ymin - y_pad, ymax + y_pad)

    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  Zapisano wykres: {filepath}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Kompilacja Numba JIT (jednorazowo)...")
    _dm = np.array([[0,1,2],[1,0,1],[2,1,0]], dtype=np.int32)
    _ne = np.array([[1,2],[0,2],[1,0]], dtype=np.int32)
    _s2 = np.array([[0,1,2],[2,0,1]], dtype=np.int32)
    _sc = np.array([0,1], dtype=np.int32)
    generate_nn_starts(_dm, _sc)
    run_invert_parallel(_s2, _dm, _ne)
    run_random_invert_parallel(_s2, _dm)
    run_transpose_parallel(_s2, _dm, _ne)
    print("Gotowo.\n")

    filenames = [
        'wi29.tsp',
        'dj38.tsp',
        'qa194.tsp',
        'uy734.tsp',
        'zi929.tsp',
        'ca4663.tsp',
        'eg7146.tsp',
        'tz6117.tsp',
        'ei8246.tsp',
        'mu1979.tsp'
    ]

    for filename in filenames:
        my_cities = read_tsp_file(filename)
        n = len(my_cities)
        if n == 0:
            continue

        print(f"==================================================")
        print(f"Przetwarzanie: {filename} (Liczba miast: {n})")
        print(f"==================================================")

        dist_matrix = create_distance_matrix(my_cities)

        k = min(n - 1, K_MAX)

        # n_starts skaluje się odwrotnie do sqrt(n); n_starts * k ≈ stała
        n_starts = min(n, max(30, int(1500 // n ** 0.5)))

        print(f"  k = {k}  |  n_starts = {n_starts}")
        t_pre = time.time()
        neighbors = compute_neighbor_lists(dist_matrix, k)
        print(f"  Prekomputacja sąsiadów: {time.time() - t_pre:.2f} s")

        starts_list = [random.sample(range(n), n) for _ in range(n_starts)]
        starts_2d = np.array(starts_list, dtype=np.int32)

        # --- Zadanie 1: Invert ---
        t1 = time.time()
        c1, s1, r1 = run_invert_parallel(starts_2d, dist_matrix, neighbors)
        t1 = time.time() - t1
        print(f"[Zadanie 1] Local Search (Invert, CL):")
        print(f"  Średni koszt:     {c1.mean():.2f}")
        print(f"  Średnia l. kroki: {s1.mean():.2f}")
        print(f"  Najlepszy koszt:  {c1.min()}")
        print(f"  Czas obliczeń:    {t1:.2f} s\n")

        # --- Zadanie 2: Random Invert ---
        t2 = time.time()
        c2, s2, r2 = run_random_invert_parallel(starts_2d, dist_matrix)
        t2 = time.time() - t2
        print(f"[Zadanie 2] Local Search (Random Invert, {n} sąsiadów):")
        print(f"  Średni koszt:     {c2.mean():.2f}")
        print(f"  Średnia l. kroki: {s2.mean():.2f}")
        print(f"  Najlepszy koszt:  {c2.min()}")
        print(f"  Czas obliczeń:    {t2:.2f} s\n")

        # --- Zadanie 3: Transpose ---
        t3 = time.time()
        c3, s3, r3 = run_transpose_parallel(starts_2d, dist_matrix, neighbors)
        t3 = time.time() - t3
        print(f"[Zadanie 3] Local Search (Transpose, CL+DLB):")
        print(f"  Średni koszt:     {c3.mean():.2f}")
        print(f"  Średnia l. kroki: {s3.mean():.2f}")
        print(f"  Najlepszy koszt:  {c3.min()}")
        print(f"  Czas obliczeń:    {t3:.2f} s\n")

        # Wykresy
        base = filename.replace('.tsp', '')
        best1 = r1[int(np.argmin(c1))].tolist()
        best2 = r2[int(np.argmin(c2))].tolist()
        best3 = r3[int(np.argmin(c3))].tolist()
        plot_route(best1, my_cities, f"{base} — Invert (koszt: {int(c1.min())})",        f"{base}_z1_invert.png")
        plot_route(best2, my_cities, f"{base} — Random Invert (koszt: {int(c2.min())})", f"{base}_z2_random.png")
        plot_route(best3, my_cities, f"{base} — Transpose (koszt: {int(c3.min())})",     f"{base}_z3_transpose.png")
