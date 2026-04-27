import random
import time
import os
import numpy as np
import multiprocessing as mp

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

def calc_total_distance(route, dist_matrix):
    r = np.asarray(route)
    return int(dist_matrix[r, np.roll(r, -1)].sum())

# --- ALGORYTMY LOCAL SEARCH (wektoryzowane przez numpy) ---

def local_search_invert(initial_route, dist_matrix, ii, jj):
    """Zadanie 1: Pełne otoczenie dla INVERT"""
    n = len(initial_route)
    route = np.array(initial_route, dtype=np.int32)
    steps = 0
    current_cost = calc_total_distance(route, dist_matrix)

    while True:
        A = route[(ii - 1) % n]
        B = route[ii]
        C = route[jj]
        D = route[(jj + 1) % n]
        deltas = dist_matrix[A, C] + dist_matrix[B, D] - dist_matrix[A, B] - dist_matrix[C, D]

        best = int(np.argmin(deltas))
        if deltas[best] < 0:
            i, j = int(ii[best]), int(jj[best])
            route[i:j+1] = route[i:j+1][::-1]
            current_cost += int(deltas[best])
            steps += 1
        else:
            break

    return current_cost, steps, route.tolist()

def local_search_random_invert(initial_route, dist_matrix):
    """Zadanie 2: Przyspieszone otoczenie - losowe n sąsiadów"""
    n = len(initial_route)
    route = np.array(initial_route, dtype=np.int32)
    steps = 0
    current_cost = calc_total_distance(route, dist_matrix)

    while True:
        pairs = np.array([sorted(random.sample(range(n), 2)) for _ in range(n)])
        valid = ~((pairs[:, 0] == 0) & (pairs[:, 1] == n - 1))
        pairs = pairs[valid]
        if len(pairs) == 0:
            break

        ii_r, jj_r = pairs[:, 0], pairs[:, 1]
        A = route[(ii_r - 1) % n];  B = route[ii_r]
        C = route[jj_r];            D = route[(jj_r + 1) % n]
        deltas = dist_matrix[A, C] + dist_matrix[B, D] - dist_matrix[A, B] - dist_matrix[C, D]

        best = int(np.argmin(deltas))
        if deltas[best] < 0:
            i, j = int(ii_r[best]), int(jj_r[best])
            route[i:j+1] = route[i:j+1][::-1]
            current_cost += int(deltas[best])
            steps += 1
        else:
            break

    return current_cost, steps, route.tolist()

def local_search_transpose(initial_route, dist_matrix, ii, jj, linear_adj, cyclic_adj, non_adj):
    """Zadanie 3 (Dla 2 roku): Pełne otoczenie dla TRANSPOSE"""
    n = len(initial_route)
    route = np.array(initial_route, dtype=np.int32)
    steps = 0
    current_cost = calc_total_distance(route, dist_matrix)

    while True:
        deltas = np.zeros(len(ii), dtype=np.int64)

        # Sąsiedzi liniowi (j == i+1)
        if linear_adj.any():
            idx = linear_adj
            A = route[(ii[idx] - 1) % n];  B = route[ii[idx]]
            C = route[jj[idx]];            D = route[(jj[idx] + 1) % n]
            deltas[idx] = dist_matrix[A, C] + dist_matrix[B, D] - dist_matrix[A, B] - dist_matrix[C, D]

        # Sąsiedzi cykliczni (i=0, j=n-1)
        if cyclic_adj.any():
            B_c = route[0];  C_c = route[n - 1]
            deltas[cyclic_adj] = (dist_matrix[C_c, route[1]] + dist_matrix[route[n - 2], B_c]
                                  - dist_matrix[B_c, route[1]] - dist_matrix[route[n - 2], C_c])

        # Nie-sąsiedzi
        if non_adj.any():
            idx = non_adj
            A = route[(ii[idx] - 1) % n];  B = route[ii[idx]];  C = route[(ii[idx] + 1) % n]
            X = route[(jj[idx] - 1) % n];  Y = route[jj[idx]];  Z = route[(jj[idx] + 1) % n]
            old_c = dist_matrix[A, B] + dist_matrix[B, C] + dist_matrix[X, Y] + dist_matrix[Y, Z]
            new_c = dist_matrix[A, Y] + dist_matrix[Y, C] + dist_matrix[X, B] + dist_matrix[B, Z]
            deltas[idx] = new_c - old_c

        best = int(np.argmin(deltas))
        if deltas[best] < 0:
            i, j = int(ii[best]), int(jj[best])
            route[i], route[j] = int(route[j]), int(route[i])
            current_cost += int(deltas[best])
            steps += 1
        else:
            break

    return current_cost, steps, route.tolist()

# --- MULTIPROCESSING: worker state (inicjalizowany raz na proces) ---

_W = {}  # worker-local state

def _pool_init(dm, ii_i, jj_i, ii_t, jj_t, la, ca, na):
    _W['dm'] = dm
    _W['ii_inv'] = ii_i;  _W['jj_inv'] = jj_i
    _W['ii_tr']  = ii_t;  _W['jj_tr']  = jj_t
    _W['la'] = la;  _W['ca'] = ca;  _W['na'] = na

def _task_invert(route):
    return local_search_invert(route, _W['dm'], _W['ii_inv'], _W['jj_inv'])

def _task_random_invert(route):
    return local_search_random_invert(route, _W['dm'])

def _task_transpose(route):
    return local_search_transpose(route, _W['dm'], _W['ii_tr'], _W['jj_tr'], _W['la'], _W['ca'], _W['na'])

# --- MAIN ---

if __name__ == "__main__":
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

    ncpus = mp.cpu_count()
    print(f"Używam {ncpus} rdzeni CPU\n")

    for filename in filenames:
        my_cities = read_tsp_file(filename)
        n = len(my_cities)
        if n == 0:
            continue

        print(f"==================================================")
        print(f"Przetwarzanie: {filename} (Liczba miast: {n})")
        print(f"==================================================")

        dist_matrix = create_distance_matrix(my_cities)
        starts = [random.sample(range(n), n) for _ in range(n)]

        ii_inv, jj_inv = np.triu_indices(n, k=1)
        valid = ~((ii_inv == 0) & (jj_inv == n - 1))
        ii_inv, jj_inv = ii_inv[valid], jj_inv[valid]

        ii_tr, jj_tr = np.triu_indices(n, k=1)
        linear_adj = (jj_tr == ii_tr + 1)
        cyclic_adj  = (ii_tr == 0) & (jj_tr == n - 1)
        non_adj     = ~linear_adj & ~cyclic_adj

        initargs = (dist_matrix, ii_inv, jj_inv, ii_tr, jj_tr, linear_adj, cyclic_adj, non_adj)

        with mp.Pool(ncpus, initializer=_pool_init, initargs=initargs) as pool:

            # --- ZADANIE 1 ---
            t1 = time.time()
            res_z1 = pool.map(_task_invert, starts)
            t1 = time.time() - t1

            avg_cost_1  = sum(r[0] for r in res_z1) / n
            avg_steps_1 = sum(r[1] for r in res_z1) / n
            best_cost_1 = min(r[0] for r in res_z1)
            print(f"[Zadanie 1] Local Search (Invert) - Pełne sąsiedztwo:")
            print(f"  Średni koszt:     {avg_cost_1:.2f}")
            print(f"  Średnia l. kroki: {avg_steps_1:.2f}")
            print(f"  Najlepszy koszt:  {best_cost_1}")
            print(f"  Czas obliczeń:    {t1:.2f} s\n")

            # --- ZADANIE 2 ---
            t2 = time.time()
            res_z2 = pool.map(_task_random_invert, starts)
            t2 = time.time() - t2

            avg_cost_2  = sum(r[0] for r in res_z2) / n
            avg_steps_2 = sum(r[1] for r in res_z2) / n
            best_cost_2 = min(r[0] for r in res_z2)
            print(f"[Zadanie 2] Local Search (Random Invert) - {n} sąsiadów:")
            print(f"  Średni koszt:     {avg_cost_2:.2f}")
            print(f"  Średnia l. kroki: {avg_steps_2:.2f}")
            print(f"  Najlepszy koszt:  {best_cost_2}")
            print(f"  Czas obliczeń:    {t2:.2f} s\n")

            # --- ZADANIE 3 ---
            t3 = time.time()
            res_z3 = pool.map(_task_transpose, starts)
            t3 = time.time() - t3

            avg_cost_3  = sum(r[0] for r in res_z3) / n
            avg_steps_3 = sum(r[1] for r in res_z3) / n
            best_cost_3 = min(r[0] for r in res_z3)
            print(f"[Zadanie 3] Local Search (Transpose) - Pełne sąsiedztwo:")
            print(f"  Średni koszt:     {avg_cost_3:.2f}")
            print(f"  Średnia l. kroki: {avg_steps_3:.2f}")
            print(f"  Najlepszy koszt:  {best_cost_3}")
            print(f"  Czas obliczeń:    {t3:.2f} s\n")
