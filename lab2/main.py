import math
import random
import time
import os
import numpy as np

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
                    x = float(parts[1])
                    y = float(parts[2])
                    cities.append((x, y))
    return cities

def create_distance_matrix(cities):
    n = len(cities)
    coords = np.array(cities)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    matrix = np.round(np.sqrt((diff**2).sum(axis=2))).astype(np.int32)
    return matrix

def calc_total_distance(route, dist_matrix):
    total = 0
    n = len(route)
    for i in range(n):
        total += dist_matrix[route[i]][route[(i+1)%n]]
    return total

# --- DELTY (Koszt zmiany - O(1) zamiast O(n) dla przyspieszenia obliczeń) ---

def get_invert_delta(route, i, j, dist_matrix):
    """Zwraca zmianę odległości po odwróceniu fragmentu route[i:j+1]"""
    n = len(route)
    A = route[(i - 1) % n]
    B = route[i]
    C = route[j]
    D = route[(j + 1) % n]
    
    # Usuwamy krawędzie A-B i C-D, dodajemy A-C i B-D
    return dist_matrix[A][C] + dist_matrix[B][D] - dist_matrix[A][B] - dist_matrix[C][D]

def get_transpose_delta(route, i, j, dist_matrix):
    """Zwraca zmianę odległości po zamianie elementów na pozycjach i oraz j"""
    n = len(route)
    if j == i + 1 or (i == 0 and j == n - 1): # Sąsiedzi
        A = route[(i - 1) % n]
        B = route[i]
        C = route[j]
        D = route[(j + 1) % n]
        return dist_matrix[A][C] + dist_matrix[B][D] - dist_matrix[A][B] - dist_matrix[C][D]
    else: # Nie sąsiedzi
        A = route[(i - 1) % n]
        B = route[i]
        C = route[(i + 1) % n]
        X = route[(j - 1) % n]
        Y = route[j]
        Z = route[(j + 1) % n]
        
        old_cost = dist_matrix[A][B] + dist_matrix[B][C] + dist_matrix[X][Y] + dist_matrix[Y][Z]
        new_cost = dist_matrix[A][Y] + dist_matrix[Y][C] + dist_matrix[X][B] + dist_matrix[B][Z]
        return new_cost - old_cost

# --- ALGORYTMY LOCAL SEARCH ---

def local_search_invert(initial_route, dist_matrix):
    """Zadanie 1: Pełne otoczenie dla INVERT"""
    n = len(initial_route)
    route = list(initial_route)
    steps = 0
    current_cost = calc_total_distance(route, dist_matrix)
    
    while True:
        best_delta = 0
        best_move = None
        
        # Szukamy najlepszego sąsiada
        for i in range(n - 1):
            for j in range(i + 1, n):
                # i=0, j=n-1 to odwrócenie całej trasy (ten sam cykl) - pomijamy
                if i == 0 and j == n - 1:
                    continue
                    
                delta = get_invert_delta(route, i, j, dist_matrix)
                if delta < best_delta:
                    best_delta = delta
                    best_move = (i, j)
                    
        if best_delta < 0:
            i, j = best_move
            # Wykonanie ruchu invert
            route[i:j+1] = reversed(route[i:j+1])
            current_cost += best_delta
            steps += 1
        else:
            break # Optimum lokalne
            
    return current_cost, steps, route

def local_search_random_invert(initial_route, dist_matrix):
    """Zadanie 2: Przyspieszone otoczenie - losowe n sąsiadów"""
    n = len(initial_route)
    route = list(initial_route)
    steps = 0
    current_cost = calc_total_distance(route, dist_matrix)
    
    while True:
        best_delta = 0
        best_move = None
        
        # Testujemy n losowych sąsiadów
        for _ in range(n):
            i, j = sorted(random.sample(range(n), 2))
            if i == 0 and j == n - 1:
                continue
            
            delta = get_invert_delta(route, i, j, dist_matrix)
            if delta < best_delta:
                best_delta = delta
                best_move = (i, j)
                
        if best_delta < 0:
            i, j = best_move
            route[i:j+1] = reversed(route[i:j+1])
            current_cost += best_delta
            steps += 1
        else:
            break
            
    return current_cost, steps, route

def local_search_transpose(initial_route, dist_matrix):
    """Zadanie 3 (Dla 2 roku): Pełne otoczenie dla TRANSPOSE"""
    n = len(initial_route)
    route = list(initial_route)
    steps = 0
    current_cost = calc_total_distance(route, dist_matrix)
    
    while True:
        best_delta = 0
        best_move = None
        
        for i in range(n - 1):
            for j in range(i + 1, n):
                delta = get_transpose_delta(route, i, j, dist_matrix)
                if delta < best_delta:
                    best_delta = delta
                    best_move = (i, j)
                    
        if best_delta < 0:
            i, j = best_move
            # Wykonanie ruchu transpose
            route[i], route[j] = route[j], route[i]
            current_cost += best_delta
            steps += 1
        else:
            break
            
    return current_cost, steps, route

if __name__ == "__main__":
    # Twoja oryginalna lista plików + nowe pliki dla tego zadania
    filenames = [
        'wi29.tsp', 
        # 'dj38.tsp', 
        # 'qa194.tsp', 
        # 'uy734.tsp', 
        # 'zi929.tsp', 
        # 'ca4663.tsp', 
        # 'eg7146.tsp', 
        # 'tz6117.tsp', 
        # 'ei8246.tsp', 
        # 'mu1979.tsp'
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
        
        # Generowanie n losowych permutacji startowych
        starts = [random.sample(range(n), n) for _ in range(n)]
        
        # --- ZADANIE 1: Invert (Pełne otoczenie) ---
        t1_start = time.time()
        res_z1 = [local_search_invert(s, dist_matrix) for s in starts]
        t1_time = time.time() - t1_start
        
        avg_cost_1 = sum(r[0] for r in res_z1) / n
        avg_steps_1 = sum(r[1] for r in res_z1) / n
        best_cost_1 = min(r[0] for r in res_z1)
        
        print(f"[Zadanie 1] Local Search (Invert) - Pełne sąsiedztwo:")
        print(f"  Średni koszt:     {avg_cost_1:.2f}")
        print(f"  Średnia l. kroki: {avg_steps_1:.2f}")
        print(f"  Najlepszy koszt:  {best_cost_1}")
        print(f"  Czas obliczeń:    {t1_time:.2f} s\n")

        # --- ZADANIE 2: Invert (Random n sąsiadów) ---
        t2_start = time.time()
        res_z2 = [local_search_random_invert(s, dist_matrix) for s in starts]
        t2_time = time.time() - t2_start
        
        avg_cost_2 = sum(r[0] for r in res_z2) / n
        avg_steps_2 = sum(r[1] for r in res_z2) / n
        best_cost_2 = min(r[0] for r in res_z2)
        
        print(f"[Zadanie 2] Local Search (Random Invert) - {n} sąsiadów:")
        print(f"  Średni koszt:     {avg_cost_2:.2f}")
        print(f"  Średnia l. kroki: {avg_steps_2:.2f}")
        print(f"  Najlepszy koszt:  {best_cost_2}")
        print(f"  Czas obliczeń:    {t2_time:.2f} s\n")

        # --- ZADANIE 3: Transpose (Pełne otoczenie) ---
        t3_start = time.time()
        res_z3 = [local_search_transpose(s, dist_matrix) for s in starts]
        t3_time = time.time() - t3_start
        
        avg_cost_3 = sum(r[0] for r in res_z3) / n
        avg_steps_3 = sum(r[1] for r in res_z3) / n
        best_cost_3 = min(r[0] for r in res_z3)
        
        print(f"[Zadanie 3] Local Search (Transpose) - Pełne sąsiedztwo:")
        print(f"  Średni koszt:     {avg_cost_3:.2f}")
        print(f"  Średnia l. kroki: {avg_steps_3:.2f}")
        print(f"  Najlepszy koszt:  {best_cost_3}")
        print(f"  Czas obliczeń:    {t3_time:.2f} s\n")