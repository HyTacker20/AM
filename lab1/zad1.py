import math
import random
import matplotlib.pyplot as plt

def read_tsp_file(filename):
    cities = []
    
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
                x = float(parts[1])
                y = float(parts[2])
                cities.append((x, y))
                
    return cities

def calc_distance(city1, city2):
    x1, y1 = city1
    x2, y2 = city2
    
    dx = x1 - x2
    dy = y1 - y2
    
    return int(math.sqrt(dx**2 + dy**2) + 0.5)

def calc_total_distance(route, cities):
    total_dist = 0
    n = len(route)
    
    for i in range(n - 1):
        city_a_index = route[i]
        city_b_index = route[i+1]
        
        city_a = cities[city_a_index]
        city_b = cities[city_b_index]
        
        total_dist += calc_distance(city_a, city_b)
        
    last_city_index = route[-1]
    first_city_index = route[0]
    total_dist += calc_distance(cities[last_city_index], cities[first_city_index])
    
    return total_dist

def get_min_results_in_chunks(results, chunk_size):
    min_results = []
    for i in range(0, len(results), chunk_size):
        chunk = results[i:i+chunk_size]
        min_results.append(min(chunk))
    return min_results

def plot_route(route, cities, title):
        x = [cities[i][0] for i in route]
        y = [cities[i][1] for i in route]
        
        x.append(cities[route[0]][0])
        y.append(cities[route[0]][1])
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, marker='o', linestyle='-', color='b', markersize=4)
        plt.plot(x[0], y[0], marker='s', color='r', markersize=8, label='Start')
        
        plt.title(title)
        plt.xlabel('Koordynata X')
        plt.ylabel('Koordynata Y')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    filenames = ['wi29.tsp', 'dj38.tsp', 'qa194.tsp', 'uy734.tsp', 'zi929.tsp']
    for filename in filenames:
        my_cities = read_tsp_file(filename)
        print(f"Wczytano {len(my_cities)} miast z pliku {filename}.\n")

        # Wylosuj 1000 permutacji wierzchołków i policz
        city_permutations = []
        for _ in range(1000):
            city_permutations.append(random.sample(range(len(my_cities)), len(my_cities)))

        results = []
        for perm in city_permutations:
            dist = calc_total_distance(perm, my_cities)
            results.append(dist)

        # (a) średnią z minimum dla każdych 10 kolejnych losowań (100 grup po 10 losowań)
        min_by_10 = get_min_results_in_chunks(results, 10)
        mean_by_10 = sum(min_by_10) / len(min_by_10)
        print(f"Średnia z minimum (po 10 losowań): {mean_by_10:.2f}")

        # (b) średnią z minimum dla każdych 50 kolejnych losowań (20 grup po 50 losowań)
        min_by_50 = get_min_results_in_chunks(results, 50)
        mean_by_50 = sum(min_by_50) / len(min_by_50)
        print(f"Średnia z minimum (po 50 losowań): {mean_by_50:.2f}")

        # (c) minimalną wartość dla tych 1000 losowań
        min_overall = min(results)
        print(f"Absolutne minimum dla 1000 losowań: {min_overall}\n")

        best_route_index = results.index(min_overall)
        best_random_route = city_permutations[best_route_index]

        plot_route(best_random_route, my_cities, f"Najlepszy cykl z losowania (Koszt: {min_overall})")