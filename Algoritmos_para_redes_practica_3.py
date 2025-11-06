import os
import math
import networkx as nx
import pandas as pd
from docplex.mp.model import Model

# Función para leer cada una de las instancias generadas 
def lee_archivo(archivo):
    G = nx.Graph()
    coords = {}
    leer_coords = False
    name = ""
    dimension = 0

    with open(archivo, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("NAME"):
                nombre = line.split(":")[1].strip()
            elif line.startswith("DIMENSION"):
                dimension = int(line.split(":")[1].strip())
            elif line.startswith("NODE_COORD_SECTION"):
                leer_coords = True
                continue
            elif line.startswith("EOF"):
                break

            if leer_coords:
                partes = line.split()
                if len(partes) >= 3:
                    nodo_id = int(partes[0]) - 1
                    x = float(partes[1])
                    y = float(partes[2])
                    coords[nodo_id] = (x, y)
                if len(coords) == dimension:
                    leer_coords = False

    # Construir matriz de distancias
    matriz = [[0]*dimension for _ in range(dimension)]
    for i in range(dimension):
        for j in range(dimension):
            if i != j:
                matriz[i][j] = distancia_euclideana(coords[i], coords[j])

    return matriz, nombre, dimension

# Función para calcular la distancia euclideana entre dos puntos
def distancia_euclideana(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def tsp_docplex(matriz):
    n = len(matriz)
    V = range(n)
    m = Model('TSP')

    # Variable de decisión binaria
    x = m.binary_var_matrix(V, V, name='x')

    # Variable auxiliar para eliminar subtours
    u = m.continuous_var_list(V, name='u', lb=1, ub=n)

    # Restricciones de entrada/salida
    m.add_constraints(m.sum(x[i, j] for j in V if i != j) == 1 for i in V)
    m.add_constraints(m.sum(x[i, j] for i in V if i != j) == 1 for j in V)

    # Restricciones Miller, Tucker y Zemlin (MTZ) para evitar subciclos
    for i in V:
        for j in V:
            if i != j and i != 0 and j != 0:
                m.add_constraint(u[i] - u[j] + n * x[i, j] <= n - 1)

    # Función objetivo: minimizar distancia total
    m.minimize(m.sum(matriz[i][j] * x[i, j] for i in V for j in V if i != j))

    sol = m.solve()

    return sol.objective_value

# Ruta para cargar las instancias
RUTA_INSTANCIAS = r"10_instancias"

mejor_solucion = {}

for archivo in os.listdir(RUTA_INSTANCIAS):
    if archivo.lower().endswith(".tsp"):
        filepath = os.path.join(RUTA_INSTANCIAS, archivo)
        try:
            matriz, nombre, dimension = lee_archivo(filepath)
            valor_optimo = tsp_docplex(matriz)
            mejor_solucion[nombre.lower()] = round(valor_optimo, 3)
            print(f"Costo óptimo de {nombre}: {valor_optimo:.3f}\n")
        except Exception as e:
            print(f"Error al procesar {archivo}: {e}\n")

# Condición para almacenar los resultados
if mejor_solucion:
    df = pd.DataFrame(list(mejor_solucion.items()), columns=["Instancia", "Mejor_solucion"])
    csv_path = os.path.join(RUTA_INSTANCIAS, "mejores_soluciones.csv")
    df.to_csv(csv_path, index=False)
else:
    print("No se generaron resultados válidos.")
