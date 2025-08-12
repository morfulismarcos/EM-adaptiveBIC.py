import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from numpy.linalg import inv
import ot
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde


#file_path = '/Users/marcosmorfulis/Desktop/Input-Puna.xlsx'
#data = pd.read_excel(file_path,sheet_name='SQ', header=None)


#file_path = '/Users/marcosmorfulis/Desktop/Input-Puna.xlsx'

file_path = '/Users/marcosmorfulis/Documents/Proyectos/doctorado/Proyectos/Age Spectra Project/Codes/EM-code/Input-Puna.xlsx'
data = pd.read_excel(file_path,sheet_name='sHEET1', header=None)


#np.random.seed(39)


######################## DEFINICION DE FUNCIONES FUNDAMENTALES #####################################
    

# FunciOn para estimar el numero de modas utilizando densidad kernel
def estimate_modes(data, bandwidth=0.3, num_points=2000, peak_threshold=0.01):
    kde = gaussian_kde(data, bw_method=bandwidth)
    x_vals = np.linspace(min(data), max(data), 1000)
    density = kde(x_vals)
    # Detectar picos en la densidad (modas) 
    peaks = (np.diff(np.sign(np.diff(density))) < 0).nonzero()[0] + 1
    modes = x_vals[peaks]
    return modes

# Funcion softmax para asegurar que las proporciones sumen 1
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Funcion de densidad f_ij (basada en la ecuacion 9 de Sambridge & C. 1994)
def f_ij(ai, tj, sigma_i, p=2):
    exponent = -1/p * (np.abs((ai - tj) / sigma_i) ** p)
    return np.exp(exponent)

# Log-verosimilitud negativa
def negative_log_likelihood(params, data, sigma, p=2):
    n_components = len(params) // 2
    t_js = params[:n_components]
    pi_js = softmax(params[n_components:])
    likelihoods = np.zeros(len(data))
    for i, ai in enumerate(data):
        f_sum = sum(pi_j * f_ij(ai, t_j, sigma[i], p) for pi_j, t_j in zip(pi_js, t_js))
        likelihoods[i] = np.log(f_sum + 1e-10)  # Evitar log(0)
    return -np.sum(likelihoods)

# Funcion para ajustar y calcular BIC para un numero dado de componentes
def fit_model(ages, sigma, n_components):

    # Detectar modas en los datos
    modes = estimate_modes(ages)
    
    # Usar las modas detectadas como medias iniciales, hasta el numero de componentes deseado
    initial_means = modes[:n_components].tolist()
    if len(initial_means) < n_components:
        # Si hay menos modas que el numero de componentes, completar con valores aleatorios en el rango de los datos
        initial_means += list(np.random.uniform(min(ages), max(ages), n_components - len(initial_means)))
    
    # Inicializar proporciones aleatorias y combinar con medias iniciales
    initial_guess = initial_means + np.random.rand(n_components).tolist()
    
    result = minimize(
        negative_log_likelihood,
        initial_guess,
        args=(ages, sigma, 2),
        method='Powell',
        options={'maxiter': 100000, 'disp': False}
    )
    log_likelihood = -result.fun
    k = 2 * n_components  # Numero de parametros (edades y proporciones)
    bic = -2 * log_likelihood + k * np.log(len(ages))
    return result.x, log_likelihood, bic

# Funcion para calcular la matriz Hessiana con diferencias finitas
def approximate_hessian(func, params, epsilon=1e-5, *args):
    n = len(params)
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            params_ij_pp = params.copy()
            params_ij_pp[i] += epsilon
            params_ij_pp[j] += epsilon
            f_ij_pp = func(params_ij_pp, *args)
            
            params_ij_pm = params.copy()
            params_ij_pm[i] += epsilon
            params_ij_pm[j] -= epsilon
            f_ij_pm = func(params_ij_pm, *args)
            
            params_ij_mp = params.copy()
            params_ij_mp[i] -= epsilon
            params_ij_mp[j] += epsilon
            f_ij_mp = func(params_ij_mp, *args)
            
            params_ij_mm = params.copy()
            params_ij_mm[i] -= epsilon
            params_ij_mm[j] -= epsilon
            f_ij_mm = func(params_ij_mm, *args)
            
            hess[i, j] = (f_ij_pp - f_ij_pm - f_ij_mp + f_ij_mm) / (4 * epsilon ** 2)
            hess[j, i] = hess[i, j]  # Simetría
    return hess

# Función para verificar el solapamiento entre medias de componentes
def passes_overlap_filter(t_js, std_t_js, threshold=1.0):
    for i in range(len(t_js)):
        for j in range(i + 1, len(t_js)):
            diff_means = np.abs(t_js[i] - t_js[j])
            sum_std = std_t_js[i] + std_t_js[j]
            if diff_means / sum_std < threshold:
                print(f"Solapamiento detectado entre los componentes {i+1} y {j+1}: |μ_i - μ_j| / (σ_i + σ_j) = {diff_means / sum_std:.2f}")
                return False
    return True

#Funcion para verificar que todas las proporciones sean mayores a cero
def passes_proportion_filter(pi_js):
    if np.any(pi_js <= 0.03):
        print(f"El modelo tiene componentes con π_j = 0.0 o negativo, lo cual es inválido.")
        return False
    return True

# Función para realizar el bootstrapping de la muestra y generar una distribucion promedio
def averaged_distribution(ages, n_resamples=10000, noise_std=2.0):
    bootstrap_samples = [np.sort(np.random.choice(ages, size=len(ages), replace=True) + np.random.normal(0, noise_std, size=len(ages))) for _ in range(n_resamples)]
    averaged_ages = np.mean(np.array(bootstrap_samples), axis=0)
    return averaged_ages



def save_bic_results(bic_results, output_file_path):
    """
    Guarda los valores de BIC calculados para cada modelo de mezcla en un archivo Excel.
    
    Parameters:
    - bic_results: Lista de diccionarios con los BICs para cada distribución y número de componentes.
    - output_file_path: Ruta donde se guardará el archivo Excel.
    """
    # Convertir a DataFrame
    bic_df = pd.DataFrame(bic_results)
    
    # Guardar en un archivo Excel
    bic_df.to_excel(output_file_path, index=False)
    
    print(f"\nLos valores de BIC se han guardado en: {output_file_path}")



################################### FIN DE LAS FUNCIONES ####################################



# Preguntar al usuario si desea trabajar con bootstrapping previo
use_bootstrap = input("¿Deseas trabajar con un bootstrapping previo a las muestras? (si/no): ").strip().lower() == 'si'

# Preguntar al usuario si desea minimizar o encontrar el elbow
selection_method = input("¿Deseas minimizar el criterio o encontrar el 'elbow'? (minimizar/elbow): ").strip().lower()

# Lista para almacenar los resultados de las distribuciones
results_summary = []
bic_summary = []

# Iterar sobre cada par de filas de edades y errores
for idx in range(0, len(data), 2):
    distribution_name = data.iloc[idx, 0]  # Nombre de la distribución
    print(f"\nProcesando muestra {distribution_name}...")

    # Extraer edades y errores para la muestra actual
    ages = data.iloc[idx, 1:].dropna().astype(float).values
    errors_2sigma = data.iloc[idx + 1, 1:].dropna().astype(float).values
    sigma = errors_2sigma / 2

    # Verificar que haya datos válidos
    if len(ages) == 0 or len(sigma) == 0:
        print(f"Muestra {distribution_name} tiene datos insuficientes.")
        continue


    # Realizar el bootstrapping si se selecciono
    if use_bootstrap:
        ages1 = averaged_distribution(ages)
        print(f"Se utilizó una distribución promediada para la muestra {distribution_name}.")
    else:
        ages1 = ages
    
    # Intentar varias veces en caso de error
    success = False
    max_attempts = 5
    attempt = 0

    while not success and attempt < max_attempts:
        try:
            # Iterar sobre 1 a 5 componentes
            n_components_range = range(1, 6)     ##############################################
            log_likelihoods = []
            bics = []
            results = []
            cov_matrices = []

            for n in n_components_range:
                params, log_likelihood, bic = fit_model(ages1, sigma, n)
                log_likelihoods.append(log_likelihood)
                bics.append(bic)
                results.append(params)

                #Calcular la matriz Hessiana y la covarianza
                hess_matrix = approximate_hessian(negative_log_likelihood, params, 1e-5, ages1, sigma, 2)
                mean_diagonal = np.mean(np.abs(np.diag(hess_matrix)))
                regularization_factor = 0.01
                epsilon = regularization_factor * mean_diagonal
                hess_matrix_regularized = hess_matrix + np.eye(hess_matrix.shape[0]) * epsilon
                cov_matrix = inv(hess_matrix_regularized)
                cov_matrices.append(cov_matrix)

                # Calcular W2 y JSD
                x_vals = np.linspace(min(ages1), max(ages1), 200)
                y_vals_combined = np.zeros_like(x_vals)
                bin_width = (max(ages1) - min(ages1)) / 20
                hist_area = len(ages1) * bin_width

                for j in range(n):
                    y_vals = np.array([f_ij(x, params[j], np.mean(sigma), p=2) for x in x_vals])
                    y_vals *= softmax(params[n:])[j] * hist_area / np.trapz(y_vals, x_vals)
                    y_vals_combined += y_vals


            # Normalizar las metricas
            bic_norm = (bics - np.min(bics)) / (np.max(bics) - np.min(bics))
            
            
            #Guardar los BIC de cada modelo enn una lista:
            bic_summary.append({
                "Distribución": distribution_name,
                "n_components": n,
                "BIC": bic_norm
                })


            # Determinar el criterio a analizar según la eleccion del usuario
            criterion = bic_norm



############################################################################################

### METODO ELBOW Y FILTROS DE SOLAPAMIENTO Y PROPORCIONES

            # Método de selección: Minimizar o encontrar el elbow
            if selection_method == 'elbow':
                # Calcular las diferencias entre valores consecutivos del criterio
                diffs = np.diff(criterion)
                print(f"Diferencias entre criterios: {diffs}")
                best_idx = None

                # Definir condiciones específicas para cada índice
                condiciones = {
                    0: {"umbral1": -0.15, "umbral2": -0.15},
                    1: {"umbral1": -0.15, "umbral2": -0.05},
                    2: {"umbral1": -0.05, "umbral2": -0.05}
                }

                # Recorrer las diferencias
                for i in range(len(diffs)):
                    print(f"Evaluando índice: {i}, diffs[i]: {diffs[i]}")
                    
                    # Condicion para el ultimo indice
                    if i == len(diffs) - 1:
                        print(f"Último índice alcanzado: {i}, diffs[i]: {diffs[i]}")
                        # Condicion correcta: xeleccionar el indice si la disminucion es insignificante (mayor que -0.05)
                        if diffs[i] > -0.05:
                            best_idx = i
                            print(f"Se selecciona el elbow en el índice (última diferencia): {best_idx}")
                            break
                        else:
                            best_idx = i+1
                    elif i in condiciones:
                        # Condiciones específicas para indices 0, 1 y 2
                        umbrales = condiciones[i]
                        print(f"Condiciones para el índice {i}: umbral1 = {umbrales['umbral1']}, umbral2 = {umbrales['umbral2']}")
                        if (diffs[i] > umbrales["umbral1"] and 
                            diffs[i + 1] > umbrales["umbral2"]):
                            best_idx = i
                            print(f"Se selecciona el elbow en el índice: {best_idx}")
                            break
                        else:
                            print(f"No se cumplen las condiciones en índice {i}: diffs[i]={diffs[i]}, diffs[i+1]={diffs[i+1]}")
                    else:
                        print(f"Índice {i} no tiene condiciones específicas y no es el último índice.")
                
                # Verificar si el índice 1 debe reconsiderarse
                modes = estimate_modes(ages1)
                if best_idx == 1 and len(modes) == 1:
                    best_idx = 0
                    print(f"Índice final seleccionado: {best_idx}, debido al número de modas = 1")

                # Si no se encuentra un "elbow" claro, tomar el mínimo por defecto
                if best_idx is None:
                    best_idx = np.argmin(criterion)
                    print(f"No se encontró un 'elbow' claro. Seleccionando índice mínimo: {best_idx}")

                # Ajustar el índice para obtener el número de componentes correcto
                if best_idx < len(n_components_range) - 1:
                    initial_best_idx = best_idx + 1
                else:
                    initial_best_idx = best_idx


############################################################################################


          # Ahora usar initial_best_idx de forma segura
            t_js = results[initial_best_idx][:n_components_range[initial_best_idx]]
            pi_js = softmax(results[initial_best_idx][n_components_range[initial_best_idx]:])
            std_errors = np.sqrt(np.diag(cov_matrices[initial_best_idx]))
            std_t_js = std_errors[:n_components_range[initial_best_idx]]


        # Aplicar los filtros al modelo del elbow
            if not (passes_overlap_filter(t_js, std_t_js, threshold=1.0) and passes_proportion_filter(pi_js)):
                print (" ")
                print(f"El modelo seleccionado por el elbow no pasó los filtros. Buscando otro modelo...")

                # Ordenar los índices de los modelos por el criterio (menor a mayor)
                sorted_indices = np.argsort(criterion)

                # Buscar un modelo alternativo con menos componentes que pase los filtros
                for idx in sorted_indices:
                    # Priorizar modelos con menos componentes para evitar el sobreajuste
                    if n_components_range[idx] < n_components_range[initial_best_idx]:
                        t_js = results[idx][:n_components_range[idx]]
                        pi_js = softmax(results[idx][n_components_range[idx]:])
                        std_errors = np.sqrt(np.diag(cov_matrices[idx]))
                        std_t_js = std_errors[:n_components_range[idx]]

                        # Seleccionar el primer modelo que pase los filtros
                        if passes_overlap_filter(t_js, std_t_js, threshold=1.0) and passes_proportion_filter(pi_js):
                            best_idx = idx
                            print (" ")
                            print(f"Se selecciona el modelo alternativo con {n_components_range[idx]} componentes.")
                            break
                else:
                    # Si no se encontro ninguno que pase los filtros, imprimir un mensaje de advertencia
                    print (" ")
                    print("No se encontró un modelo alternativo que cumpla con los filtros. Ningún modelo es válido.")
                    continue  # Saltar a la siguiente muestra si ningún modelo es adecuado

            # Seleccionar el mejor modelo
            best_n_components = n_components_range[best_idx]
            best_params = results[best_idx]
            best_cov_matrix = cov_matrices[best_idx]
            t_js = best_params[:best_n_components]
            pi_js = softmax(best_params[best_n_components:])
            std_errors = np.sqrt(np.diag(best_cov_matrix))
            std_t_js = std_errors[:best_n_components]



            # Almacenar los resultados de la distribución
            result_dict = {
                "Distribuciones": distribution_name,
                "n-components": best_n_components,
            }
            for i in range(1, 6):
                result_dict[f"mean{i}"] = t_js[i - 1] if i <= best_n_components else None
                result_dict[f"std{i}"] = std_t_js[i - 1] if i <= best_n_components else None

            results_summary.append(result_dict)

            # Imprimir el modelo seleccionado para esta distribución
            print(f"\nDistribución {distribution_name}:")
            print(f"Número de componentes: {best_n_components}")
            print("Edades de los componentes (t_j):", t_js)
            print("Proporciones de los componentes (π_j):", pi_js)
            print("Desviaciones estándar de las edades (t_j):", std_t_js)
            


            # Rango de valores para el gráfico
            x_vals = np.linspace(min(ages), max(ages), 200)
            y_vals_combined = np.zeros_like(x_vals)

            # Definir el número de bins y el ancho de bin basado en el histograma total
            num_bins =11
            bin_width = (max(ages) - min(ages)) / num_bins
            hist_area = len(ages) * bin_width

            # Crear listas para almacenar los datos asignados a cada componente
            component_data = [[] for _ in range(best_n_components)]

            # Asignar cada dato a una componente mediante muestreo ponderado
            for age in ages:
                probabilities = np.array([f_ij(age, t_js[j], np.mean(sigma), p=2) * pi_js[j] for j in range(best_n_components)])
                probabilities /= probabilities.sum()  # Normalizar para que sumen 1
                assigned_component = np.random.choice(best_n_components, p=probabilities)
                component_data[assigned_component].append(age)

            # Graficar el histograma para cada componente con el mismo numero de bins y ancho de bin
            colors = ['blue', 'green', 'red', 'purple', 'cyan']  # Añadir más colores si es necesario
            for j in range(best_n_components):
                component_area = len(component_data[j]) * bin_width
                plt.hist(component_data[j], bins=num_bins, density=False, alpha=0.3, color=colors[j % len(colors)],
                         label=f'Datos de Componente {j+1} (π_j={pi_js[j]:.2f})')

                # Graficar curva de densidad para el componente j ajustada a su área específica
                y_vals = np.array([f_ij(x, t_js[j], np.mean(sigma), p=2) for x in x_vals])
                y_vals *= component_area / np.trapz(y_vals, x_vals)
                y_vals_combined += y_vals
                plt.plot(x_vals, y_vals, color=colors[j % len(colors)], linestyle='--', label=f'Curva de Componente {j+1}')

            # Graficar la distribucin combinada
            plt.plot(x_vals, y_vals_combined, 'k-', label='Distribución combinada')

         # Graficar la densidad de kernel del histograma de datos reales
            kde = gaussian_kde(ages)
            plt.plot(x_vals, kde(x_vals) * hist_area, 'r-', label='Kernel de datos observados')

         # Configuracion del grafico
            plt.xlabel('Edades (t_j)')
            plt.ylabel('Frecuencia')
            plt.title(f'Histograma y distribuciones para la muestra {distribution_name}')
            plt.xlim(40, 120)
            plt.savefig('/Users/marcosmorfulis/Desktop/NADA.eps')
            plt.show()

            
            # Si se completa exitosamente, salir del bucle
            success = True

        except (np.linalg.LinAlgError, ValueError) as e:
            # Si ocurre un error (e.g., matriz singular), intentar de nuevo...
            print(f"Error en la muestra {distribution_name}, intento {attempt + 1}: {e}")
            attempt += 1

    if not success:
        print(f"No se pudo procesar la muestra {distribution_name} después de {max_attempts} intentos.")
        continue
    
 # Guardar los resultados en un archivo Excel
output_df = pd.DataFrame(results_summary)
output_file_path = "/Users/marcosmorfulis/Desktop/resultados_distribuciones4.xlsx"
output_df.to_excel(output_file_path, index=False)
print (" ")
print(f"\nLos resultados se han guardado en: {output_file_path}")

output_bic_file = "/Users/marcosmorfulis/Desktop/BIC_results.xlsx"
save_bic_results(bic_summary, output_bic_file)

