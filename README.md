# rede_neuronal
Rede neuronal para el análisis algorítmico y estructuras de datos

# Descripción del proyecto
Se debe utilizar modelos ML (Machine Learning) o DL (Deep Learning), para:
- Predecir la complejidad algorítmica de un código.
- Optimizar el ordenamiento de listas usando RL (Reinforced Learning).
- Resolver operaciones en pilas, colas y arboles binarios.

# 1- Red neuronal de complejidad argorítmica

- Entrada: Representación vectorial de código (usando embeddings de tokens o ASTs*).
- Modelo:
    * CNN/Transformer para extraer features del código.
    * Clasificación multitarea: Salidas separadas para O, Ω, ϴ.
- Dataset:
    * Generar ejemplos de algoritmos etiquetados con su complejidad (ej: O(n^2) para bucles anidados).
    * o Usar sintetizadores de código (ej: GitHub Copilot para variaciones)

# 2- Red neuronal para ordenamiento de lisas

- Aprendizaje Supervisado:
    * Entrenar una RNN/LSTM (Recurrent Neural Network with long term and short term memory) para predecir el siguiente paso en un algoritmo de ordenamiento (ej: intercambiar elementos en Bubble Sort)
- Aprendizaje por Refuerzo (RL):
    * Agente (DQN/PPO) - (Deep Q-Network / Proximal Policy Optimization) que elige entre acciones como swap, pivot, merge para minimizar comparaciones.
    * Recompensa: -1 por comparación, +100 si la lista está ordenada.

# 3- Redes para estructuras de datos

- A. Pilas y Colas:
    * Modelo: RNN o GNN (Graph Neural Network) para aprender operaciones:
        ° Ej: Predecir el resultado de pop() después de múltiples push().
    * Dataset: Secuencias de operaciones aleatorias (ej: push(3) → push(5) → pop() → 5).
- B. Árboles Binarios:
    * Modelo: GNN para:
        ° Buscar nodos, recorridos (in-order), o balancear el árbol.
    * Dataset: Árboles generados aleatoriamente con operaciones etiquetadas.

# 4- Tecnologías Clave
    * Lenguaje: Python (PyTorch/TensorFlow).
    * Procesamiento de código: Librerías como libcst, tree-sitter para parsear código.
    * RL: OpenAI Gym o entorno custom para ordenamiento.
    * GNNs: PyTorch Geometric (para árboles/grafos).
# 5- Entregables
    1. Modelos entrenados:
        * Predictor de complejidad (CNN/Transformer).
        * Agente de RL para ordenamiento.
        * RNN/GNN para pilas/colas/árboles.
    2. Interfaz:
        * CLI o web (Flask) para subir código y recibir predicciones.
    3. Dataset sintético:
        * Ejemplos de código + complejidades, operaciones de estructuras.
    4. Documentación:
        * Explicación del modelo y resultados (ej: precisión en predicciones de Big-O)
# 6- Retos y Soluciones
    * Generar datos suficientes: Usar síntesis de código o scraping de GitHub.
    * Overfitting en predicción de Big-O: Aumentar datos con variaciones de código.
    * RL inestable: Empezar con algoritmos simples (ej: Insertion Sort).
#7- ¿Cómo Empezar?
    * Recolectar o generar datos (ej: 10k snippets de código con su Big-O).
    * Entrenar el predictor de complejidad (usar un subset de algoritmos conocidos).
    * Implementar el entorno de RL para ordenamiento.