import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

# Загрузка данных из Excel-файла
file_path = 'Бельская.xlsx'
sheet_name = 'Corr_matrix'
data = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0)

# Очистка данных от NaN значений и создание симметричной матрицы
data = data.fillna(0)
data = data + data.T - pd.DataFrame(np.diag(np.diag(data.values)), index=data.index, columns=data.columns)

# Создание графа
G = nx.Graph()
threshold = 0.3

# Добавление узлов и рёбер в граф на основе значений корреляции
for i in range(len(data.columns)):
    for j in range(i + 1, len(data.columns)):
        corr_value = data.iloc[i, j]
        if abs(corr_value) >= threshold:
            G.add_edge(data.columns[i], data.columns[j], weight=corr_value)

# Разделение всех названий на две строки для улучшения читаемости
#labels = {node: '\n'.join(node.split(', ', 1)) if ', ' in node else node for node in G.nodes()}
labels = {node: node for node in G.nodes()}

# Настройка цвета: зеленый для положительных, красный для отрицательных
cmap = sns.diverging_palette(10, 150, as_cmap=True)  # Палитра для матрицы
edge_colors = ['darkgreen' if G[u][v]['weight'] > 0 else 'darkred' for u, v in G.edges()]
widths = [abs(G[u][v]['weight']) * 6 for u, v in G.edges()]
alphas = [abs(G[u][v]['weight']) for u, v in G.edges()]  # Прозрачность линий в зависимости от корреляции

# --- Визуализация корреляционного графа ---
plt.figure(figsize=(24, 24))
pos = nx.circular_layout(G)
nx.draw_networkx_nodes(G, pos, node_size=8000, node_color='skyblue')
for (u, v, d), color, width, alpha in zip(G.edges(data=True), edge_colors, widths, alphas):
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, edge_color=color, alpha=alpha)

# Расчет позиций для подписей с использованием полярных координат
label_pos = {}
radius = 1.
for i, node in enumerate(G.nodes()):
    angle = 2 * np.pi * i / len(G.nodes())
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    label_pos[node] = (x, y)
nx.draw_networkx_labels(G, label_pos, labels=labels, font_size=12, font_family="sans-serif", font_weight="bold")

# Настройка заголовка и сохранение графа
plt.title("Корреляционная сеть", fontsize=18, y=0.87)
plt.margins(x=0.2, y=0.2)
plt.axis('off')
plt.savefig("correlation_network_graph.png", dpi=300, bbox_inches='tight', pad_inches=0.5)
plt.close()

# --- Визуализация половинной корреляционной матрицы ---
# Создание маски для верхней половины матрицы
mask = np.triu(np.ones_like(data, dtype=bool))

plt.figure(figsize=(12, 10))
sns.heatmap(data, annot=True, mask=mask, cmap=cmap, center=0, linewidths=0.5, fmt=".1f", cbar_kws={"shrink": .8})
plt.title("Корреляционная матрица")
plt.savefig("correlation_matrix_colored.png", dpi=300, bbox_inches='tight', pad_inches=0.5)
plt.close()
