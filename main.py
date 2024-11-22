import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Папки проекта
input_folder = 'XLSX'
output_folder = 'PICTURES'

# Настройки
node_size = 5000  # Размер узлов на графе

correlation_method = 'pearson'  # Метод корреляции (Пирсон)
# correlation_method = 'kendall'  # Метод корреляции (Кендалла)
# correlation_method = 'spearman'  # Метод корреляции (Спирмена)

# Создаем папку для сохранения результатов, если её нет
os.makedirs(output_folder, exist_ok=True)

# Обработка каждого Excel-файла
for file_name in os.listdir(input_folder):
    if file_name.endswith('.xlsx'):
        file_path = os.path.join(input_folder, file_name)
        file_base_name = os.path.splitext(file_name)[0]

        # --- Работа с листом Corr_matrix ---
        sheet_corr = 'Corr_matrix'
        try:
            corr_matrix_excel = pd.read_excel(file_path, sheet_name=sheet_corr, index_col=0)
            corr_matrix_excel = corr_matrix_excel.fillna(0)

            # Восстанавливаем полную матрицу
            corr_matrix_full = corr_matrix_excel + corr_matrix_excel.T - np.diag(np.diag(corr_matrix_excel))

            # Построение графа по полной матрице
            G_excel = nx.Graph()
            for i in range(len(corr_matrix_full.columns)):
                for j in range(i + 1, len(corr_matrix_full.columns)):
                    weight = corr_matrix_full.iloc[i, j]
                    if abs(weight) > 0:
                        G_excel.add_edge(corr_matrix_full.columns[i], corr_matrix_full.columns[j], weight=weight)

            # Генерация единого layout
            pos = nx.circular_layout(G_excel)

            # Визуализация графа по Excel
            edge_colors = ['darkgreen' if G_excel[u][v]['weight'] > 0 else 'darkred' for u, v in G_excel.edges()]
            widths = [abs(G_excel[u][v]['weight']) * 6 for u, v in G_excel.edges()]
            alphas = [min(max(abs(G_excel[u][v]['weight']), 0), 1) for u, v in G_excel.edges()]

            plt.figure(figsize=(12, 12))
            nx.draw_networkx_nodes(G_excel, pos, node_size=node_size, node_color='skyblue')
            nx.draw_networkx_edges(G_excel, pos, edge_color=edge_colors, width=widths, alpha=alphas)
            nx.draw_networkx_labels(G_excel, pos, font_size=14, font_weight='bold')
            plt.title(f"Корреляционная сеть (Excel) сорта {file_base_name}", fontsize=16, fontweight='bold')
            plt.savefig(os.path.join(output_folder, f"{file_base_name}_corr_network_excel.png"), dpi=300)
            plt.close()

            # Визуализация сокращённой матрицы
            mask = np.triu(np.ones_like(corr_matrix_excel, dtype=bool))  # Скрываем верхнюю часть
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                corr_matrix_excel, annot=True, mask=mask, cmap='RdYlGn', center=0, fmt=".1f",
                cbar_kws={'label': 'Корреляция'}
            )
            plt.title(f"Корреляционная матрица (Excel) сорта {file_base_name}", fontsize=16, fontweight='bold')
            plt.savefig(os.path.join(output_folder, f"{file_base_name}_corr_matrix_excel.png"), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Ошибка при обработке листа Corr_matrix в файле {file_name}: {e}")

        # --- Работа с листом Data ---
        sheet_data = 'Data'
        try:
            # Считывание данных
            data = pd.read_excel(file_path, sheet_name=sheet_data, header=1)
            param_numbers = data.columns.tolist()

            # Расчёт корреляционной матрицы
            corr_matrix_data = data.corr(method=correlation_method)

            # Построение графа по данным
            G_data = nx.Graph()
            for i in range(len(corr_matrix_data.columns)):
                for j in range(i + 1, len(corr_matrix_data.columns)):
                    weight = corr_matrix_data.iloc[i, j]
                    if abs(weight) > 0.3:
                        G_data.add_edge(param_numbers[i], param_numbers[j], weight=weight)

            # Визуализация графа по данным
            edge_colors = ['darkgreen' if G_data[u][v]['weight'] > 0 else 'darkred' for u, v in G_data.edges()]
            widths = [abs(G_data[u][v]['weight']) * 6 for u, v in G_data.edges()]
            alphas = [min(max(abs(G_data[u][v]['weight']), 0), 1) for u, v in G_data.edges()]

            plt.figure(figsize=(12, 12))
            nx.draw_networkx_nodes(G_data, pos, node_size=node_size, node_color='skyblue')
            nx.draw_networkx_edges(G_data, pos, edge_color=edge_colors, width=widths, alpha=alphas)
            nx.draw_networkx_labels(G_data, pos, labels=dict(zip(param_numbers, param_numbers)), font_size=14, font_weight='bold')
            plt.title(f"Корреляционная сеть (метод Пирсона) сорта {file_base_name}", fontsize=16, fontweight='bold')
            plt.savefig(os.path.join(output_folder, f"{file_base_name}_corr_network_data.png"), dpi=300)
            plt.close()

            # Визуализация сокращённой матрицы по данным
            mask = np.triu(np.ones_like(corr_matrix_data, dtype=bool))  # Скрываем верхнюю часть
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                corr_matrix_data, annot=True, mask=mask, cmap='RdYlGn', center=0, fmt=".1f",
                cbar_kws={'label': 'Корреляция'}
            )
            plt.title(f"Корреляционная матрица (метод Пирсона) сорта {file_base_name}", fontsize=16, fontweight='bold')
            plt.savefig(os.path.join(output_folder, f"{file_base_name}_corr_matrix_data.png"), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Ошибка при обработке листа Data в файле {file_name}: {e}")
            