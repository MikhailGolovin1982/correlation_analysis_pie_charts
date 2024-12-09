import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Настройки
input_folder = 'XLSX'
output_folder = 'PICTURES'
node_size = 5000  # Размер узлов на графе
font_size_nodes = 14  # Размер шрифта для узлов
correlation_method = 'pearson'  # Метод корреляции (Пирсон)
r2_threshold = 0.3  # Пороговое значение для включения рёбер на графе R^2
node_order = 'clockwise'  # 'clockwise' или 'counterclockwise'

# Создаем папку для сохранения результатов, если её нет
os.makedirs(output_folder, exist_ok=True)

# Функция для перестановки столбцов
def rearrange_columns(file_path, sheet_name='Data'):
    """
    Функция для перестановки столбцов на листе Data в Excel файле,
    оставляя вторую строку (индексация начинается с 0) неизменной.
    """
    column_mapping = {
        7: 21,  # H -> V
        8: 22,  # I -> W
        9: 19,  # J -> T
        10: 20,  # K -> U
        11: 17,  # L -> R
        12: 18,  # M -> S
        13: 15,  # N -> P
        14: 16,  # O -> Q
        15: 13,  # P -> N
        16: 14,  # Q -> O
        17: 11,  # R -> L
        18: 12,  # S -> M
        19: 9,   # T -> J
        20: 10,  # U -> K
        21: 7,   # V -> H
        22: 8    # W -> I
    }

    try:
        # Загружаем файл Excel
        excel_data = pd.ExcelFile(file_path)

        # Проверяем, есть ли лист Data
        if sheet_name not in excel_data.sheet_names:
            print(f"Лист {sheet_name} отсутствует в файле {file_path}")
            return

        # Считываем данные с листа Data
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

        # Сохраняем вторую строку
        second_row = df.iloc[1, :].copy()

        # Переставляем столбцы
        column_order = df.columns.tolist()
        new_order = [column_mapping.get(col, col) for col in column_order]
        df = df.iloc[:, new_order]

        # Восстанавливаем вторую строку
        df.iloc[1, :] = second_row.values

        # Сохраняем файл с изменениями
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

        print(f"Файл {file_path} успешно обновлён.")

    except Exception as e:
        print(f"Ошибка при обработке файла {file_path}: {e}")


# Перебираем файлы в папке XLSX и применяем изменения
for file_name in os.listdir(input_folder):
    if file_name.endswith('.xlsx'):
        file_path = os.path.join(input_folder, file_name)
        rearrange_columns(file_path)

# --- Основной функционал проекта ---
for file_name in os.listdir(input_folder):
    if file_name.endswith('.xlsx'):
        file_path = os.path.join(input_folder, file_name)
        file_base_name = os.path.splitext(file_name)[0]

        # --- Работа с листом Data ---
        sheet_data = 'Data'
        try:
            # Считывание данных
            data = pd.read_excel(file_path, sheet_name=sheet_data, header=1)
            param_numbers = data.columns.tolist()

            # Расчёт корреляционной матрицы
            corr_matrix_data = data.corr(method=correlation_method)

            # Построение графа по корреляционной матрице
            G_corr = nx.Graph()

            # Добавляем все узлы
            for node in param_numbers:
                G_corr.add_node(node)

            # Добавляем рёбра
            for i in range(len(corr_matrix_data.columns)):
                for j in range(i + 1, len(corr_matrix_data.columns)):
                    weight = corr_matrix_data.iloc[i, j]
                    if abs(weight) > 0.3:
                        G_corr.add_edge(param_numbers[i], param_numbers[j], weight=weight)

            # Упорядочение узлов
            sorted_nodes = sorted(param_numbers)
            if node_order == 'counterclockwise':
                sorted_nodes = sorted_nodes[::-1]

            # Генерация layout
            pos = nx.circular_layout(G_corr)
            if node_order == 'counterclockwise':
                pos = {node: (-x, y) for node, (x, y) in pos.items()}

            # Визуализация графа корреляции
            edge_colors = ['darkgreen' if G_corr[u][v]['weight'] > 0 else 'darkred' for u, v in G_corr.edges()]
            widths = [abs(G_corr[u][v]['weight']) * 6 for u, v in G_corr.edges()]
            alphas = [min(max(abs(G_corr[u][v]['weight']), 0), 1) for u, v in G_corr.edges()]

            plt.figure(figsize=(12, 12))
            nx.draw_networkx_nodes(G_corr, pos, node_size=node_size, node_color='skyblue')
            nx.draw_networkx_edges(G_corr, pos, edge_color=edge_colors, width=widths, alpha=alphas)
            nx.draw_networkx_labels(
                G_corr, pos, labels=dict(zip(sorted_nodes, sorted_nodes)),
                font_size=font_size_nodes, font_weight='bold'
            )
            plt.title(f"Корреляционная сеть (метод Пирсона) сорта {file_base_name}", fontsize=16, fontweight='bold')
            plt.savefig(os.path.join(output_folder, f"{file_base_name}_corr_network_data.png"), dpi=300)
            plt.close()

            # Визуализация корреляционной матрицы
            mask = np.triu(np.ones_like(corr_matrix_data, dtype=bool))  # Скрываем верхнюю часть
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                corr_matrix_data, annot=True, mask=mask, cmap='RdYlGn', center=0, fmt=".1f",
                cbar_kws={'label': 'Корреляция'}
            )
            plt.title(f"Корреляционная матрица (метод Пирсона) сорта {file_base_name}", fontsize=16, fontweight='bold')
            plt.savefig(os.path.join(output_folder, f"{file_base_name}_corr_matrix_data.png"), dpi=300)
            plt.close()

            # --- Расчёт и визуализация R^2 ---
            r_squared_matrix = corr_matrix_data**2

            # Построение графа по R^2
            G_r2 = nx.Graph()

            # Добавляем все узлы
            for node in sorted_nodes:
                G_r2.add_node(node)

            # Добавляем рёбра с весами R^2
            for i in range(len(r_squared_matrix.columns)):
                for j in range(i + 1, len(r_squared_matrix.columns)):
                    r2_value = r_squared_matrix.iloc[i, j]
                    if r2_value >= r2_threshold:
                        G_r2.add_edge(sorted_nodes[i], sorted_nodes[j], weight=r2_value)

            # Визуализация графа R^2
            edge_colors_r2 = ['darkgreen' if G_r2[u][v]['weight'] > 0 else 'darkred' for u, v in G_r2.edges()]
            widths_r2 = [G_r2[u][v]['weight'] * 10 for u, v in G_r2.edges()]

            plt.figure(figsize=(12, 12))
            nx.draw_networkx_nodes(G_r2, pos, node_size=node_size, node_color='skyblue')
            nx.draw_networkx_edges(G_r2, pos, edge_color=edge_colors_r2, width=widths_r2)
            nx.draw_networkx_labels(
                G_r2, pos, labels=dict(zip(sorted_nodes, sorted_nodes)),
                font_size=font_size_nodes, font_weight='bold'
            )
            plt.title(f"Корреляционная сеть R^2 (метод Пирсона) сорта {file_base_name}", fontsize=16, fontweight='bold')
            plt.savefig(os.path.join(output_folder, f"{file_base_name}_corr_network_r2.png"), dpi=300)
            plt.close()

            # Визуализация матрицы R^2
            mask = np.triu(np.ones_like(r_squared_matrix, dtype=bool))  # Скрываем верхнюю часть
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                r_squared_matrix, annot=True, mask=mask, cmap='RdYlGn', center=0, fmt=".2f",
                cbar_kws={'label': 'Коэффициент детерминации (R^2)'}
            )
            plt.title(f"Корреляционная матрица R^2 (метод Пирсона) сорта {file_base_name}", fontsize=16, fontweight='bold')
            plt.savefig(os.path.join(output_folder, f"{file_base_name}_corr_matrix_r2.png"), dpi=300)
            plt.close()

        except Exception as e:
            print(f"Ошибка при обработке файла {file_name}: {e}")
