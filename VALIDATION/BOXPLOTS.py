import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Cargar CSV
csv_path = r"C:\Users\carme\OneDrive\Escritorio\M.UCM\TFM\VALIDATION\metricas_globales.csv"
df = pd.read_csv(csv_path)

print(df.head())



df_long = df.melt(id_vars=["Imagen","Clase"],
                  value_vars=["Dice","Recall"],
                  var_name="Metrica",
                  value_name="Valor")

plt.figure(figsize=(7,6))

# Colores cajas
palette_box = {"Tumor":"lightcoral", "Musculo":"lightgreen"}

# Colores puntos
palette_points = {"Tumor":"darkred", "Musculo":"darkgreen"}

sns.boxplot(
    data=df_long,
    x="Metrica",
    y="Valor",
    hue="Clase",
    palette=palette_box
)

sns.stripplot(
    data=df_long,
    x="Metrica",
    y="Valor",
    hue="Clase",
    palette=palette_points,
    dodge=True,
    size=7,
    alpha=0.8
)


handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[:2], labels[:2], title="Clase")

plt.ylabel("Valor de la métrica")
plt.title("Distribución de métricas a nivel de slide")

plt.show()




# # Cargar CSV
# csv_path = r"C:\Users\carme\OneDrive\Escritorio\M.UCM\TFM\metricas_globales.csv"
# df = pd.read_csv(csv_path)

# print(df.head())



# df_long = df.melt(id_vars=["Imagen","Clase"],
#                   value_vars=["Dice","Recall","Precision"],
#                   var_name="Metrica",
#                   value_name="Valor")

# plt.figure(figsize=(8,6))

# # Colores claros para cajas
# palette_box = {"Tumor":"lightcoral", "Musculo":"lightgreen"}

# # Colores oscuros para puntos
# palette_points = {"Tumor":"darkred", "Musculo":"darkgreen"}

# # Boxplot
# sns.boxplot(
#     data=df_long,
#     x="Metrica",
#     y="Valor",
#     hue="Clase",
#     palette=palette_box
# )

# # Puntos
# sns.stripplot(
#     data=df_long,
#     x="Metrica",
#     y="Valor",
#     hue="Clase",
#     palette=palette_points,
#     dodge=True,
#     size=7,
#     alpha=0.8
# )

# # quitar leyenda duplicada
# handles, labels = plt.gca().get_legend_handles_labels()
# plt.legend(handles[:2], labels[:2], title="Clase")

# plt.ylabel("Valor de la métrica")
# plt.title("Distribución de métricas (Dice, Recall y Precision) por clase")

# plt.show()