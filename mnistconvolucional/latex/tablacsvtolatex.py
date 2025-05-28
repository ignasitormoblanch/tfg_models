import pandas as pd
from tabulate import tabulate


def main():
    # Nombre del archivo CSV (asegúrate de que esté en la misma carpeta que este script)
    csv_file = "tablaperceptron - Full 1.csv"

    # Leer el CSV con pandas
    df = pd.read_csv(csv_file)

    # Convertir la tabla a formato LaTeX con tabulate
    latex_table = tabulate(df, headers='keys', tablefmt='latex', showindex=False)

    # Nombre del archivo de salida (puede ser .txt o .tex)
    output_file = "tabla_tfg.txt"

    # Escribir la tabla en el archivo de texto
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(latex_table)

    print("La tabla en formato LaTeX se ha guardado en el archivo:", output_file)


if __name__ == "__main__":
    main()
