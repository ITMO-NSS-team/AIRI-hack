# coding: utf-8

from ase.db import connect

if __name__ == '__main__':

    
    database_path = "train.db"  # Путь к файлу базы данных
    database = connect(database_path)  # Загрузка базы данных
    row = database.get(1)  # Взятие первого элемента из базы. Нумерация начинается с 1
    n_atoms = row.natoms  # Количество атомов в молекуле
    numbers = row.numbers  # Атомные числа атомов молекулы
    symbols = row.symbols  # Названия атомов молекулы
    positions = row.positions  # Координаты атомов. np.array размера n_atoms x 3
    energy = row.data.get('energy') # Энергия. float значение энергии