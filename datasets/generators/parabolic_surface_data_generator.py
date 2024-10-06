import os.path
import csv
from datasets import file_paths
from datasets.generators.common import randomness

X_VALUES = list(range(-50, 51))
Y_VALUES = list(range(-50, 51))


def f(x: float, y: float) -> float:
    return x*x + y*y + x*y


def main():
    csv_file = os.path.abspath(file_paths.parabolic_surface)
    if os.path.exists(csv_file):
        os.remove(csv_file)

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'z'])

        for x in X_VALUES:
            for y in Y_VALUES:
                z = round(f(x, y) + randomness(0, 1), 2)
                writer.writerow([x, y, z])


if __name__ == '__main__':
    main()
