import os
import csv
from typing import List

from datasets import file_paths

X_RANGE = list(range(-50, 51))
Y_RANGE = list(range(-50, 51))


def square(x: float) -> float:
    return x*x


class Circle:
    def __init__(self, x: float, y: float, r: float):
        self.x = x
        self.y = y
        self.r = r

    def is_inside(self, x, y) -> bool:
        return square(x-self.x) + square(y-self.y) <= square(self.r)


c1 = Circle(0, 0, 5)
c2 = Circle(10, 8, 8)
c3 = Circle(-1, -3, 4)


def get_classes(x: float, y: float) -> List[int]:
    classes = [0, 0, 0, 0]

    if not c1.is_inside(x, y) and \
        not c2.is_inside(x, y) and \
            not c2.is_inside(x, y):
        classes[0] = 1

    if c1.is_inside(x, y):
        classes[1] = 1

    if c2.is_inside(x, y):
        classes[2] = 1

    if c3.is_inside(x, y):
        classes[3] = 1

    return classes


def main():
    csv_file = os.path.abspath(file_paths.circle_classification)
    if os.path.exists(csv_file):
        os.remove(csv_file)

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'outside_circles', 'c1', 'c2', 'c3'])

        for x in X_RANGE:
            for y in Y_RANGE:
                classes = get_classes(x, y)
                writer.writerow([x, y, classes[0], classes[1], classes[2], classes[3]])


if __name__ == '__main__':
    main()
