class FloatCategoriesCalculator:
    def __init__(self, min, max, step):
        self.min = min
        self.max = max
        self.step = step
        self.categories = int((max - min) // step + 1)

    def get_categories_bounds(self) -> list[float]:
        categories_bounds = []
        for i in range(self.categories + 1):
            categories_bounds.append(self.min + i * self.step)
        return categories_bounds

    def get_category_center(self, category_index: int) -> float:
        return self.min + (category_index + 0.5) * self.step
