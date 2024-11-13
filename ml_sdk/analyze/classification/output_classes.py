class OutputClasses:
    def __init__(self, main: str, classes: list[str]):
        self._classes = classes
        self._main = main

    def get_classes(self):
        return self._classes

    def get_main(self):
        return self._main
