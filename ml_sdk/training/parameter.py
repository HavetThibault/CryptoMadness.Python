

class Parameter:
    def __init__(self, name, value):
        self._name = name
        self._value = value

    def get_name(self):
        return self._name

    def get_value(self):
        return self._value

    def __eq__(self, other):
        if type(other) is Parameter:
            return other.get_value() == self._value
        return False

