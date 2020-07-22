class Computer():
    def __init__(self, computer, ram, storage):
        self.computer = computer
        self.ram = ram
        self.storage = storage

# Class Mobile inherits Computer


class Mobile(Computer):
    def __init__(self, computer, ram, storage, model):
        super().__init__(computer, ram, storage)
        self.model = model


App = Mobile('Apple', 2, 64, 'iPhone X')
print('The mobile is:', App.computer)
print('The RAM is:', App.ram)
print('The storage is:', App.storage)
print('The model is:', App.model)
