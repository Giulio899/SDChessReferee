class Counter():

    def __init__(self):
        self.count=0

    def plus(self):
        self.count=self.count+1

    def get(self):
        return str(self.count)