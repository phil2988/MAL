class MyClass:
    __privateVariable = -10
    publicVariable = 420

    def __init__(self, value: int):
        self.instanceVariable = value

    def printPrivateVarable(self):
        print("Private variable is: " + str(self.__privateVariable))
    
    def printInstanceBasedVariable(self):
        print("Instance variable is: " + str(self.instanceVariable))

    def __str__(self):
        return f"From str method of MyClass: instance variable is {self.instanceVariable}, __privateVariable is {self.__privateVariable}"