# Task Qb

# from myModule import myFunctions as myMod

# myMod.sayHello()
# myMod.sayHellos(5)

# Task Qe

# from myClass import MyClass

# myObjectX = MyClass()

# myObjectX.printPrivateVarable()
# print("Public variable is: " + str(myObjectX.publicVariable))
# # print("Private variable is: " + str(myObjectX.__privateVariable)) <-- doesnt work


# # just to show how the name mangling works
# print("Bypassing nameMandling to get private variable: " + str(myObjectX._MyClass__privateVariable))


# myObjectX = MyClass(5)
# myObjectY = MyClass(10)

# print("Instance variable is: " + str(myObjectX.instanceVariable))
# print("Instance variable is: " + str(myObjectY.instanceVariable))


# myObjectX = MyClass(5)
# myObjectY = MyClass(10)

# print(str(myObjectX))
# print(str(myObjectY))

from myModule import myFunctions as mf

mf.sayHello()
