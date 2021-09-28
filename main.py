import os

path = r'C:\Users\User\PycharmProjects\pythonProject2\Test'
files = os.listdir(path)
x = [i for i in files if i.find(".jpg") == -1]
x = [i[0] for i in x]
#print(files[0].find(".jpg"))

print(x)