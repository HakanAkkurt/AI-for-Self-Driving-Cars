import numpy as np

grocery_items = {'bananas': 2.99, 'apples': 1.59, 'ananas': 7.88}
item = 'brussel sprouts'
if item in grocery_items:
    print('found the', item)
else:
    print('could not find the', item)
    print(grocery_items)


grocery_items.update({item: 2.77})
print(grocery_items)


months = ['Januar', 'Ferbuar', 'März']
for month in months:
    print(month)

for number in range(0, 10):
    print(number)

movies = {'Titanic': 1997, 'Jurassic Park': 1995}

for (key, value) in movies.items():
    print('The movie \'{}\', was made in {}'.format(key, value))


movies = {'Titanic': 1997, 'Jurassic Park': 1995}

for (key, value) in movies.items():
    print(f'The movie {key}, was made in {value}!')


def rectangleArea(length, width):
    """comment"""
    return length * width


print(rectangleArea(5, 5))

numbers = [1, 2, 3, 4, 5, 6]

print(list(filter(lambda number: number % 2 == 0, numbers)))


listTwo = list(range(1, 4))
listThree = list(range(1, 4))
listSum = []

for index in range(3):
    listTwo[index] = listTwo[index] ** 2
    listThree[index] = listThree[index] ** 3
    listSum.append(listTwo[index] + listThree[index])
print(listSum)


arrayTwo = np.arange(1, 4) ** 2
arrayThree = np.arange(1, 4) ** 3
print(arrayTwo + arrayThree)

sampleArray = np.array([1, 2, 3])
print(np.power(sampleArray, 4))

print(np.negative(sampleArray))

print(np.exp(sampleArray))

print(np.log(sampleArray))

print(np.sin(sampleArray))

x = np.arange(3)
y = np.arange(3, 6)
z = np.arange(6, 9)

multiArray = np.array([x, y, z])
print(multiArray)
print(multiArray.shape)
print(multiArray > 5)
print(multiArray[multiArray > 5])
print(multiArray.max())
print(multiArray.ravel())

w = np.linspace(1, 49, 49)
print(w)
