import pandas as pd
from sklearn import linear_model, metrics, model_selection

url = "https://raw.githubusercontent.com/FallicoFunctions/Fish_Dataset/main/Fish%20Dataset"

names = ['Species', 'Weight (in grams)', 'Vertical Length in CM', 'Diagonal Length in CM', 'Cross Length in CM',
         'Height in CM', 'Diagonal Width in CM']

data_frame = pd.read_csv(url, names=names)

Y = data_frame.values[:, 0]
X = data_frame.values[:, 1:]

fish_model = linear_model.LogisticRegression(max_iter=10000000)

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.3)

fish_model.fit(x_train, y_train)

y_prediction = fish_model.predict(x_test)

print(metrics.accuracy_score(y_test, y_prediction))

while True:
    try:
        # value = eval(input("Enter four whole or decimal numbers, e.g. 1, 2.4, 3, 4.5:\n"))
        print("Welcome to the fish species identifier. You will need these six measurements to determine the species "
              "of fish you have:")
        print("Weight (in grams), Vertical Length in CM, Diagonal Length in CM, Cross Length in CM, Height in CM, "
              "Diagonal Width in CM")
        print("Please enter the measurements on the same line, each separated by a space.")
        first, second, third, fourth, fifth, sixth = input("Enter 6 whole or decimal numbers, e.g. 100, 21.4, 3, 4.5, 5, 6.1:\n").split(" ")
        first = float(first)
        second = float(second)
        third = float(third)
        fourth = float(fourth)
        fifth = float(fifth)
        sixth = float(sixth)
        break
    except NameError:
        print("Invalid input")

print(fish_model.predict([[first, second, third, fourth, fifth, sixth]]))
