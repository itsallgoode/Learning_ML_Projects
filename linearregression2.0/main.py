

# This is linear regression implemented from scratch
# I used this Youtube tutorial: https://youtu.be/VmbA0pi2cRQ?si=Qvrt2vtWFU3W4r-x
# I also added in early stopping so it stops training when it detects no further improvements

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')


def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].YearsExperience
        y = points.iloc[i].Salary
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))


def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].YearsExperience
        y = points.iloc[i].Salary

        m_gradient += -(2 / n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2 / n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b


m = 0
b = 0
L = 0.001
epochs = 100000
patience = 10
best_loss = float('inf')
patience_counter = 0

for i in range(epochs):
    if i % 50 == 0:
        print(f"Epoch: {i}")
    m, b = gradient_descent(m, b, data, L)

    current_loss = loss_function(m, b, data)

    if current_loss < best_loss:
        best_loss = current_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping")
        break

print(m, b)

plt.scatter(data.YearsExperience, data.Salary, color="black")
plt.plot(list(range(0, 11)), [m * x + b for x in range(0, 11)], color="red")
plt.show()
