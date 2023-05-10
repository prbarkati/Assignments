import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the neural network with a single input layer and output layer
model = Sequential()
model.add(Dense(16, input_dim=1, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model to classify odd/even numbers
X = np.array([i for i in range(1000)])  # Generate 1000 numbers from 0 to 999
Y = np.array([i % 2 for i in X])  # Label each number as 0 (even) or 1 (odd)
model.fit(X, Y, epochs=20, batch_size=10)

# Get a number from the user
num = int(input("Enter a number: "))

# Use the neural network to predict whether the number is odd or even
prediction = model.predict(np.array([num]))

# Print the result
if prediction < 0.5:
    print(f"{num} is even")
else:
    print(f"{num} is odd")
