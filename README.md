# Sigmoid-Activation-Function

# README.md
"""
# Sigmoid Activation Function Implementation

This repository contains the implementation of the sigmoid activation function commonly used in machine learning and neural networks. The sigmoid function maps input values to a range between 0 and 1, making it suitable for probabilistic interpretation and logistic regression.

## The Role of the Sigmoid Function

### Probabilistic Interpretation
The sigmoid function outputs values in the range (0, 1), which can be interpreted as probabilities. For example, given an input \( x \), the output of the sigmoid function can represent the likelihood of a binary event occurring. This makes it particularly useful in models where the output needs to represent probabilities, such as classification tasks.

### Logistic Regression
In logistic regression, the sigmoid function is used to map linear combinations of input features (produced by a linear model) into a probability. The model predicts the probability of the dependent variable belonging to a certain class, typically class 1 in a binary classification problem. The decision boundary is determined by a threshold (e.g., 0.5), where outputs greater than or equal to the threshold are classified as class 1, and outputs below are classified as class 0.

The mathematical form of the sigmoid function ensures that extreme input values result in outputs very close to 0 or 1, stabilizing the predictions for extreme cases.

## Setup Instructions

1. Clone the repository:
    ```bash
    git clone <repository_url>
    ```

2. Navigate to the directory:
    ```bash
    cd sigmoid_function
    ```

3. Install the required libraries (if not already installed):
    ```bash
    pip install numpy
    ```

## Example Usage

### Running the Script

1. Run the script directly to input values:
    ```bash
    python sigmoid.py
    ```

2. Enter a single value or a list of comma-separated values when prompted. For example:
    ```
    Enter a number or a list of numbers (comma-separated): 1, -1, 0
    ```
    Output:
    ```
    Sigmoid result: [0.7310585786300049, 0.2689414213699951, 0.5]
    Stable Sigmoid result: [0.7310585786300049, 0.2689414213699951, 0.5]
    ```

### Importing the Functions

```python
from sigmoid import sigmoid, sigmoid_stable
```

### Compute Sigmoid for Single Values

```python
# Example: x = 0
result = sigmoid(0)
print("Sigmoid(0):", result)
# Output: Sigmoid(0): 0.5
```

### Compute Sigmoid for Arrays

```python
# Example: x = [1, -1, 0]
result = sigmoid([1, -1, 0])
print("Sigmoid([1, -1, 0]):", result)
# Output: [0.73105858, 0.26894142, 0.5]
```

### Compute Sigmoid with Numerical Stability

```python
# Example: Large Positive and Negative Values
result = sigmoid_stable([1000, -1000])
print("Sigmoid Stable([1000, -1000]):", result)
# Output: [1.0, 0.0]
```

### Multi-Dimensional Arrays

```python
# Example: Multi-dimensional array
result = sigmoid([[1, 2], [-1, -2]])
print("Sigmoid([[1, 2], [-1, -2]]):", result)
# Output: 
# [[0.73105858 0.88079708]
#  [0.26894142 0.11920292]]
```

## Contribution

Feel free to open an issue or submit a pull request if you find bugs or have suggestions for improvements.

## License

This project is licensed under the MIT License.

"""
