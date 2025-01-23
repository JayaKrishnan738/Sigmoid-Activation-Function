# sigmoid.py
import numpy as np
def sigmoid(x):
    """
    Compute the sigmoid activation function.

    Parameters:
    x (float, int, list, or np.array): Input value(s)

    Returns:
    float or np.array: Sigmoid output in the range (0, 1)
    """
    # Ensure compatibility with scalars and arrays
    x = np.array(x)  # Convert input to a NumPy array if it isn't already
    return 1 / (1 + np.exp(-x))

def sigmoid_stable(x):
    """
    Compute the sigmoid activation function (numerically stable).

    Parameters:
    x (float, int, list, or np.array): Input value(s)

    Returns:
    float or np.array: Sigmoid output in the range (0, 1)
    """
    x = np.array(x)
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))

if __name__ == "__main__":
    user_input = input("Enter a number or a list of numbers (comma-separated): ")
    try:
        # Parse input to handle both single numbers and lists
        if "," in user_input:
            x = [float(i) for i in user_input.split(",")]
        else:
            x = float(user_input)

        # Compute sigmoid and stable sigmoid
        print("Sigmoid result:", sigmoid(x))
        print("Stable Sigmoid result:", sigmoid_stable(x))
    except ValueError:
        print("Invalid input. Please enter a valid number or list of numbers.")