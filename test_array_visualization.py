#!/usr/bin/env python3
"""
Test script to visualize the drawing to binary array conversion
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def visualize_array(array, title="Binary Array Visualization"):
    """Visualize a 28x28 binary array"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Create a grid visualization
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] == 1:
                # Draw filled square for 1s (black)
                rect = Rectangle((j, array.shape[0]-1-i), 1, 1,
                               facecolor='black', edgecolor='gray', linewidth=0.5)
                ax.add_patch(rect)
            else:
                # Draw empty square for 0s (white)
                rect = Rectangle((j, array.shape[0]-1-i), 1, 1,
                               facecolor='white', edgecolor='gray', linewidth=0.5)
                ax.add_patch(rect)

    ax.set_xlim(0, array.shape[1])
    ax.set_ylim(0, array.shape[0])
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(f'Columns (0-{array.shape[1]-1})')
    ax.set_ylabel(f'Rows (0-{array.shape[0]-1})')

    # Add grid
    ax.set_xticks(range(0, array.shape[1]+1, 4))
    ax.set_yticks(range(0, array.shape[0]+1, 4))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def test_sample_patterns():
    """Test with some sample patterns"""

    # Create a simple "1" pattern
    array_1 = np.zeros((28, 28), dtype=int)
    array_1[5:23, 13:15] = 1  # Vertical line
    array_1[5:8, 10:13] = 1   # Top diagonal

    # Create a simple "0" pattern
    array_0 = np.zeros((28, 28), dtype=int)
    # Outer circle
    for i in range(28):
        for j in range(28):
            center_i, center_j = 14, 14
            distance = np.sqrt((i - center_i)**2 + (j - center_j)**2)
            if 8 <= distance <= 10:
                array_0[i, j] = 1

    # Visualize both
    visualize_array(array_1, "Sample Pattern: Digit 1")
    visualize_array(array_0, "Sample Pattern: Digit 0")

    print("Sample 1 array:")
    print(array_1)
    print(f"\nNon-zero pixels in 1: {np.sum(array_1)}")

    print("\nSample 0 array:")
    print(array_0)
    print(f"Non-zero pixels in 0: {np.sum(array_0)}")

if __name__ == "__main__":
    test_sample_patterns()
