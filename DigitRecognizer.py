import numpy as np
import json
import os


class DigitRecognizer:
    """
    A class that handles digit recognition, pattern matching, and learning functionality.
    This class is separate from the GUI and focuses purely on the recognition logic.
    """

    def __init__(self):
        # Grid settings for pattern matching
        self.grid_size = 14  # 14x14 grid for pattern matching

        # File for storing learned patterns
        self.learned_patterns_file = "learned_patterns.json"

        # Initialize digit patterns
        self.digit_patterns = self.create_digit_patterns()

        # Load learned patterns
        self.learned_patterns = self.load_learned_patterns()

    def preprocess_drawing(self, array):
        """Preprocess the drawing array to center and scale it appropriately"""
        # Find the bounding box of the drawn pixels
        rows, cols = np.where(array > 0)

        if len(rows) == 0:
            return array  # Empty array

        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()

        # Extract the drawing within its bounding box
        drawing_height = max_row - min_row + 1
        drawing_width = max_col - min_col + 1

        # Calculate if the drawing is too small relative to canvas
        canvas_utilization = (drawing_height * drawing_width) / (self.grid_size * self.grid_size)

        # If drawing uses less than 25% of the canvas, apply scaling and centering
        if canvas_utilization < 0.25 or drawing_height < self.grid_size * 0.5 or drawing_width < self.grid_size * 0.5:
            return self.center_and_scale_drawing(array, min_row, max_row, min_col, max_col)

        # If drawing is reasonably sized, center it if it's off-center
        center_row = (min_row + max_row) / 2
        center_col = (min_col + max_col) / 2
        canvas_center = self.grid_size / 2

        # If drawing is significantly off-center, center it
        if abs(center_row - canvas_center) > 2 or abs(center_col - canvas_center) > 2:
            return self.center_drawing(array, min_row, max_row, min_col, max_col)

        return array

    def center_and_scale_drawing(self, array, min_row, max_row, min_col, max_col):
        """Center and scale up a small drawing to better fill the canvas"""
        # Extract the drawing
        drawing_height = max_row - min_row + 1
        drawing_width = max_col - min_col + 1

        # Calculate scaling factor to make drawing fill more of the canvas
        # Target the drawing to use about 70% of canvas size
        target_size = int(self.grid_size * 0.7)
        scale_factor = min(target_size / drawing_height, target_size / drawing_width)
        scale_factor = min(scale_factor, 3.0)  # Cap scaling to prevent artifacts

        if scale_factor <= 1.0:
            # If no scaling needed, center
            return self.center_drawing(array, min_row, max_row, min_col, max_col)

        # Create new array
        new_array = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # Calculate new dimensions
        new_height = int(drawing_height * scale_factor)
        new_width = int(drawing_width * scale_factor)

        # Calculate center position
        start_row = max(0, (self.grid_size - new_height) // 2)
        start_col = max(0, (self.grid_size - new_width) // 2)

        # Scale the drawing using nearest neighbor interpolation
        for i in range(new_height):
            for j in range(new_width):
                orig_row = min_row + int(i / scale_factor)
                orig_col = min_col + int(j / scale_factor)

                if (start_row + i < self.grid_size and start_col + j < self.grid_size and
                    orig_row <= max_row and orig_col <= max_col):
                    new_array[start_row + i, start_col + j] = array[orig_row, orig_col]

        return new_array

    def center_drawing(self, array, min_row, max_row, min_col, max_col):
        """Center the drawing without scaling"""
        drawing_height = max_row - min_row + 1
        drawing_width = max_col - min_col + 1

        # Calculate center position
        start_row = max(0, (self.grid_size - drawing_height) // 2)
        start_col = max(0, (self.grid_size - drawing_width) // 2)

        # Create new centered array
        new_array = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # Copy the drawing to center position
        for i in range(drawing_height):
            for j in range(drawing_width):
                if (start_row + i < self.grid_size and start_col + j < self.grid_size):
                    new_array[start_row + i, start_col + j] = array[min_row + i, min_col + j]

        return new_array

    def create_digit_patterns(self):
        """Create predefined patterns for digits 0-9"""
        patterns = {}

        # Digit 0 patterns
        patterns[0] = [
            # Pattern 1: Oval shape
            np.array([
                [0,0,1,1,1,1,1,1,1,1,1,1,0,0],
                [0,1,1,1,1,1,1,1,1,1,1,1,1,0],
                [1,1,1,0,0,0,0,0,0,0,0,1,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,1,0,0,0,0,0,0,0,0,1,1,1],
                [0,1,1,1,1,1,1,1,1,1,1,1,1,0],
                [0,0,1,1,1,1,1,1,1,1,1,1,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            ])
        ]

        # Digit 1 patterns
        patterns[1] = [
            # Pattern 1: Straight line
            np.array([
                [0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,1,1,1,0,0,0,0,0,0],
                [0,0,0,0,1,1,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                [0,1,1,1,1,1,1,1,1,1,1,1,1,0],
                [0,1,1,1,1,1,1,1,1,1,1,1,1,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            ]),
            # Pattern 2: Simple vertical line
            np.array([
                [0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            ])
        ]

        # Digit 2 patterns
        patterns[2] = [
            np.array([
                [0,0,1,1,1,1,1,1,1,1,1,1,0,0],
                [0,1,1,1,1,1,1,1,1,1,1,1,1,0],
                [1,1,1,0,0,0,0,0,0,0,0,1,1,1],
                [0,0,0,0,0,0,0,0,0,0,0,1,1,1],
                [0,0,0,0,0,0,0,0,0,0,1,1,1,0],
                [0,0,0,0,0,0,0,0,0,1,1,1,0,0],
                [0,0,0,0,0,0,0,1,1,1,1,0,0,0],
                [0,0,0,0,0,1,1,1,1,0,0,0,0,0],
                [0,0,0,1,1,1,1,0,0,0,0,0,0,0],
                [0,0,1,1,1,0,0,0,0,0,0,0,0,0],
                [0,1,1,1,0,0,0,0,0,0,0,0,0,0],
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            ])
        ]

        # Digit 3 patterns
        patterns[3] = [
            np.array([
                [0,1,1,1,1,1,1,1,1,1,1,1,1,0],
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,0,0,0,0,0,0,0,0,1,1,1],
                [0,0,0,0,0,0,0,0,0,0,0,1,1,1],
                [0,0,0,0,0,0,0,0,0,0,1,1,1,0],
                [0,0,0,0,1,1,1,1,1,1,1,1,0,0],
                [0,0,0,0,1,1,1,1,1,1,1,1,1,0],
                [0,0,0,0,0,0,0,0,0,0,0,1,1,1],
                [0,0,0,0,0,0,0,0,0,0,0,1,1,1],
                [0,0,0,0,0,0,0,0,0,0,0,1,1,1],
                [1,1,1,0,0,0,0,0,0,0,0,1,1,1],
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,1,1,1,1,1,1,1,1,1,1,1,1,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            ])
        ]

        # Digit 4 patterns
        patterns[4] = [
            np.array([
                [0,0,0,0,0,0,0,1,1,1,0,0,0,0],
                [0,0,0,0,0,0,1,1,1,1,0,0,0,0],
                [0,0,0,0,0,1,1,1,1,1,0,0,0,0],
                [0,0,0,0,1,1,1,0,1,1,0,0,0,0],
                [0,0,0,1,1,1,0,0,1,1,0,0,0,0],
                [0,0,1,1,1,0,0,0,1,1,0,0,0,0],
                [0,1,1,1,0,0,0,0,1,1,0,0,0,0],
                [1,1,1,0,0,0,0,0,1,1,0,0,0,0],
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,0,0,0,0,0,1,1,0,0,0,0],
                [0,0,0,0,0,0,0,0,1,1,0,0,0,0],
                [0,0,0,0,0,0,0,0,1,1,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            ])
        ]

        # Digit 5 patterns
        patterns[5] = [
            np.array([
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [1,1,1,0,0,0,0,0,0,0,0,0,0,0],
                [1,1,1,0,0,0,0,0,0,0,0,0,0,0],
                [1,1,1,0,0,0,0,0,0,0,0,0,0,0],
                [1,1,1,1,1,1,1,1,1,1,1,1,0,0],
                [1,1,1,1,1,1,1,1,1,1,1,1,1,0],
                [0,0,0,0,0,0,0,0,0,0,0,1,1,1],
                [0,0,0,0,0,0,0,0,0,0,0,1,1,1],
                [0,0,0,0,0,0,0,0,0,0,0,1,1,1],
                [1,1,1,0,0,0,0,0,0,0,0,1,1,1],
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,1,1,1,1,1,1,1,1,1,1,1,1,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            ])
        ]

        # Digit 6 patterns
        patterns[6] = [
            np.array([
                [0,0,1,1,1,1,1,1,1,1,1,1,0,0],
                [0,1,1,1,1,1,1,1,1,1,1,1,1,0],
                [1,1,1,0,0,0,0,0,0,0,0,0,0,0],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                [1,1,0,1,1,1,1,1,1,1,1,1,0,0],
                [1,1,1,1,1,1,1,1,1,1,1,1,1,0],
                [1,1,1,0,0,0,0,0,0,0,0,1,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,1,0,0,0,0,0,0,0,0,1,1,1],
                [0,1,1,1,1,1,1,1,1,1,1,1,1,0],
                [0,0,1,1,1,1,1,1,1,1,1,1,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            ])
        ]

        # Digit 7 patterns
        patterns[7] = [
            # Pattern 1: Simple 7
            np.array([
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,0,0,0,0,0,0,0,0,1,1,1],
                [0,0,0,0,0,0,0,0,0,0,1,1,1,0],
                [0,0,0,0,0,0,0,0,0,1,1,1,0,0],
                [0,0,0,0,0,0,0,0,1,1,1,0,0,0],
                [0,0,0,0,0,0,0,1,1,1,0,0,0,0],
                [0,0,0,0,0,0,1,1,1,0,0,0,0,0],
                [0,0,0,0,0,1,1,1,0,0,0,0,0,0],
                [0,0,0,0,1,1,1,0,0,0,0,0,0,0],
                [0,0,0,1,1,1,0,0,0,0,0,0,0,0],
                [0,0,1,1,1,0,0,0,0,0,0,0,0,0],
                [0,1,1,1,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            ]),
            # Pattern 2: 7 with cross
            np.array([
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,0,0,0,0,0,0,0,0,1,1,1],
                [0,0,0,0,0,0,0,0,0,0,1,1,1,0],
                [0,0,0,0,0,0,0,0,0,1,1,1,0,0],
                [0,0,0,0,0,0,0,0,1,1,1,0,0,0],
                [0,0,0,1,1,1,1,1,1,1,0,0,0,0],
                [0,0,0,0,0,0,1,1,1,0,0,0,0,0],
                [0,0,0,0,0,1,1,1,0,0,0,0,0,0],
                [0,0,0,0,1,1,1,0,0,0,0,0,0,0],
                [0,0,0,1,1,1,0,0,0,0,0,0,0,0],
                [0,0,1,1,1,0,0,0,0,0,0,0,0,0],
                [0,1,1,1,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            ])
        ]

        # Digit 8 patterns
        patterns[8] = [
            np.array([
                [0,0,1,1,1,1,1,1,1,1,1,1,0,0],
                [0,1,1,1,1,1,1,1,1,1,1,1,1,0],
                [1,1,1,0,0,0,0,0,0,0,0,1,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,1,0,0,0,0,0,0,0,0,1,1,1],
                [0,1,1,1,1,1,1,1,1,1,1,1,1,0],
                [0,1,1,1,1,1,1,1,1,1,1,1,1,0],
                [1,1,1,0,0,0,0,0,0,0,0,1,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,1,0,0,0,0,0,0,0,0,1,1,1],
                [0,1,1,1,1,1,1,1,1,1,1,1,1,0],
                [0,0,1,1,1,1,1,1,1,1,1,1,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            ])
        ]

        # Digit 9 patterns
        patterns[9] = [
            np.array([
                [0,0,1,1,1,1,1,1,1,1,1,1,0,0],
                [0,1,1,1,1,1,1,1,1,1,1,1,1,0],
                [1,1,1,0,0,0,0,0,0,0,0,1,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,1,0,0,0,0,0,0,0,0,1,1,1],
                [0,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,0,1,1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                [1,1,1,0,0,0,0,0,0,0,0,1,1,1],
                [0,1,1,1,1,1,1,1,1,1,1,1,1,0],
                [0,0,1,1,1,1,1,1,1,1,1,1,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            ])
        ]

        return patterns

    def calculate_similarity(self, drawn_array, pattern):
        """Calculate similarity between drawn array and pattern using multiple metrics"""
        # Normalize both arrays to ensure they're binary
        drawn_binary = (drawn_array > 0).astype(int)
        pattern_binary = (pattern > 0).astype(int)

        # Calculate intersection over union (IoU)
        intersection = np.sum(drawn_binary & pattern_binary)
        union = np.sum(drawn_binary | pattern_binary)

        if union == 0:
            return 0.0

        iou = intersection / union

        # Calculate coverage (how much of the pattern is covered)
        pattern_pixels = np.sum(pattern_binary)
        if pattern_pixels == 0:
            return 0.0
        coverage = intersection / pattern_pixels

        # Calculate precision (how much of the drawing matches the pattern)
        drawn_pixels = np.sum(drawn_binary)
        if drawn_pixels == 0:
            return 0.0
        precision = intersection / drawn_pixels

        # Weighted combination of metrics
        similarity = 0.4 * iou + 0.3 * coverage + 0.3 * precision

        return similarity

    def predict_digit(self, drawn_array):
        """
        Predict the digit based on the drawing array.
        Returns a tuple of (predicted_digit, confidence, all_similarities)
        """
        try:
            # Preprocess the drawing
            processed_array = self.preprocess_drawing(drawn_array)

            best_digit = None
            best_similarity = 0.0
            similarities = {}

            # Compare with all patterns (both predefined and learned)
            for digit in range(10):
                max_similarity_for_digit = 0.0

                # Check predefined patterns
                if digit in self.digit_patterns:
                    for pattern in self.digit_patterns[digit]:
                        similarity = self.calculate_similarity(processed_array, pattern)
                        max_similarity_for_digit = max(max_similarity_for_digit, similarity)

                # Check learned patterns and give them higher weight
                digit_str = str(digit)
                if digit_str in self.learned_patterns:
                    for pattern_data in self.learned_patterns[digit_str]:
                        try:
                            learned_pattern = np.array(pattern_data['pattern'])
                            learned_similarity = self.calculate_similarity(processed_array, learned_pattern)
                            # Boost learned pattern similarity based on confidence and feedback
                            confidence_boost = 1.0 + (pattern_data.get('confidence', 1.0) * 0.2)
                            feedback_boost = 1.0 + (pattern_data.get('feedback_count', 1) * 0.1)
                            learned_similarity *= confidence_boost * feedback_boost
                            max_similarity_for_digit = max(max_similarity_for_digit, learned_similarity)
                        except (ValueError, KeyError) as e:
                            print(f"Error processing learned pattern for digit {digit}: {e}")
                            continue

                similarities[digit] = max_similarity_for_digit

                if max_similarity_for_digit > best_similarity:
                    best_similarity = max_similarity_for_digit
                    best_digit = digit

            return best_digit, best_similarity, similarities

        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0, {}

    def add_positive_feedback(self, drawn_array, predicted_digit):
        """Add a pattern as a positive example for the predicted digit"""
        try:
            digit = str(predicted_digit)

            # Initialize if first time for this digit
            if digit not in self.learned_patterns:
                self.learned_patterns[digit] = []

            # Add the current drawing as a learned pattern
            pattern_data = {
                'pattern': drawn_array.tolist(),
                'confidence': 1.0,
                'feedback_count': 1
            }

            self.learned_patterns[digit].append(pattern_data)
            return True

        except Exception as e:
            print(f"Positive feedback error: {e}")
            return False

    def add_negative_feedback(self, drawn_array, predicted_digit, correct_digit):
        """Add a pattern as a correction from predicted_digit to correct_digit"""
        try:
            digit_str = str(correct_digit)

            # Initialize if first time for this digit
            if digit_str not in self.learned_patterns:
                self.learned_patterns[digit_str] = []

            # Add the current drawing as a learned pattern with the correct label
            pattern_data = {
                'pattern': drawn_array.tolist(),
                'confidence': 1.0,
                'feedback_count': 1,
                'corrected_from': predicted_digit
            }

            self.learned_patterns[digit_str].append(pattern_data)
            return True

        except Exception as e:
            print(f"Negative feedback error: {e}")
            return False

    def remove_last_pattern(self, digit):
        """Remove the last pattern added for a digit (used for undo)"""
        try:
            digit_str = str(digit)
            if digit_str in self.learned_patterns and len(self.learned_patterns[digit_str]) > 0:
                self.learned_patterns[digit_str].pop()

                # If this was the only pattern for this digit, remove the digit entirely
                if len(self.learned_patterns[digit_str]) == 0:
                    del self.learned_patterns[digit_str]

                return True
            return False
        except Exception as e:
            print(f"Remove pattern error: {e}")
            return False

    def load_learned_patterns(self):
        """Load learned patterns from file with error handling"""
        try:
            if os.path.exists(self.learned_patterns_file):
                with open(self.learned_patterns_file, 'r') as f:
                    data = json.load(f)
                    # Validate the loaded data
                    if isinstance(data, dict):
                        # Convert patterns back to numpy arrays for processing
                        for digit, patterns in data.items():
                            if isinstance(patterns, list):
                                for pattern_data in patterns:
                                    if isinstance(pattern_data, dict) and 'pattern' in pattern_data:
                                        # Pattern data is valid
                                        continue
                                    else:
                                        print(f"Invalid pattern data for digit {digit}")
                        return data
                    else:
                        print("Invalid learned patterns file format")
                        return {}
            else:
                print("No learned patterns file found, starting fresh")
                return {}
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading learned patterns: {e}")
            # Try to load backup if available
            backup_file = self.learned_patterns_file + ".backup"
            if os.path.exists(backup_file):
                try:
                    with open(backup_file, 'r') as f:
                        print("Loading from backup file")
                        return json.load(f)
                except:
                    print("Backup file also corrupted")
            return {}
        except Exception as e:
            print(f"Unexpected error loading patterns: {e}")
            return {}

    def save_learned_patterns(self):
        """Save learned patterns to file with backup and error handling"""
        try:
            # Create backup of existing file
            if os.path.exists(self.learned_patterns_file):
                backup_file = self.learned_patterns_file + ".backup"
                try:
                    import shutil
                    shutil.copy2(self.learned_patterns_file, backup_file)
                except Exception as backup_error:
                    print(f"Warning: Could not create backup: {backup_error}")

            # Save current patterns with custom formatting
            with open(self.learned_patterns_file, 'w') as f:
                self._write_formatted_patterns(f, self.learned_patterns)

            print(f"Learned patterns saved successfully to {self.learned_patterns_file}")
            return True

        except (IOError, json.JSONEncodeError) as e:
            print(f"Failed to save learned patterns: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error saving patterns: {e}")
            return False

    def _write_formatted_patterns(self, file, patterns_dict):
        """Write patterns to file with readable array formatting"""
        file.write("{\n")

        digit_keys = sorted(patterns_dict.keys(), key=lambda x: int(x))

        for i, digit in enumerate(digit_keys):
            patterns_list = patterns_dict[digit]
            file.write(f'  "{digit}": [\n')

            for j, pattern_data in enumerate(patterns_list):
                file.write("    {\n")

                # Write non-pattern fields first
                for key, value in pattern_data.items():
                    if key != 'pattern':
                        if isinstance(value, str):
                            file.write(f'      "{key}": "{value}",\n')
                        else:
                            file.write(f'      "{key}": {value},\n')

                # Write pattern array with grid formatting
                file.write('      "pattern": [\n')
                pattern_array = pattern_data['pattern']

                for row_idx, row in enumerate(pattern_array):
                    file.write("        [")
                    row_str = ",".join(str(val) for val in row)
                    file.write(row_str)
                    if row_idx < len(pattern_array) - 1:
                        file.write("],\n")
                    else:
                        file.write("]\n")

                file.write("      ]\n")

                if j < len(patterns_list) - 1:
                    file.write("    },\n")
                else:
                    file.write("    }\n")

            if i < len(digit_keys) - 1:
                file.write("  ],\n")
            else:
                file.write("  ]\n")

        file.write("}\n")

    def get_pattern_count(self, digit=None):
        """Get the count of learned patterns for a digit or all digits"""
        if digit is not None:
            digit_str = str(digit)
            return len(self.learned_patterns.get(digit_str, []))
        else:
            total = 0
            for patterns in self.learned_patterns.values():
                total += len(patterns)
            return total
