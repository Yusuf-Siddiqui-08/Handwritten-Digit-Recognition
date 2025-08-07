# Handwritten Digit Recognition

A Python application for recognizing handwritten digits using a custom pattern-matching algorithm and interactive feedback learning. The program features a Tkinter-based GUI for drawing digits, visualizing recognition, and managing learned patterns.

## Features

- **Draw Digits**: Use the mouse to draw digits (0-9) on a 28x28 grid canvas.
- **Digit Prediction**: Predicts the drawn digit using a custom similarity-based recognizer (no neural networks required).
- **Confidence Display**: Shows a confidence score for each prediction.
- **Feedback Learning**:
  - **Positive Feedback**: Confirm correct predictions to improve recognition.
  - **Negative Feedback**: Correct wrong predictions and teach the recognizer the right digit.
  - **Undo Feedback**: Undo the last feedback action (positive or negative).
- **Pattern Management**:
  - **Pattern Storage**: All learned patterns are saved in `learned_patterns.json`.
  - **Duplicate Removal**: Remove duplicate patterns to keep the dataset clean.
  - **Pattern Count**: View the number of learned patterns per digit and in total.
- **Debug Tools**:
  - **Array Visualization**: Visualize the 28x28 binary array of your drawing and the preprocessed version.
  - **System Info**: View pattern statistics and feedback dropdown state.
- **Safe File Handling**: Automatic backup and error handling for pattern files.

## How It Works

1. **Drawing**: Draw a digit on the canvas. The program converts your drawing into a 28x28 binary array.
2. **Prediction**: The recognizer compares your drawing to all learned patterns using a combination of pixel overlap, coverage, precision, and structural similarity.
3. **Feedback**: You can confirm or correct the prediction. Each feedback action adds a new pattern to the recognizer, improving its accuracy over time.
4. **Learning**: The recognizer uses only learned patterns (no preloaded MNIST data), so it adapts to your handwriting style.

## Getting Started

### Requirements
- Python 3.7+
- `numpy`
- `matplotlib`

Install dependencies with:
```bash
pip install numpy matplotlib
```

### Running the Program

1. Run the main script:
   ```bash
   python main.py
   ```
2. The GUI window will open. Draw a digit and click **Predict**.
3. Use the feedback options to help the recognizer learn.

## File Overview

- `main.py`: The main GUI application.
- `DigitRecognizer.py`: The core recognition and learning logic.
- `learned_patterns.json`: Stores all learned digit patterns.
- `learned_patterns.json.backup`: Automatic backup of learned patterns.

## Advanced Features

- **Debug Window**: Click the **Debug** button to open advanced tools:
  - Visualize arrays.
  - Remove duplicate patterns.
  - View per-digit pattern counts.
- **Pattern Preprocessing**: Smart centering and gentle scaling of drawings for robust recognition.
- **Safe Saving**: Patterns are saved with backup and error handling to prevent data loss.

## Tips

- The recognizer starts with no patterns. The more feedback you provide, the better it gets!
- Use the **Undo** button if you make a mistake when giving feedback.
- Use the **Debug** window to clean up duplicates and monitor learning progress.

## License

This project is provided for educational and personal use. No external datasets or neural networks are required.

