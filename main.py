import tkinter as tk
import numpy as np
from DigitRecognizer import DigitRecognizer


class DigitRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognition")
        self.root.geometry("500x750")
        self.root.configure(bg='lightgray')

        # Canvas settings
        self.canvas_size = 280
        self.grid_size = 28  # 28x28 grid for pattern matching
        self.cell_size = self.canvas_size // self.grid_size

        # Drawing state
        self.drawing = False
        self.drawn_pixels = set()
        self.last_prediction = None
        self.last_drawn_array = None
        self.feedback_given = False  # Track if feedback has been provided
        self.feedback_type = None    # Track type of feedback ('positive' or 'negative')

        # Undo functionality state
        self.undo_available = False
        self.last_action_data = None  # Store data needed to undo the last action

        # Dropdown state for feedback
        self.feedback_dropdown_open = False

        # Initialize the digit recognizer
        self.recognizer = DigitRecognizer()

        # Create GUI elements
        self.setup_gui()

    def setup_gui(self):
        # Title
        title_label = tk.Label(self.root, text="Draw a Digit (0-9)",
                              font=("Arial", 16, "bold"), bg='lightgray')
        title_label.pack(pady=10)

        # Canvas frame
        canvas_frame = tk.Frame(self.root, bg='black', bd=2)
        canvas_frame.pack(pady=10)

        # Drawing canvas
        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_size, height=self.canvas_size,
                               bg='white', cursor='pencil')
        self.canvas.pack(padx=2, pady=2)

        # Bind mouse events for drawing
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        # Buttons frame
        button_frame = tk.Frame(self.root, bg='lightgray')
        button_frame.pack(pady=10)

        # Clear button
        clear_btn = tk.Button(button_frame, text="Clear", command=self.clear_canvas,
                             font=("Arial", 12), bg='lightcoral', width=15)
        clear_btn.pack(side="left", padx=10)

        # Predict button
        predict_btn = tk.Button(button_frame, text="Predict", command=self.predict_digit,
                               font=("Arial", 12), bg='lightgreen', width=15)
        predict_btn.pack(side="left", padx=10)

        # Prediction result frame
        result_frame = tk.Frame(self.root, bg='white', bd=2, relief='solid')
        result_frame.pack(pady=20, padx=50, fill='x')

        # Prediction label
        tk.Label(result_frame, text="Prediction:", font=("Arial", 12, "bold"),
                bg='white').pack(pady=5)

        self.prediction_label = tk.Label(result_frame, text="Draw a digit and click Predict",
                                        font=("Arial", 24, "bold"), bg='white', fg='blue')
        self.prediction_label.pack(pady=10)

        # Confidence label
        self.confidence_label = tk.Label(result_frame, text="",
                                        font=("Arial", 10), bg='white', fg='gray')
        self.confidence_label.pack(pady=5)

        # Feedback container frame (always visible when there's a prediction)
        self.feedback_container = tk.Frame(result_frame, bg='white')

        # Dropdown button for feedback (initially hidden)
        self.dropdown_btn = tk.Button(self.feedback_container, text="▼ Feedback Options",
                                     command=self.toggle_feedback_dropdown,
                                     font=("Arial", 12), bg='lightblue', width=20)
        self.dropdown_btn.pack(pady=5)

        # Feedback frame (collapsible content - initially hidden)
        self.feedback_frame = tk.Frame(self.feedback_container, bg='white')

        # Feedback question
        feedback_question = tk.Label(self.feedback_frame, text="Was this prediction correct?",
                                   font=("Arial", 12), bg='white', fg='black')
        feedback_question.pack(pady=(10, 5))

        # Feedback buttons frame
        feedback_buttons_frame = tk.Frame(self.feedback_frame, bg='white')
        feedback_buttons_frame.pack(pady=5)

        # Thumbs up button
        self.thumbs_up_btn = tk.Button(feedback_buttons_frame, text="👍 Correct",
                                      command=self.positive_feedback,
                                      font=("Arial", 12), bg='lightgreen',
                                      width=12, relief='raised')
        self.thumbs_up_btn.pack(side="left", padx=10)

        # Thumbs down button
        self.thumbs_down_btn = tk.Button(feedback_buttons_frame, text="👎 Wrong",
                                        command=self.negative_feedback,
                                        font=("Arial", 12), bg='lightcoral',
                                        width=12, relief='raised')
        self.thumbs_down_btn.pack(side="left", padx=10)

        # Undo button (initially hidden)
        self.undo_btn = tk.Button(feedback_buttons_frame, text="↶ Undo",
                                 command=self.undo_feedback,
                                 font=("Arial", 12), bg='lightyellow',
                                 width=12, relief='raised', state='disabled')
        self.undo_btn.pack(side="left", padx=10)

        # Feedback status label
        self.feedback_status_label = tk.Label(self.feedback_frame, text="",
                                             font=("Arial", 10), bg='white', fg='green')
        self.feedback_status_label.pack(pady=5)

        # Initially hide the feedback frame (dropdown content)
        # Don't pack the feedback_frame yet - it will be shown/hidden by dropdown

        # Don't pack the feedback_container yet - it will be shown when there's a prediction

    def start_drawing(self, event):
        self.drawing = True
        self.draw(event)

    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            # Draw a small circle
            radius = 8
            self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius,
                                   fill='black', outline='black')

            # Store drawn pixels for pattern matching
            grid_x = min(x // self.cell_size, self.grid_size - 1)
            grid_y = min(y // self.cell_size, self.grid_size - 1)
            self.drawn_pixels.add((grid_x, grid_y))

    def stop_drawing(self, event):
        self.drawing = False

    def clear_canvas(self):
        self.canvas.delete("all")
        self.drawn_pixels.clear()
        self.prediction_label.config(text="Draw a digit and click Predict")
        self.confidence_label.config(text="")
        self.hide_feedback_buttons()
        self.last_prediction = None
        self.last_drawn_array = None
        self.reset_feedback_state()  # Reset feedback state when clearing

    def reset_feedback_state(self):
        """Reset feedback state for new drawing"""
        self.feedback_given = False
        self.feedback_type = None
        self.undo_available = False
        self.last_action_data = None
        # Reset button appearance
        self.thumbs_up_btn.config(bg='lightgreen', state='normal', relief='raised')
        self.thumbs_down_btn.config(bg='lightcoral', state='normal', relief='raised')
        self.undo_btn.config(state='disabled', bg='lightgray')

    def get_drawing_array(self):
        """Convert drawing to 28x28 binary array"""
        array = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for x, y in self.drawn_pixels:
            array[y, x] = 1
        return array

    def predict_digit(self):
        """Predict the digit based on the drawing with comprehensive error handling"""
        try:
            # Check if canvas is empty
            if not self.drawn_pixels:
                self.prediction_label.config(text="����� Canvas is empty!", fg='red')
                self.confidence_label.config(text="Please draw a digit first")
                self.hide_feedback_buttons()
                return

            # Check if drawing is too small (less than 3 pixels)
            if len(self.drawn_pixels) < 3:
                self.prediction_label.config(text="❌ Drawing too small!", fg='red')
                self.confidence_label.config(text="Please draw a larger digit")
                self.hide_feedback_buttons()
                return

            drawn_array = self.get_drawing_array()
            self.last_drawn_array = drawn_array

            # Use the recognizer to predict
            best_digit, best_similarity, similarities = self.recognizer.predict_digit(drawn_array)

            # Store the prediction for feedback
            self.last_prediction = best_digit

            # Display result with better error handling
            if best_digit is not None and best_similarity > 0.05:  # Lowered threshold for better detection
                self.prediction_label.config(text=f"Predicted Digit: {best_digit}", fg='blue')
                confidence_percent = min(int(best_similarity * 100), 99)  # Cap at 99%
                self.confidence_label.config(text=f"Confidence: {confidence_percent}%", fg='gray')

                # Show feedback buttons only for valid predictions
                self.show_feedback_buttons()
            else:
                self.prediction_label.config(text="❌ Unable to recognize digit", fg='red')
                self.confidence_label.config(text="Try drawing more clearly or a different style", fg='gray')
                self.hide_feedback_buttons()

        except Exception as e:
            # Catch any unexpected errors
            self.prediction_label.config(text="❌ Error during prediction", fg='red')
            self.confidence_label.config(text=f"Error: {str(e)}", fg='gray')
            self.hide_feedback_buttons()
            print(f"Prediction error: {e}")

    def positive_feedback(self):
        """Handle positive feedback with error handling and prevent duplicate submissions"""
        try:
            # Check if feedback has already been given
            if self.feedback_given:
                self.feedback_status_label.config(
                    text=f"⚠️ Feedback already submitted ({self.feedback_type})", 
                    fg='orange'
                )
                return
            
            if self.last_prediction is not None and self.last_drawn_array is not None:
                # Store action data for undo before making changes
                self.last_action_data = {
                    'action_type': 'positive_feedback',
                    'digit': self.last_prediction,
                    'pattern_added': self.last_drawn_array.tolist(),
                }

                # Add the feedback to the recognizer
                success = self.recognizer.add_positive_feedback(self.last_drawn_array, self.last_prediction)

                if success:
                    learned_count = self.recognizer.get_pattern_count(self.last_prediction)
                    self.feedback_status_label.config(
                        text=f"✓ Saved! Now have {learned_count} examples of digit {self.last_prediction}",
                        fg='green'
                    )

                    # Update feedback state and button appearance
                    self.feedback_given = True
                    self.feedback_type = 'positive'
                    self.undo_available = True
                    self.update_feedback_buttons()

                    # Save to file with error handling
                    if not self.recognizer.save_learned_patterns():
                        self.feedback_status_label.config(
                            text="❌ Error saving pattern",
                            fg='red'
                        )
                else:
                    self.feedback_status_label.config(text="❌ Error adding positive feedback", fg='red')
            else:
                self.feedback_status_label.config(text="❌ No prediction to confirm.", fg='red')
        except Exception as e:
            self.feedback_status_label.config(text=f"❌ Error: {str(e)}", fg='red')
            print(f"Positive feedback error: {e}")

    def negative_feedback(self):
        """Handle negative feedback - ask user for correct digit"""
        try:
            # Check if feedback has already been given
            if self.feedback_given:
                self.feedback_status_label.config(
                    text=f"⚠️ Feedback already submitted ({self.feedback_type})",
                    fg='orange'
                )
                return

            if self.last_prediction is not None and self.last_drawn_array is not None:
                # Create a dialog to ask for the correct digit
                self.create_correction_dialog()
            else:
                self.feedback_status_label.config(text="❌ No prediction to correct.", fg='red')
        except Exception as e:
            self.feedback_status_label.config(text=f"❌ Error: {str(e)}", fg='red')

    def create_correction_dialog(self):
        """Create a dialog for the user to input the correct digit"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Correct the Prediction")
        dialog.geometry("400x300")
        dialog.configure(bg='lightgray')
        dialog.transient(self.root)
        dialog.grab_set()

        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

        # Instructions
        tk.Label(dialog, text=f"The prediction was: {self.last_prediction}",
                font=("Arial", 12), bg='lightgray').pack(pady=10)
        tk.Label(dialog, text="What is the correct digit?",
                font=("Arial", 12, "bold"), bg='lightgray').pack(pady=5)

        # Digit selection frame
        digit_frame = tk.Frame(dialog, bg='lightgray')
        digit_frame.pack(pady=15)

        # Create buttons for digits 0-9
        for i in range(10):
            btn = tk.Button(digit_frame, text=str(i), font=("Arial", 14, "bold"),
                           width=4, height=2, command=lambda d=i: self.correct_prediction(d, dialog))
            btn.grid(row=i//5, column=i%5, padx=8, pady=8)

        # Cancel button
        cancel_btn = tk.Button(dialog, text="Cancel", command=dialog.destroy,
                              font=("Arial", 12), bg='lightcoral', width=12)
        cancel_btn.pack(pady=20)

    def correct_prediction(self, correct_digit, dialog):
        """Save the corrected prediction as a learned pattern"""
        try:
            # Store action data for undo before making changes
            self.last_action_data = {
                'action_type': 'negative_feedback',
                'digit': correct_digit,
                'pattern_added': self.last_drawn_array.tolist(),
                'corrected_from': self.last_prediction,
            }

            # Add the correction to the recognizer
            success = self.recognizer.add_negative_feedback(
                self.last_drawn_array, self.last_prediction, correct_digit
            )

            if success:
                learned_count = self.recognizer.get_pattern_count(correct_digit)
                self.feedback_status_label.config(
                    text=f"✓ Corrected! Saved as digit {correct_digit}. Now have {learned_count} examples.",
                    fg='green'
                )

                # Update feedback state
                self.feedback_given = True
                self.feedback_type = 'negative'
                self.undo_available = True
                self.update_feedback_buttons()

                # Save to file
                if not self.recognizer.save_learned_patterns():
                    self.feedback_status_label.config(
                        text="❌ Error saving correction",
                        fg='red'
                    )
            else:
                self.feedback_status_label.config(text="❌ Error adding correction", fg='red')

            dialog.destroy()

        except Exception as e:
            self.feedback_status_label.config(text=f"❌ Error: {str(e)}", fg='red')
            dialog.destroy()

    def update_feedback_buttons(self):
        """Update the appearance of feedback buttons based on feedback state"""
        if self.feedback_given:
            if self.feedback_type == 'positive':
                self.thumbs_up_btn.config(bg='darkgreen', state='disabled', relief='sunken')
                self.thumbs_down_btn.config(bg='lightgray', state='disabled', relief='flat')
            elif self.feedback_type == 'negative':
                self.thumbs_up_btn.config(bg='lightgray', state='disabled', relief='flat')
                self.thumbs_down_btn.config(bg='darkred', state='disabled', relief='sunken')

            # Enable undo button when feedback is given
            if self.undo_available:
                self.undo_btn.config(state='normal', bg='lightyellow')
        else:
            # Reset to normal state
            self.thumbs_up_btn.config(bg='lightgreen', state='normal', relief='raised')
            self.thumbs_down_btn.config(bg='lightcoral', state='normal', relief='raised')
            self.undo_btn.config(state='disabled', bg='lightgray')

    def show_feedback_buttons(self):
        """Show the feedback dropdown button and reset feedback state"""
        self.reset_feedback_state()
        # Reset dropdown state
        self.feedback_dropdown_open = False
        self.dropdown_btn.config(text="▼ Feedback Options")
        # Make sure the dropdown content is hidden initially
        self.feedback_frame.pack_forget()
        # Show the feedback container (which contains the dropdown button)
        self.feedback_container.pack(pady=10)

    def hide_feedback_buttons(self):
        """Hide the entire feedback section"""
        # Hide the entire feedback container
        self.feedback_container.pack_forget()
        # Also hide the dropdown content if it was open
        self.feedback_frame.pack_forget()
        # Reset dropdown state
        self.feedback_dropdown_open = False
        self.reset_feedback_state()

    def toggle_feedback_dropdown(self):
        """Toggle the feedback dropdown menu"""
        if self.feedback_dropdown_open:
            # Close the dropdown
            self.feedback_frame.pack_forget()
            self.feedback_dropdown_open = False
            self.dropdown_btn.config(text="▼ Feedback Options")
        else:
            # Open the dropdown
            self.feedback_frame.pack(pady=10)
            self.feedback_dropdown_open = True
            self.dropdown_btn.config(text="▲ Feedback Options")

    def undo_feedback(self):
        """Undo the last feedback action"""
        try:
            if not self.undo_available or not self.last_action_data:
                self.feedback_status_label.config(text="❌ No action to undo.", fg='red')
                return

            action_data = self.last_action_data
            action_type = action_data.get('action_type')
            digit = action_data.get('digit')

            if action_type == 'positive_feedback':
                # Remove the pattern that was added for positive feedback
                success = self.recognizer.remove_last_pattern(digit)
                if success:
                    self.feedback_status_label.config(
                        text=f"↶ Undone! Removed pattern for digit {digit}",
                        fg='blue'
                    )
                else:
                    self.feedback_status_label.config(
                        text="❌ Error: Pattern not found to undo",
                        fg='red'
                    )

            elif action_type == 'negative_feedback':
                # Remove the pattern that was added for the corrected digit
                success = self.recognizer.remove_last_pattern(digit)
                if success:
                    corrected_from = action_data.get('corrected_from', 'unknown')
                    self.feedback_status_label.config(
                        text=f"↶ Undone! Removed correction from {corrected_from} to {digit}",
                        fg='blue'
                    )
                else:
                    self.feedback_status_label.config(
                        text="❌ Error: Correction not found to undo",
                        fg='red'
                    )

            # Reset feedback state
            self.feedback_given = False
            self.feedback_type = None
            self.undo_available = False
            self.last_action_data = None

            # Update button appearances
            self.update_feedback_buttons()

            # Save the changes to file
            if not self.recognizer.save_learned_patterns():
                self.feedback_status_label.config(
                    text="❌ Error saving after undo",
                    fg='red'
                )

        except Exception as e:
            self.feedback_status_label.config(text=f"❌ Undo error: {str(e)}", fg='red')
            print(f"Undo error: {e}")


def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = DigitRecognitionGUI(root)

    # Handle window closing gracefully
    def on_closing():
        try:
            app.recognizer.save_learned_patterns()
        except Exception as e:
            print(f"Error saving on exit: {e}")
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
