import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from DigitRecognizer import DigitRecognizer


class DebugWindow:
    """Separate window for debug settings and controls"""

    def __init__(self, parent_gui):
        self.parent_gui = parent_gui
        self.window = None

    def apply_theme(self):
        """Apply the current theme to the debug window"""
        if not self.window:
            return

        theme = self.parent_gui.get_current_theme()
        self.window.configure(bg=theme['bg'])

        # Update all child widgets
        for widget in self.window.winfo_children():
            self._update_widget_theme(widget, theme)

    def _update_widget_theme(self, widget, theme):
        """Recursively update widget themes"""
        try:
            widget_class = widget.winfo_class()
            if widget_class in ['Label', 'Button', 'Frame', 'Checkbutton']:
                if widget_class == 'Button':
                    # Don't change button colors if they have specific colors
                    current_bg = widget.cget('bg')
                    if current_bg in ['lightblue', 'orange', 'lightcoral']:
                        pass  # Keep special button colors
                    else:
                        widget.configure(bg=theme['button_bg'], fg=theme['fg'])
                else:
                    widget.configure(bg=theme['bg'], fg=theme['fg'])
        except tk.TclError:
            pass  # Some widgets might not support these options

        # Recursively update children
        for child in widget.winfo_children():
            self._update_widget_theme(child, theme)

    def show_window(self):
        """Create and show the debug window"""
        if self.window is not None:
            # If window already exists, just bring it to front
            self.window.lift()
            self.window.focus_force()
            return

        # Create new debug window
        self.window = tk.Toplevel(self.parent_gui.root)
        self.window.title("Debug Settings")
        self.window.geometry("500x400")
        theme = self.parent_gui.get_current_theme()
        self.window.configure(bg=theme['bg'])
        self.window.transient(self.parent_gui.root)

        # Handle window closing
        self.window.protocol("WM_DELETE_WINDOW", self.close_window)

        # Center the window
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth() // 2) - (self.window.winfo_width() // 2)
        y = (self.window.winfo_screenheight() // 2) - (self.window.winfo_height() // 2)
        self.window.geometry(f"+{x}+{y}")

        self.setup_debug_gui()

    def close_window(self):
        """Close the debug window"""
        if self.window:
            self.window.destroy()
            self.window = None

    def setup_debug_gui(self):
        """Set up the debug window GUI elements"""
        theme = self.parent_gui.get_current_theme()

        # Title
        title_label = tk.Label(self.window, text="Debug Settings",
                              font=("Arial", 16, "bold"), bg=theme['bg'], fg=theme['fg'])
        title_label.pack(pady=10)

        # Array visualization settings
        vis_frame = tk.Frame(self.window, bg=theme['bg'], relief='solid', borderwidth=1)
        vis_frame.pack(pady=10, padx=10, fill=tk.X)

        tk.Label(vis_frame, text="Array Visualization",
                font=("Arial", 12, "bold"), bg=theme['bg'], fg=theme['fg']).pack(pady=5)

        # Array visualization checkbox
        array_vis_frame = tk.Frame(vis_frame, bg=theme['bg'])
        array_vis_frame.pack(anchor=tk.W, padx=10, pady=5)

        tk.Label(array_vis_frame, text="Show Array Visualization:",
                font=("Arial", 10), bg=theme['bg'], fg=theme['fg']).pack(side=tk.LEFT)

        self.array_visualization_checkbox = tk.Checkbutton(
            array_vis_frame,
            variable=self.parent_gui.array_visualization_enabled,
            onvalue=True,
            offvalue=False,
            font=("Arial", 10),
            bg=theme['bg'], fg=theme['fg']
        )
        self.array_visualization_checkbox.pack(side=tk.LEFT, padx=5)

        # Test visualization button
        test_vis_btn = tk.Button(vis_frame, text="Test Array Visualization",
                                command=self.test_array_visualization,
                                font=("Arial", 10), bg='lightblue', width=20)
        test_vis_btn.pack(pady=5)

        # System information
        info_frame = tk.Frame(self.window, bg=theme['bg'], relief='solid', borderwidth=1)
        info_frame.pack(pady=10, padx=10, fill=tk.X)

        tk.Label(info_frame, text="System Information",
                font=("Arial", 12, "bold"), bg=theme['bg'], fg=theme['fg']).pack(pady=5)

        # Pattern count
        total_patterns = self.parent_gui.recognizer.get_pattern_count()
        pattern_info = tk.Label(info_frame, text=f"Total learned patterns: {total_patterns}",
                               font=("Arial", 10), bg=theme['bg'], fg=theme['fg'])
        pattern_info.pack(pady=2)

        # Feedback dropdown state
        self.dropdown_state_label = tk.Label(info_frame, text="Feedback dropdown: Closed",
                                             font=("Arial", 10), bg=theme['bg'], fg='gray')
        self.dropdown_state_label.pack(pady=2)

        # Pattern management
        pattern_frame = tk.Frame(self.window, bg=theme['bg'], relief='solid', borderwidth=1)
        pattern_frame.pack(pady=10, padx=10, fill=tk.X)

        tk.Label(pattern_frame, text="Pattern Management",
                font=("Arial", 12, "bold"), bg=theme['bg'], fg=theme['fg']).pack(pady=5)

        # Remove duplicates button
        remove_dup_btn = tk.Button(pattern_frame, text="Remove Duplicate Patterns",
                                  command=self.remove_duplicates,
                                  font=("Arial", 10), bg='orange', width=25)
        remove_dup_btn.pack(pady=5)

        # Pattern details by digit
        details_frame = tk.Frame(pattern_frame, bg=theme['bg'])
        details_frame.pack(pady=5, fill=tk.X)

        tk.Label(details_frame, text="Patterns per digit:",
                font=("Arial", 10, "bold"), bg=theme['bg'], fg=theme['fg']).pack()

        # Show pattern count for each digit
        for digit in range(10):
            count = self.parent_gui.recognizer.get_pattern_count(digit)
            if count > 0:
                digit_label = tk.Label(details_frame, text=f"Digit {digit}: {count} patterns",
                                      font=("Arial", 9), bg=theme['bg'], fg=theme['fg'])
                digit_label.pack()

        # Close button
        close_btn = tk.Button(self.window, text="Close", command=self.close_window,
                             font=("Arial", 12), bg='lightcoral', width=12)
        close_btn.pack(pady=20)

    def test_array_visualization(self):
        """Test the array visualization with a sample pattern"""
        # Create a simple test pattern
        test_array = np.zeros((28, 28), dtype=int)
        # Draw a simple digit-like pattern
        test_array[10:18, 10:15] = 1  # Vertical line
        test_array[14:16, 10:18] = 1  # Horizontal line

        self.parent_gui.visualize_array(test_array, "Test Pattern (Sample)")

    def remove_duplicates(self):
        """Remove duplicate patterns and update display"""
        try:
            removed = self.parent_gui.recognizer.remove_duplicates_from_patterns()
            if removed:
                # Refresh the debug window to show updated counts
                if self.window:
                    self.window.destroy()
                    self.window = None
                    self.show_window()
            else:
                tk.messagebox.showinfo("No Duplicates", "No duplicate patterns found.")
        except Exception as e:
            tk.messagebox.showerror("Error", f"Error removing duplicates: {e}")

    def update_dropdown_state(self, state):
        """Update the dropdown state display"""
        if self.window and hasattr(self, 'dropdown_state_label'):
            self.dropdown_state_label.config(text=f"Feedback dropdown: {state}")


class DigitRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognition")
        self.root.geometry("500x750")  # Increased height from 650 to 750

        # Theme management
        self.is_dark_mode = False  # Start with light mode
        self.themes = {
            'light': {
                'bg': 'lightgray',
                'fg': 'black',
                'button_bg': 'lightgray',
                'canvas_bg': 'white'
            },
            'dark': {
                'bg': '#2b2b2b',
                'fg': 'white',
                'button_bg': '#404040',
                'canvas_bg': '#1e1e1e'
            }
        }

        # Canvas settings
        self.canvas_size = 280
        self.grid_size = 28  # 28x28 grid for pattern matching
        self.cell_size = self.canvas_size // self.grid_size

        # Drawing state
        self.drawing = False
        self.drawn_pixels = set()
        self.canvas_items = []  # Track all drawn items for theme switching
        self.last_prediction = None
        self.last_drawn_array = None
        self.feedback_given = False  # Track if feedback has been provided
        self.feedback_type = None    # Track type of feedback ('positive' or 'negative')

        # Undo functionality state
        self.undo_available = False
        self.last_action_data = None  # Store data needed to undo the last action

        # Dropdown state for feedback
        self.feedback_dropdown_open = False

        # Debug settings
        self.array_visualization_enabled = tk.BooleanVar(value=False)  # Default: disabled

        # Initialize debug window
        self.debug_window = DebugWindow(self)

        # Initialize the digit recognizer
        self.recognizer = DigitRecognizer()

        # Create GUI elements
        self.setup_gui()

        # Apply initial theme after all widgets are created
        self.apply_theme()

    def get_current_theme(self):
        """Get the current theme dictionary"""
        return self.themes['dark'] if self.is_dark_mode else self.themes['light']

    def toggle_theme(self):
        """Toggle between light and dark themes"""
        self.is_dark_mode = not self.is_dark_mode
        self.apply_theme()

        # Update debug window theme if it's open
        if self.debug_window.window:
            self.debug_window.apply_theme()

    def apply_theme(self):
        """Apply the current theme to all widgets"""
        theme = self.get_current_theme()

        # Apply to root window
        self.root.configure(bg=theme['bg'])

        # Update all existing widgets if they exist
        if hasattr(self, 'title_label'):
            self._update_all_widgets()

    def _update_all_widgets(self):
        """Update all widgets with current theme"""
        theme = self.get_current_theme()

        # Update main widgets
        widgets_to_update = [
            self.title_label, self.instructions, self.prediction_label,
            self.confidence_label, self.feedback_status_label
        ]

        for widget in widgets_to_update:
            if widget.winfo_exists():
                widget.configure(bg=theme['bg'], fg=theme['fg'])

        # Update frames
        frames_to_update = [
            self.button_frame, self.feedback_container, self.feedback_frame,
            self.feedback_btn_frame, self.pattern_count_frame
        ]

        for frame in frames_to_update:
            if frame.winfo_exists():
                frame.configure(bg=theme['bg'])

        # Update pattern info label specifically
        if hasattr(self, 'pattern_info') and self.pattern_info.winfo_exists():
            if self.is_dark_mode:
                # In dark mode: make text white and remove any background/border
                self.pattern_info.configure(
                    bg=theme['bg'], 
                    fg='white',
                    relief='flat',
                    highlightthickness=0,
                    bd=0
                )
            else:
                # In light mode: keep as is
                self.pattern_info.configure(
                    bg=theme['bg'], 
                    fg='gray',
                    relief='flat',
                    highlightthickness=0,
                    bd=0
                )

        # Update canvas background
        if self.canvas.winfo_exists():
            self.canvas.configure(bg=theme['canvas_bg'])

        # Update buttons (keeping special colors for certain buttons)
        special_color_buttons = {
            self.predict_btn: 'lightblue',
            self.clear_btn: 'lightyellow',
            self.debug_btn: 'lightsteelblue',
            self.dropdown_btn: 'lightsteelblue',
            self.thumbs_up_btn: 'lightgreen',
            self.thumbs_down_btn: 'lightcoral'
        }

        for button, special_color in special_color_buttons.items():
            if button.winfo_exists():
                if self.is_dark_mode:
                    # Darken the special colors for dark mode
                    dark_colors = {
                        'lightblue': '#4A90E2',
                        'lightyellow': '#F5A623',
                        'lightsteelblue': '#5A9FD4',
                        'lightgreen': '#7ED321',
                        'lightcoral': '#D0021B'
                    }
                    # Remove white boxes by setting all border/highlight properties
                    button.configure(
                        bg=dark_colors.get(special_color, special_color),
                        fg='black',
                        relief='flat',
                        highlightbackground=theme['bg'],
                        highlightcolor=theme['bg'],
                        highlightthickness=0,
                        bd=0
                    )
                else:
                    # Light mode - remove borders completely
                    button.configure(
                        bg=special_color,
                        fg='black',
                        relief='flat',
                        highlightbackground=theme['bg'],
                        highlightcolor=theme['bg'],
                        highlightthickness=0,
                        bd=0
                    )

        # Update the top frame background
        if hasattr(self, 'top_frame') and self.top_frame.winfo_exists():
            self.top_frame.configure(bg=theme['bg'])

        # Update the undo button separately as it changes dynamically
        if self.undo_btn.winfo_exists():
            if self.undo_btn['state'] == 'disabled':
                button_bg = '#505050' if self.is_dark_mode else 'lightgray'
            else:
                button_bg = '#F5A623' if self.is_dark_mode else 'lightyellow'

            # Remove white boxes for undo button too
            self.undo_btn.configure(
                bg=button_bg,
                fg='black',
                relief='flat',
                highlightbackground=theme['bg'],
                highlightcolor=theme['bg'],
                highlightthickness=0,
                bd=0
            )

        # Update canvas border color based on theme
        if self.canvas.winfo_exists():
            if self.is_dark_mode:
                self.canvas.configure(bg=theme['canvas_bg'], highlightbackground='black',
                                    highlightcolor='black', highlightthickness=2)
            else:
                self.canvas.configure(bg=theme['canvas_bg'], highlightbackground='black',
                                    highlightcolor='black', highlightthickness=0)

        # Update theme toggle button
        if hasattr(self, 'theme_toggle_btn') and self.theme_toggle_btn.winfo_exists():
            if self.is_dark_mode:
                self.theme_toggle_btn.configure(
                    text="☀️",
                    bg='#404040',
                    fg='white',
                    relief='flat',
                    highlightbackground=theme['bg'],
                    highlightcolor=theme['bg'],
                    highlightthickness=0,
                    bd=0
                )
            else:
                self.theme_toggle_btn.configure(
                    text="🌙",
                    bg='lightgray',
                    fg='black',
                    relief='flat',
                    highlightbackground=theme['bg'],
                    highlightcolor=theme['bg'],
                    highlightthickness=0,
                    bd=0
                )

        # Update all canvas items to match the new theme
        self._update_canvas_item_colors()

    def _update_canvas_item_colors(self):
        """Update the colors of all canvas drawing items to match the current theme"""
        # Get the appropriate pen color for the current theme
        pen_color = 'white' if self.is_dark_mode else 'black'

        # Update all canvas items (drawn circles)
        canvas_items = self.canvas.find_all()
        for item in canvas_items:
            # Update the fill and outline colors of each canvas item
            self.canvas.itemconfig(item, fill=pen_color, outline=pen_color)

    def visualize_array(self, array, title="Binary Array Visualization"):
        """Visualize a 28x28 binary array using matplotlib"""
        try:
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

        except Exception as e:
            print(f"Error visualizing array: {e}")

    def print_array_info(self, array, title="Array Information"):
        """Print detailed information about the array"""
        print(f"\n{title}")
        print("=" * len(title))
        print(f"Array shape: {array.shape}")
        print(f"Non-zero pixels: {np.sum(array)}")
        print(f"Zero pixels: {np.sum(array == 0)}")
        print(f"Total pixels: {array.size}")
        print(f"Fill percentage: {(np.sum(array) / array.size) * 100:.1f}%")

        # Find bounding box
        rows, cols = np.where(array > 0)
        if len(rows) > 0:
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            print(f"Bounding box: rows {min_row}-{max_row}, cols {min_col}-{max_col}")
            print(f"Drawing size: {max_row - min_row + 1} x {max_col - min_col + 1}")
        else:
            print("No drawn pixels found")

        print("\nBinary Array:")
        print(array)
        print()

    def start_drawing(self, event):
        self.drawing = True
        self.draw(event)

    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            # Draw a small circle with theme-appropriate color
            radius = 8
            theme = self.get_current_theme()
            pen_color = 'white' if self.is_dark_mode else 'black'

            # Create an oval (circle) on the canvas
            oval = self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius,
                                          fill=pen_color, outline=pen_color)

            # Store drawn pixel and associated canvas item for tracking
            grid_x = min(x // self.cell_size, self.grid_size - 1)
            grid_y = min(y // self.cell_size, self.grid_size - 1)
            self.drawn_pixels.add((grid_x, grid_y))

            # Track the canvas item with its color for theme updates
            self.canvas_items.append({
                'type': 'oval',
                'widget': oval,
                'color': pen_color
            })

    def stop_drawing(self, event):
        self.drawing = False

    def clear_canvas(self):
        self.canvas.delete("all")
        self.drawn_pixels.clear()
        self.canvas_items.clear()  # Clear the canvas items tracking list
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
        """Convert drawing to 28x28 binary array with improved coverage detection"""
        array = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # Create a higher resolution coverage map first
        coverage_map = np.zeros((self.grid_size, self.grid_size), dtype=float)

        # Get all canvas items (the drawn circles)
        canvas_items = self.canvas.find_all()

        for item in canvas_items:
            # Get the bounding box of each drawn circle
            coords = self.canvas.coords(item)
            if len(coords) == 4:  # oval coordinates: x1, y1, x2, y2
                x1, y1, x2, y2 = coords
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                radius = (x2 - x1) / 2

                # Check which grid cells this circle affects
                for grid_y in range(self.grid_size):
                    for grid_x in range(self.grid_size):
                        # Calculate the center of this grid cell in canvas coordinates
                        cell_center_x = (grid_x + 0.5) * self.cell_size
                        cell_center_y = (grid_y + 0.5) * self.cell_size

                        # Calculate distance from circle center to cell center
                        distance = np.sqrt((center_x - cell_center_x)**2 + (center_y - cell_center_y)**2)

                        # If the circle overlaps with this cell, add coverage
                        if distance <= radius + (self.cell_size * 0.7):  # Allow some overlap
                            # Calculate coverage strength based on distance
                            coverage_strength = max(0, 1 - (distance / (radius + self.cell_size * 0.5)))
                            coverage_map[grid_y, grid_x] += coverage_strength

        # Convert coverage map to binary array with threshold
        # Lower threshold makes it more likely to register as 1
        threshold = 0.3  # Adjust this value: lower = more sensitive
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if coverage_map[y, x] >= threshold:
                    array[y, x] = 1

        return array

    def predict_digit(self):
        """Predict the digit based on the drawing with comprehensive error handling"""
        try:
            # Check if canvas is empty
            if not self.drawn_pixels:
                self.prediction_label.config(text="❌ Canvas is empty!", fg='red')
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

            # Only print and visualize arrays if debug mode is enabled
            if self.array_visualization_enabled.get():
                self.print_array_info(drawn_array, "Raw Drawing Array")

                # Get preprocessed array from recognizer
                processed_array = self.recognizer.preprocess_drawing(drawn_array)
                self.print_array_info(processed_array, "Preprocessed Array")

                # Visualize both arrays
                self.visualize_array(drawn_array, "Raw Drawing (28x28 Grid)")
                self.visualize_array(processed_array, "Preprocessed Drawing (Centered/Scaled)")

            # Use the recognizer to predict
            best_digit, best_similarity, similarities = self.recognizer.predict_digit(drawn_array)

            # Store the prediction for feedback
            self.last_prediction = best_digit

            # Print prediction details
            print(f"\nPrediction Results:")
            print(f"Best digit: {best_digit}")
            print(f"Best similarity: {best_similarity:.4f}")
            print(f"All similarities: {similarities}")

            # Display result with better error handling
            if best_digit is not None and best_similarity > 0.05: # Lowered threshold for better detection
                # Use standard theme color for prediction text instead of blue
                theme = self.get_current_theme()
                self.prediction_label.config(text=f"Predicted Digit: {best_digit}", fg=theme['fg'])
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
                self.feedback_status_label.config(text="�� Error adding correction", fg='red')

            dialog.destroy()

        except Exception as e:
            self.feedback_status_label.config(text=f"❌ Error: {str(e)}", fg='red')
            dialog.destroy()

    def update_feedback_buttons(self):
        """Update the appearance of feedback buttons based on feedback state"""
        theme = self.get_current_theme()

        if self.feedback_given:
            if self.feedback_type == 'positive':
                # Use theme-appropriate disabled colors
                disabled_bg = '#404040' if self.is_dark_mode else 'lightgray'
                self.thumbs_up_btn.config(
                    bg='darkgreen', fg='black', state='disabled', relief='sunken',
                    highlightbackground=theme['bg'], highlightcolor=theme['bg'],
                    highlightthickness=0, bd=0
                )
                self.thumbs_down_btn.config(
                    bg=disabled_bg, fg='black', state='disabled', relief='flat',
                    highlightbackground=theme['bg'], highlightcolor=theme['bg'],
                    highlightthickness=0, bd=0
                )
            elif self.feedback_type == 'negative':
                disabled_bg = '#404040' if self.is_dark_mode else 'lightgray'
                self.thumbs_up_btn.config(
                    bg=disabled_bg, fg='black', state='disabled', relief='flat',
                    highlightbackground=theme['bg'], highlightcolor=theme['bg'],
                    highlightthickness=0, bd=0
                )
                self.thumbs_down_btn.config(
                    bg='darkred', fg='black', state='disabled', relief='sunken',
                    highlightbackground=theme['bg'], highlightcolor=theme['bg'],
                    highlightthickness=0, bd=0
                )

            # Enable undo button when feedback is given
            if self.undo_available:
                undo_bg = '#F5A623' if self.is_dark_mode else 'lightyellow'
                self.undo_btn.config(
                    state='normal', bg=undo_bg, fg='black',
                    highlightbackground=theme['bg'], highlightcolor=theme['bg'],
                    highlightthickness=0, bd=0
                )
        else:
            # Reset to normal state with theme-appropriate colors
            up_bg = '#7ED321' if self.is_dark_mode else 'lightgreen'
            down_bg = '#D0021B' if self.is_dark_mode else 'lightcoral'
            disabled_bg = '#505050' if self.is_dark_mode else 'lightgray'

            self.thumbs_up_btn.config(
                bg=up_bg, fg='black', state='normal', relief='raised',
                highlightbackground=theme['bg'], highlightcolor=theme['bg'],
                highlightthickness=0, bd=0
            )
            self.thumbs_down_btn.config(
                bg=down_bg, fg='black', state='normal', relief='raised',
                highlightbackground=theme['bg'], highlightcolor=theme['bg'],
                highlightthickness=0, bd=0
            )
            self.undo_btn.config(
                state='disabled', bg=disabled_bg, fg='black',
                highlightbackground=theme['bg'], highlightcolor=theme['bg'],
                highlightthickness=0, bd=0
            )

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
            state = "Closed"
        else:
            # Open the dropdown
            self.feedback_frame.pack(pady=10)
            self.feedback_dropdown_open = True
            self.dropdown_btn.config(text="▲ Feedback Options")
            state = "Open"

        # Update debug window if it's open
        self.debug_window.update_dropdown_state(state)

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

    def setup_gui(self):
        """Set up all GUI elements"""
        # Create a top frame for title and theme toggle
        self.top_frame = tk.Frame(self.root, bg='lightgray')
        self.top_frame.pack(fill=tk.X, pady=10)

        # Title (centered in top frame)
        self.title_label = tk.Label(self.top_frame, text="Handwritten Digit Recognition",
                              font=("Arial", 16, "bold"), bg='lightgray')
        self.title_label.pack(expand=True)  # Center the title properly

        # Theme toggle button (positioned absolutely in top right)
        self.theme_toggle_btn = tk.Button(self.top_frame, text="🌙", command=self.toggle_theme,
                                         font=("Arial", 12, "bold"), bg='lightgray', width=3, height=1)
        self.theme_toggle_btn.place(relx=1.0, rely=0.5, anchor='e', x=-10)  # Position in top right

        # Instructions
        self.instructions = tk.Label(self.root, text="Draw a digit in the box below and click Predict",
                               font=("Arial", 12), bg='lightgray')
        self.instructions.pack(pady=5)

        # Drawing canvas
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size,
                               bg='white', relief='solid', borderwidth=2)
        self.canvas.pack(pady=20)

        # Bind mouse events for drawing
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        # Button frame
        self.button_frame = tk.Frame(self.root, bg='lightgray')
        self.button_frame.pack(pady=10)

        # Predict button
        self.predict_btn = tk.Button(self.button_frame, text="Predict", command=self.predict_digit,
                               font=("Arial", 12, "bold"), bg='lightblue', width=10)
        self.predict_btn.pack(side=tk.LEFT, padx=5)

        # Clear button
        self.clear_btn = tk.Button(self.button_frame, text="Clear", command=self.clear_canvas,
                             font=("Arial", 12), bg='lightyellow', width=10)
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        # Debug button
        self.debug_btn = tk.Button(self.button_frame, text="Debug", command=self.debug_window.show_window,
                             font=("Arial", 12), bg='lightsteelblue', width=10)
        self.debug_btn.pack(side=tk.LEFT, padx=5)

        # Prediction result
        self.prediction_label = tk.Label(self.root, text="Draw a digit and click Predict",
                                        font=("Arial", 14, "bold"), bg='lightgray', fg='black')
        self.prediction_label.pack(pady=10)

        # Confidence label
        self.confidence_label = tk.Label(self.root, text="", font=("Arial", 10), bg='lightgray')
        self.confidence_label.pack()

        # Feedback container (initially hidden)
        self.feedback_container = tk.Frame(self.root, bg='lightgray')
        # Don't pack initially - will be shown when prediction is made

        # Dropdown button for feedback options
        self.dropdown_btn = tk.Button(self.feedback_container, text="▼ Feedback Options",
                                     command=self.toggle_feedback_dropdown,
                                     font=("Arial", 11, "bold"), bg='lightsteelblue', width=20)
        self.dropdown_btn.pack(pady=5)

        # Feedback options frame (dropdown content)
        self.feedback_frame = tk.Frame(self.feedback_container, bg='lightgray', relief='solid', borderwidth=1)
        # Don't pack initially - will be shown when dropdown is opened

        # Feedback buttons
        self.feedback_btn_frame = tk.Frame(self.feedback_frame, bg='lightgray')
        self.feedback_btn_frame.pack(pady=5)

        self.thumbs_up_btn = tk.Button(self.feedback_btn_frame, text="👍 Correct",
                                      command=self.positive_feedback,
                                      font=("Arial", 10, "bold"), bg='lightgreen', width=12)
        self.thumbs_up_btn.pack(side=tk.LEFT, padx=5)

        self.thumbs_down_btn = tk.Button(self.feedback_btn_frame, text="👎 Wrong",
                                        command=self.negative_feedback,
                                        font=("Arial", 10, "bold"), bg='lightcoral', width=12)
        self.thumbs_down_btn.pack(side=tk.LEFT, padx=5)

        # Undo button
        self.undo_btn = tk.Button(self.feedback_frame, text="↶ Undo Last Feedback",
                                 command=self.undo_feedback,
                                 font=("Arial", 10), bg='lightgray', width=20, state='disabled')
        self.undo_btn.pack(pady=5)

        # Feedback status label
        self.feedback_status_label = tk.Label(self.root, text="", font=("Arial", 10),
                                             bg='lightgray', wraplength=400)
        self.feedback_status_label.pack(pady=5)

        # Pattern count display (moved to bottom)
        self.pattern_count_frame = tk.Frame(self.root, bg='lightgray')
        self.pattern_count_frame.pack(side=tk.BOTTOM, pady=10)

        total_patterns = self.recognizer.get_pattern_count()
        self.pattern_info = tk.Label(self.pattern_count_frame, text=f"Total learned patterns: {total_patterns}",
                               font=("Arial", 9), bg='lightgray', fg='gray')
        self.pattern_info.pack()



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
