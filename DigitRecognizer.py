import numpy as np
import json
import os
import hashlib


class DigitRecognizer:
    """
    A class that handles digit recognition, pattern matching, and learning functionality.
    This class is separate from the GUI and focuses purely on the recognition logic.
    """

    def __init__(self):
        # Grid settings for pattern matching
        self.grid_size = 28  # 28x28 grid for pattern matching

        # File for storing learned patterns
        self.learned_patterns_file = "learned_patterns.json"

        # Pattern caching for performance optimization
        self.cached_patterns = {}  # {digit: [{'array': np_array, 'confidence': float, 'feedback_count': int}]}
        self.cache_valid = False   # Track if cache is valid

        # Initialize digit patterns
        self.digit_patterns = self.create_digit_patterns()

        # Load learned patterns and automatically remove duplicates
        self.learned_patterns = self.load_learned_patterns()
        self.remove_duplicates_from_patterns()

        # Build initial cache
        self._rebuild_pattern_cache()

    def preprocess_drawing(self, array):
        """Preprocess the drawing array with smart centering (no scaling)"""
        # Find the bounding box of the drawn pixels
        rows, cols = np.where(array > 0)

        if len(rows) == 0:
            return array  # Empty array

        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()

        # Extract the drawing within its bounding box
        drawing_height = max_row - min_row + 1
        drawing_width = max_col - min_col + 1

        # Calculate canvas utilization and center offset
        canvas_utilization = (drawing_height * drawing_width) / (self.grid_size * self.grid_size)

        center_row = (min_row + max_row) / 2
        center_col = (min_col + max_col) / 2
        canvas_center = self.grid_size / 2

        center_offset = max(abs(center_row - canvas_center), abs(center_col - canvas_center))

        # Apply centering for small drawings that are very off-center
        # More generous thresholds since we removed scaling
        if canvas_utilization < 0.15 and center_offset > 6:
            return self.gentle_center_drawing(array, min_row, max_row, min_col, max_col)

        # For all other cases, return the original drawing unchanged
        return array

    def smart_crop_and_scale(self, array, min_row, max_row, min_col, max_col):
        """Crop around the digit with padding, then scale up intelligently"""
        drawing_height = max_row - min_row + 1
        drawing_width = max_col - min_col + 1

        # Add padding around the digit (2-3 pixels on each side)
        padding = 3
        crop_min_row = max(0, min_row - padding)
        crop_max_row = min(self.grid_size - 1, max_row + padding)
        crop_min_col = max(0, min_col - padding)
        crop_max_col = min(self.grid_size - 1, max_col + padding)

        # Extract the cropped region
        cropped_height = crop_max_row - crop_min_row + 1
        cropped_width = crop_max_col - crop_min_col + 1

        # Create cropped array
        cropped_array = array[crop_min_row:crop_max_row + 1, crop_min_col:crop_max_col + 1]

        # Calculate scale factor to make the cropped region fill a good portion of the canvas
        # Target: make the scaled digit use about 60-70% of the canvas
        target_size = int(self.grid_size * 0.65)
        scale_factor = target_size / max(cropped_height, cropped_width)

        # Cap the scaling to prevent over-enlargement
        scale_factor = min(scale_factor, 6.0)  # Allow more aggressive scaling for tiny digits

        # Calculate new dimensions after scaling
        new_height = int(cropped_height * scale_factor)
        new_width = int(cropped_width * scale_factor)

        # Center the scaled cropped region in the full canvas
        start_row = max(0, (self.grid_size - new_height) // 2)
        start_col = max(0, (self.grid_size - new_width) // 2)

        # Create new array
        new_array = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # Apply scaling using nearest neighbor interpolation
        for i in range(new_height):
            for j in range(new_width):
                # Map back to cropped coordinates
                orig_row = int(i / scale_factor)
                orig_col = int(j / scale_factor)

                # Ensure we stay within bounds
                if (start_row + i < self.grid_size and start_col + j < self.grid_size and
                    orig_row < cropped_height and orig_col < cropped_width):
                    new_array[start_row + i, start_col + j] = cropped_array[orig_row, orig_col]

        return new_array

    def gentle_center_drawing(self, array, min_row, max_row, min_col, max_col):
        """Apply proper centering with full centering when triggered"""
        drawing_height = max_row - min_row + 1
        drawing_width = max_col - min_col + 1

        # Calculate where the drawing should be positioned to be centered
        canvas_center = self.grid_size / 2

        # Calculate the top-left position to center the drawing
        target_start_row = int(canvas_center - drawing_height / 2)
        target_start_col = int(canvas_center - drawing_width / 2)

        # Ensure the target position keeps the drawing within bounds
        target_start_row = max(0, min(target_start_row, self.grid_size - drawing_height))
        target_start_col = max(0, min(target_start_col, self.grid_size - drawing_width))

        # Create new array
        new_array = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # Copy the drawing to the centered position
        for i in range(drawing_height):
            for j in range(drawing_width):
                old_row = min_row + i
                old_col = min_col + j
                new_row = target_start_row + i
                new_col = target_start_col + j

                # Copy the pixel if both source and destination are valid
                if (0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size and
                    0 <= old_row < self.grid_size and 0 <= old_col < self.grid_size):
                    new_array[new_row, new_col] = array[old_row, old_col]

        return new_array

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
        """Create empty patterns dictionary - using only learned patterns"""
        # Return empty dictionary since we're only using learned patterns
        return {}

    def calculate_similarity(self, drawn_array, pattern):
        """Calculate similarity between drawn array and pattern using vectorized NumPy operations for optimal performance"""
        # Normalize both arrays to ensure they're binary using vectorized operations
        drawn_binary = (drawn_array > 0).astype(np.int8)  # Use int8 for memory efficiency
        pattern_binary = (pattern > 0).astype(np.int8)

        # Vectorized intersection and union calculations
        intersection_array = drawn_binary & pattern_binary
        union_array = drawn_binary | pattern_binary

        intersection = np.sum(intersection_array)
        union = np.sum(union_array)

        if union == 0:
            return 0.0

        # Vectorized pixel counts
        pattern_pixels = np.sum(pattern_binary)
        drawn_pixels = np.sum(drawn_binary)

        if pattern_pixels == 0 or drawn_pixels == 0:
            return 0.0

        # Calculate all metrics using vectorized operations
        iou = intersection / union
        coverage = intersection / pattern_pixels
        precision = intersection / drawn_pixels

        # Calculate structural similarity using vectorized center of mass
        structural_similarity = self.calculate_structural_similarity_vectorized(drawn_binary, pattern_binary)

        # Enhanced weighted combination for 28x28 resolution
        similarity = 0.35 * iou + 0.25 * coverage + 0.25 * precision + 0.15 * structural_similarity

        return similarity

    def calculate_structural_similarity_vectorized(self, drawn_binary, pattern_binary):
        """Calculate structural similarity using fully vectorized NumPy operations"""
        try:
            # Use vectorized operations to find coordinates
            drawn_coords = np.nonzero(drawn_binary)
            pattern_coords = np.nonzero(pattern_binary)

            if len(drawn_coords[0]) == 0 or len(pattern_coords[0]) == 0:
                return 0.0

            # Vectorized center of mass calculation
            drawn_com = np.array([np.mean(drawn_coords[0]), np.mean(drawn_coords[1])])
            pattern_com = np.array([np.mean(pattern_coords[0]), np.mean(pattern_coords[1])])

            # Vectorized distance calculation
            com_diff = drawn_com - pattern_com
            com_distance = np.sqrt(np.sum(com_diff ** 2))
            com_similarity = max(0, 1 - (com_distance / (self.grid_size * 0.5)))

            # Vectorized spread calculation
            drawn_spread = float(np.std(drawn_coords[0]) + np.std(drawn_coords[1]))
            pattern_spread = float(np.std(pattern_coords[0]) + np.std(pattern_coords[1]))

            if pattern_spread > 0:
                spread_ratio = min(drawn_spread, pattern_spread) / max(drawn_spread, pattern_spread)
            else:
                spread_ratio = 1.0 if drawn_spread == 0 else 0.0

            return 0.7 * com_similarity + 0.3 * spread_ratio

        except Exception as e:
            print(f"Structural similarity error: {e}")
            return 0.0

    def predict_digit(self, drawn_array):
        """
        Predict the digit using cached patterns for optimal performance
        """
        try:
            # Preprocess the drawing once
            processed_array = self.preprocess_drawing(drawn_array)

            # Early exit if no patterns available
            if not self.cached_patterns:
                print("No cached patterns available for comparison")
                return None, 0.0, {}

            # Ensure cache is valid before prediction
            self._ensure_cache_valid()

            # Use cached patterns for much faster processing
            pattern_results = []
            total_patterns = 0

            # Process all cached patterns directly
            for digit, cached_patterns_list in self.cached_patterns.items():
                for pattern_info in cached_patterns_list:
                    total_patterns += 1

                    # Calculate both similarities using pre-converted arrays
                    preprocessed_sim = self.calculate_similarity(processed_array, pattern_info['array'])
                    raw_sim = self.calculate_similarity(drawn_array, pattern_info['array'])

                    # Apply boosts once
                    confidence_boost = 1.0 + (pattern_info['confidence'] * 0.2)
                    feedback_boost = 1.0 + (pattern_info['feedback_count'] * 0.1)
                    boost_factor = confidence_boost * feedback_boost

                    # Store results if above threshold
                    prep_boosted = preprocessed_sim * boost_factor
                    raw_boosted = raw_sim * boost_factor

                    if prep_boosted >= 0.05 or raw_boosted >= 0.05:  # Early filtering
                        pattern_results.append({
                            'digit': digit,
                            'prep_similarity': prep_boosted,
                            'raw_similarity': raw_boosted,
                            'prep_raw': preprocessed_sim,
                            'raw_raw': raw_sim
                        })

            if not pattern_results:
                print("No patterns meet minimum similarity threshold")
                return None, 0.0, {}

            # Sort once by best overall similarity
            pattern_results.sort(key=lambda x: max(x['prep_similarity'], x['raw_similarity']), reverse=True)

            # Limit processing to top 30 patterns total (instead of 15 each)
            top_patterns = pattern_results[:30]

            # Build hashmaps efficiently
            prep_stats = {}  # {digit: {'votes': count, 'sim_sum': total, 'max_sim': max}}
            raw_stats = {}
            combined_stats = {}

            for pattern in top_patterns:
                digit = pattern['digit']

                # Preprocessed stats
                if pattern['prep_similarity'] >= 0.05:
                    if digit not in prep_stats:
                        prep_stats[digit] = {'votes': 0, 'sim_sum': 0, 'max_sim': 0}
                    prep_stats[digit]['votes'] += 1
                    prep_stats[digit]['sim_sum'] += pattern['prep_similarity']
                    prep_stats[digit]['max_sim'] = max(prep_stats[digit]['max_sim'], pattern['prep_similarity'])

                # Raw stats
                if pattern['raw_similarity'] >= 0.05:
                    if digit not in raw_stats:
                        raw_stats[digit] = {'votes': 0, 'sim_sum': 0, 'max_sim': 0}
                    raw_stats[digit]['votes'] += 1
                    raw_stats[digit]['sim_sum'] += pattern['raw_similarity']
                    raw_stats[digit]['max_sim'] = max(raw_stats[digit]['max_sim'], pattern['raw_similarity'])

                # Combined stats
                if digit not in combined_stats:
                    combined_stats[digit] = {'votes': 0, 'sim_sum': 0, 'max_sim': 0}
                combined_stats[digit]['votes'] += 1
                combined_stats[digit]['sim_sum'] += max(pattern['prep_similarity'], pattern['raw_similarity'])
                combined_stats[digit]['max_sim'] = max(combined_stats[digit]['max_sim'],
                                                      max(pattern['prep_similarity'], pattern['raw_similarity']))

            # Calculate final scores efficiently
            all_digits = set(prep_stats.keys()) | set(raw_stats.keys())
            digit_scores = {}

            for digit in all_digits:
                # Get stats with defaults
                prep = prep_stats.get(digit, {'votes': 0, 'sim_sum': 0, 'max_sim': 0})
                raw = raw_stats.get(digit, {'votes': 0, 'sim_sum': 0, 'max_sim': 0})
                combined = combined_stats.get(digit, {'votes': 0, 'sim_sum': 0, 'max_sim': 0})

                # Calculate averages and weights efficiently
                prep_avg = prep['sim_sum'] / prep['votes'] if prep['votes'] > 0 else 0
                raw_avg = raw['sim_sum'] / raw['votes'] if raw['votes'] > 0 else 0
                combined_avg = combined['sim_sum'] / combined['votes'] if combined['votes'] > 0 else 0

                prep_vote_weight = min(prep['votes'] / 5.0, 1.0)
                raw_vote_weight = min(raw['votes'] / 5.0, 1.0)
                combined_vote_weight = min(combined['votes'] / 10.0, 1.0)

                # Vectorized score calculation
                prep_score = 0.7 * prep['max_sim'] + 0.2 * prep_avg + 0.1 * prep_vote_weight
                raw_score = 0.7 * raw['max_sim'] + 0.2 * raw_avg + 0.1 * raw_vote_weight
                combined_score = 0.7 * combined['max_sim'] + 0.2 * combined_avg + 0.1 * combined_vote_weight

                digit_scores[digit] = 0.5 * prep_score + 0.2 * raw_score + 0.3 * combined_score

            # Find best prediction
            best_digit = max(digit_scores, key=digit_scores.get)
            best_score = digit_scores[best_digit]

            # Create final similarities dict
            similarities = {digit: digit_scores.get(digit, 0.0) for digit in range(10)}

            # Simplified logging (only when needed)
            print(f"\nCached Analysis: {total_patterns} patterns, {len(top_patterns)} top matches")
            print(f"Final scores: {dict(sorted(digit_scores.items()))}")

            return best_digit, best_score, similarities

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

            # IMPORTANT: Preprocess the drawn array before saving
            preprocessed_array = self.preprocess_drawing(drawn_array)

            # Add the preprocessed drawing as a learned pattern
            pattern_data = {
                'pattern': preprocessed_array.tolist(),
                'confidence': 1.0,
                'feedback_count': 1
            }

            self.learned_patterns[digit].append(pattern_data)
            self._invalidate_cache()  # Invalidate cache on new feedback
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

            # IMPORTANT: Preprocess the drawn array before saving
            preprocessed_array = self.preprocess_drawing(drawn_array)

            # Add the preprocessed drawing as a learned pattern with the correct label
            pattern_data = {
                'pattern': preprocessed_array.tolist(),
                'confidence': 1.0,
                'feedback_count': 1,
                'corrected_from': predicted_digit
            }

            self.learned_patterns[digit_str].append(pattern_data)
            self._invalidate_cache()  # Invalidate cache on new feedback
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

                self._invalidate_cache()  # Invalidate cache on pattern removal
                return True
            return False
        except Exception as e:
            print(f"Remove pattern error: {e}")
            return False

    def pattern_to_hash(self, pattern):
        """Convert a pattern array to a hash for duplicate detection"""
        # Convert pattern to string and hash it
        pattern_str = str(pattern)
        return hashlib.md5(pattern_str.encode()).hexdigest()

    def remove_duplicates_from_patterns(self):
        """Remove duplicate patterns from learned patterns automatically"""
        try:
            if not self.learned_patterns:
                return False

            duplicates_removed = 0
            patterns_modified = False

            # Process each digit
            for digit, patterns in self.learned_patterns.items():
                if not patterns:
                    continue

                original_count = len(patterns)

                # Track unique patterns using hashes
                seen_hashes = set()
                unique_patterns = []

                for pattern_data in patterns:
                    if 'pattern' not in pattern_data:
                        continue

                    pattern = pattern_data['pattern']
                    pattern_hash = self.pattern_to_hash(pattern)

                    if pattern_hash not in seen_hashes:
                        seen_hashes.add(pattern_hash)
                        unique_patterns.append(pattern_data)
                    else:
                        duplicates_removed += 1

                # Update if duplicates were found
                if len(unique_patterns) != original_count:
                    self.learned_patterns[digit] = unique_patterns
                    patterns_modified = True
                    print(f"Removed {original_count - len(unique_patterns)} duplicate patterns for digit {digit}")

            # Save if any duplicates were removed
            if patterns_modified:
                print(f"Total duplicates removed: {duplicates_removed}")
                self.save_learned_patterns()
                return True

            return False

        except Exception as e:
            print(f"Error removing duplicates: {e}")
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

    def _invalidate_cache(self):
        """Invalidate the pattern cache when patterns are modified"""
        self.cache_valid = False

    def _ensure_cache_valid(self):
        """Ensure the cache is valid, rebuild if necessary"""
        if not self.cache_valid:
            self._rebuild_pattern_cache()

    def _rebuild_pattern_cache(self):
        """Rebuild the pattern cache from learned patterns"""
        try:
            self.cached_patterns = {}
            self.cache_valid = False

            # Convert all patterns to numpy arrays and cache them
            for digit, patterns_list in self.learned_patterns.items():
                digit_int = int(digit)
                cached_list = []

                for pattern_data in patterns_list:
                    try:
                        # Convert pattern to numpy array
                        pattern_array = np.array(pattern_data['pattern'])

                        # Cache the pattern with its metadata
                        cached_list.append({
                            'array': pattern_array,
                            'confidence': pattern_data.get('confidence', 1.0),
                            'feedback_count': pattern_data.get('feedback_count', 1)
                        })
                    except (ValueError, KeyError):
                        continue

                if cached_list:
                    self.cached_patterns[digit_int] = cached_list

            self.cache_valid = True
            print(f"Pattern cache rebuilt: {sum(len(patterns) for patterns in self.cached_patterns.values())} patterns cached")

        except Exception as e:
            print(f"Error rebuilding pattern cache: {e}")
            self.cache_valid = False
