#!/usr/bin/env python3
"""
Script to remove duplicate patterns from learned_patterns.json
"""

import json
import hashlib
import shutil
from datetime import datetime

def pattern_to_hash(pattern):
    """Convert a pattern array to a hash for duplicate detection"""
    # Convert pattern to string and hash it
    pattern_str = str(pattern)
    return hashlib.md5(pattern_str.encode()).hexdigest()

def remove_duplicates():
    """Remove duplicate patterns from learned_patterns.json"""

    # Backup the original file
    backup_name = f"learned_patterns_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    shutil.copy2("learned_patterns.json", backup_name)
    print(f"Created backup: {backup_name}")

    # Load the patterns
    with open("learned_patterns.json", 'r') as f:
        data = json.load(f)

    total_patterns_before = 0
    total_patterns_after = 0
    duplicates_removed = 0

    # Process each digit
    for digit, patterns in data.items():
        original_count = len(patterns)
        total_patterns_before += original_count

        print(f"Processing digit {digit}: {original_count} patterns")

        # Track unique patterns using hashes
        seen_hashes = set()
        unique_patterns = []

        for pattern_data in patterns:
            pattern = pattern_data['pattern']
            pattern_hash = pattern_to_hash(pattern)

            if pattern_hash not in seen_hashes:
                seen_hashes.add(pattern_hash)
                unique_patterns.append(pattern_data)
            else:
                duplicates_removed += 1
                print(f"  Removed duplicate pattern for digit {digit}")

        # Update the data
        data[digit] = unique_patterns
        new_count = len(unique_patterns)
        total_patterns_after += new_count

        if original_count != new_count:
            print(f"  Digit {digit}: {original_count} -> {new_count} patterns ({original_count - new_count} duplicates removed)")

    # Save the cleaned data
    with open("learned_patterns.json", 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nSummary:")
    print(f"Total patterns before: {total_patterns_before}")
    print(f"Total patterns after: {total_patterns_after}")
    print(f"Total duplicates removed: {duplicates_removed}")
    print(f"Cleaned file saved as learned_patterns.json")
    print(f"Original backed up as {backup_name}")

if __name__ == "__main__":
    remove_duplicates()
