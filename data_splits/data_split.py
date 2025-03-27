import json
import random

def merge_json_files(file1, file2):
    # Load JSON data from both files
    with open(file1, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
    with open(file2, 'r', encoding='utf-8') as f:
        data2 = json.load(f)
    # Merge the dictionaries (if duplicate keys exist, data2's value will override data1's)
    merged_data = {**data1, **data2}
    return merged_data

def random_split(data, train_ratio=0.8, seed=None):
    # Set seed for reproducibility if provided
    if seed is not None:
        random.seed(seed)
    keys = list(data.keys())
    random.shuffle(keys)  # Randomize the keys
    split_index = int(len(keys) * train_ratio)
    train_keys = keys[:split_index]
    test_keys = keys[split_index:]
    
    train_data = {k: data[k] for k in train_keys}
    test_data = {k: data[k] for k in test_keys}
    return train_data, test_data

def main():
    sample_file = 'sample.json'
    validation_file = 'validation.json'
    
    # Merge the JSON files
    merged_data = merge_json_files(sample_file, validation_file)
    print(f"Merged data contains {len(merged_data)} items.")
    
    # Perform a randomized split into training (80%) and testing (20%) datasets.
    train_data, test_data = random_split(merged_data, train_ratio=0.8, seed=42)
    print(f"Training set: {len(train_data)} items, Testing set: {len(test_data)} items.")
    
    # Write the training data to a file
    with open('training_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4)
    
    # Write the testing data to a file
    with open('testing_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4)
    
    print("Randomized split datasets have been successfully created.")

if __name__ == '__main__':
    main()
