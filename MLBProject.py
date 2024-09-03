import numpy as np
import os
import re
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from tensorflow.python.keras.layers import GlobalMaxPooling1D
from itertools import islice
import argparse

# Create vocabulary dictionaries to map characters to indices
DNA_VOCAB = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
RNA_VOCAB = {'A': 0, 'C': 1, 'G': 2, 'U': 3}

REAL_RESULTS_FILE_NAME = 'RBP1.txt'  # TODO: REMOVE WHEN WE SUBMIT
MAX_LINES_TO_READ = 50000
DEFAULT_CHAR_PROBABILITY = 0.25

def parse_arguments():
    """
    Parse command line arguments and return them as a list.
    """
    parser = argparse.ArgumentParser(description="Your program's description here")
    parser.add_argument("output_file", type=str, help="Output file name")
    parser.add_argument("rncmpt_file", type=str, help="RNCMPT file name")
    parser.add_argument("input_file", type=str, help="Input file name")
    parser.add_argument("rbns_files", type=str, nargs='+', help="RBNS file names")
    return parser.parse_args()


def load_sequences_from_file(file_name, num_lines=None):
    """
    Read sequences from a file where each line contains a sequence and a number.
    Return a list of all sequences in the file.

    :param file_name: The name of the file to read sequences from.
    :param num_lines: The number of lines to read (None means read all lines).
    """
    sequences = []
    with open(file_name, 'r') as file:
        lines_to_read = islice(file, num_lines) if num_lines is not None else file
        for line in lines_to_read:
            sequence, _ = line.strip().split('\t')
            sequences.append(sequence)
    return sequences


def preprocess_sequence(sequence, vocabulary):
    """
    Convert a single sequence to numerical format using one-hot encoding based on the provided vocabulary.
    Return a tuple of converted sequences.
    """
    sequence_length = len(sequence)
    num_characters = len(vocabulary)
    one_hot_sequence = [[0] * num_characters for _ in range(sequence_length)]

    for position, character in enumerate(sequence):
        if character.upper() == "N":
            one_hot_sequence[position] = [DEFAULT_CHAR_PROBABILITY, DEFAULT_CHAR_PROBABILITY, DEFAULT_CHAR_PROBABILITY, DEFAULT_CHAR_PROBABILITY]  # Special handling for 'N'
            continue
        character_index = vocabulary.get(character, -1)
        if character_index != -1:
            one_hot_sequence[position][character_index] = 1

    one_hot_sequence_tuple = tuple(tuple(row) for row in one_hot_sequence)
    return one_hot_sequence_tuple



def extract_number(file_name, separator='_'):
    """
    Extract the number (concentration) from a file name. Assumes the number comes after the specified separator.
    """
    # Use regular expression to extract the numeric part from the file name
    pattern = f'{separator}(\d+)'
    match = re.search(pattern, file_name)
    if match:
        return int(match.group(1))
    return -1  # Return a default value if no number is found


def write_results_to_file(output_filename, results_array):
    """
    Write the results to an output file.
    """
    with open(output_filename, 'w') as output_file:
        for result in results_array.flat:
            output_file.write(str(result) + '\n')



def calculate_pearson_correlation(model_results):
    """
    Calculate the Pearson correlation between the model's results and the true results, if available.
    If the true results file is not available, return None.
    """
    try:
        with open(REAL_RESULTS_FILE_NAME, 'r') as file:
            lines = file.readlines()

        real_results_for_new_sequences = [float(line.strip()) for line in lines]
        real_array = np.array(real_results_for_new_sequences)

        # Calculate the Pearson correlation coefficient and p-value
        correlation_coefficient, p_value = pearsonr(model_results, real_array)

        return correlation_coefficient
    except FileNotFoundError:
        # Handle the case when the true results file is not available
        return None



if __name__ == "__main__":
    args = parse_arguments()

    output_file = args.output_file
    rncmpt_file = args.rncmpt_file
    input_file = args.input_file
    rbns_files = args.rbns_files

    f1 = open(rncmpt_file, "r")
    rnacmpt_sequences = f1.read().splitlines()

    input_and_concentrations_file_names = [input_file] + rbns_files

    encoded_sequences = {}

    # Define the number of classes (= number of files)
    num_classes = len(input_and_concentrations_file_names)

    # Sort file_names so that "input" file comes first, and the rest in ascending order of concentration
    input_and_concentrations_file_names = sorted(input_and_concentrations_file_names, key=extract_number)

    padding_value = (DEFAULT_CHAR_PROBABILITY, DEFAULT_CHAR_PROBABILITY, DEFAULT_CHAR_PROBABILITY, DEFAULT_CHAR_PROBABILITY)  # Equal distribution for each base

    for i, file_name in enumerate(input_and_concentrations_file_names):
        sequences = []
        try:
            sequences = load_sequences_from_file(file_name, num_lines=MAX_LINES_TO_READ)
        except Exception as e:
            print(f"Can't load sequences from file: {file_name}. Error: {e}")
            num_classes -= 1
            continue

        for sequence in sequences:
            encoded_sequence = preprocess_sequence(sequence, DNA_VOCAB)
            if encoded_sequence in encoded_sequences:
                index = list(encoded_sequences.keys()).index(encoded_sequence)
                encoded_sequences[encoded_sequence][i] = 1
            else:
                encoded_sequences[encoded_sequence] = [0] * num_classes
                encoded_sequences[encoded_sequence][i] = 1

    sequences = list(encoded_sequences.keys())
    labels = list(encoded_sequences.values())

    max_sequence_length = max(len(seq) for seq in rnacmpt_sequences)

    padded_seq = [seq + (padding_value,) * (max_sequence_length - len(seq)) for seq in sequences]

    # Combine sequences and labels into data and labels arrays
    data = np.array(padded_seq)
    encoded_labels = np.array(labels, dtype=int)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(data, encoded_labels, test_size=0.2, random_state=42)

    # Build the model
    sequence_length = data.shape[1]
    label_dimension = encoded_labels.shape[1]
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(sequence_length, 4)),
        GlobalMaxPooling1D(),
        Dense(units=128, activation='relu'),
        Dense(units=num_classes)  # Output layer for multi-class classification
    ])

    # Compile the model
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

    # Evaluate the model
    loss, mae = model.evaluate(X_val, y_val)

    processed_rnacmpt_sequences = []

    for seq in rnacmpt_sequences:
        processed = preprocess_sequence(seq, RNA_VOCAB)
        processed_rnacmpt_sequences.append(processed)

    processed_rnacmpt_sequences = list(processed_rnacmpt_sequences)

    padded_sequences = [seq + (padding_value,) * (max_sequence_length - len(seq)) for seq in processed_rnacmpt_sequences]

    data_sequences = np.array(padded_sequences)
    predictions = model.predict(data_sequences)

    # Convert the vector to a float value
    converted_predictions = [sum(array[-2:]) - array[0] for array in predictions]

    results_array = np.array(converted_predictions)
    write_array_to_file(output_file, results_array)

    # TODO: REMOVE WHEN WE SUBMIT
    print("Pearson Correlation: ", calculate_pearson_correlation(results_array))
