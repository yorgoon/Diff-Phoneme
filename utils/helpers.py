# helpers.py

import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.patches as mpatches
import editdistance
from collections import defaultdict

# If not already defined, import or define these variables in your config.py or another module
# from config import device, phoneme_vocab, phoneme_groups, phoneme_to_group
from config import DEVICE, PHONEME_VOCAB, PHONEME_GROUPS, PHONEME_TO_GROUP, MAX_SEQ_LEN

# =========================
# Helper Functions and Classes
# =========================

def update_vocabulary_and_dict(train_labels, test_labels, word_list, phoneme_vocab, phoneme_dict):
    """
    Update the phoneme vocabulary and dictionary based on the provided train and test labels.

    Args:
    - train_labels (list): The list of labels from the training set.
    - test_labels (list): The list of labels from the test set.
    - word_list (list): The list of words corresponding to the labels.
    - phoneme_vocab (dict): The original phoneme vocabulary (mapping from phoneme string to index).
    - phoneme_dict (dict): The dictionary mapping labels to sequences of phoneme indices.

    Returns:
    - updated_word_list (list): Updated word list corresponding to the train and test labels.
    - updated_phoneme_vocab (dict): Updated phoneme vocabulary containing only used phonemes.
    - updated_phoneme_dict (dict): Updated phoneme dictionary using the new phoneme indices.
    """
    
    # Convert the train and test labels into tensors
    train_labels_tensor = torch.tensor(train_labels)
    test_labels_tensor = torch.tensor(test_labels)

    # Combine train and test labels
    combined_labels = torch.cat((train_labels_tensor, test_labels_tensor)).unique().tolist()

    # Collect all the phonemes used in the phoneme sequences corresponding to the combined labels
    used_phonemes = set()
    for label in combined_labels:
        phoneme_sequence = phoneme_dict[label]
        for phoneme_index in phoneme_sequence:
            # Reverse lookup the phoneme string from its index
            phoneme_str = next((k for k, v in phoneme_vocab.items() if v == phoneme_index), None)
            if phoneme_str is not None:
                used_phonemes.add(phoneme_str)

    # Create the new phoneme vocabulary by filtering only the used phonemes from the original vocab
    updated_phoneme_vocab = {phoneme: idx for idx, phoneme in enumerate(sorted(used_phonemes))}

    # Create a new phoneme dictionary where phonemes are represented by the updated indices
    updated_phoneme_dict = {}
    for label in combined_labels:
        original_phoneme_sequence = phoneme_dict[label]
        updated_phoneme_sequence = [
            updated_phoneme_vocab[next((k for k, v in phoneme_vocab.items() if v == phoneme), None)] 
            for phoneme in original_phoneme_sequence
        ]
        updated_phoneme_dict[label] = updated_phoneme_sequence

    # Create updated word list based on combined labels
    updated_word_list = [word_list[label] for label in combined_labels]

    return updated_word_list, updated_phoneme_vocab, updated_phoneme_dict

    # Create updated word list based on combined labels
    updated_word_list = [word_list[label] for label in combined_labels]

    return updated_word_list, updated_phoneme_vocab, updated_phoneme_dict

def compute_phoneme_error_rate(true_phonemes, predicted_phonemes):
    """
    Compute the Phoneme Error Rate (PER) for a batch of sequences.

    Args:
    - true_phonemes (List[List[int]]): List of lists where each sublist is a sequence of true phoneme indices.
    - predicted_phonemes (List[List[int]]): List of lists where each sublist is a sequence of predicted phoneme indices.

    Returns:
    - float: The average Phoneme Error Rate (PER) for the batch.
    """
    total_distance = 0
    total_phonemes = 0

    for true_seq, pred_seq in zip(true_phonemes, predicted_phonemes):
        if not true_seq:
            continue
        if not pred_seq:
            total_distance += len(true_seq)
            total_phonemes += len(true_seq)
            continue

        # Compute the Levenshtein distance (edit distance)
        edit_distance = editdistance.eval(true_seq, pred_seq)
        total_distance += edit_distance
        total_phonemes += len(true_seq)

    per = total_distance / total_phonemes if total_phonemes > 0 else float('inf')
    return per

def evaluate_model_with_mds(model, seq_model, dataloader, phoneme_vocab, phoneme_groups, phoneme_to_group, device):
    """
    Evaluate the model and plot MDS of the hidden states.

    Args:
    - model (nn.Module): The main model.
    - seq_model (nn.Module): The sequence model.
    - classifier (nn.Module): The classifier model.
    - dataloader (DataLoader): The data loader.
    - phoneme_vocab (dict): Phoneme vocabulary.
    - phoneme_groups (dict): Dictionary of phoneme groups.
    - phoneme_to_group (dict): Mapping from phoneme to its group.
    - device (torch.device): The device to run computations on.

    Returns:
    - None
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    model.eval()
    seq_model.eval()

    total_word_accuracy = 0
    total_phoneme_error_rate = 0
    total_sequences = 0

    all_true_phonemes = []
    all_predicted_phonemes = []

    # Assign colors to each phoneme group
    group_colors = {
        'vowels': '#1f78b4',         # Blue
        'diphthongs': '#33a02c',     # Green
        'stops_plosives': '#fb9a99', # Light pink-red
        'affricates': '#e31a1c',     # Red
        'fricatives': '#6a3d9a',     # Purple
        'nasals': '#ff7f00',         # Orange
        'liquids': '#b2df8a',        # Light green
        'semivowels': '#a6cee3',     # Light blue
        'silence': '#fdbf6f',        # Yellow-orange
    }

    # Map phonemes to colors
    phoneme_color_map = {phoneme: group_colors[group] for phoneme, group in phoneme_to_group.items()}

    all_hidden_states = []
    all_phoneme_labels = []

    rest_index = phoneme_vocab['sil']  # Get the index of "sil" in the phoneme vocabulary
    eos_index = phoneme_vocab['sil']   # Assuming 'sil' is used as the EOS token

    with torch.no_grad():
        for eeg_batch, phoneme_batch, label in dataloader:
            eeg_batch, phoneme_batch, label = eeg_batch.to(device), phoneme_batch.to(device), label.to(device)

            # Forward pass through models
            dn30 = model(eeg_batch.unsqueeze(1))  # Add channel dimension
            seq_outputs, hidden_states = seq_model(dn30)
            _, predicted_indices = torch.max(seq_outputs, dim=2)

            for i in range(eeg_batch.size(0)):
                true_sequence = phoneme_batch[i].tolist()
                pred_sequence = predicted_indices[i].tolist()

                # Filter out "sil" phonemes
                filtered_true_sequence = [p for p in true_sequence if p != rest_index and p != eos_index]
                filtered_pred_sequence = [p for p in pred_sequence if p != rest_index and p != eos_index]

                min_length = min(len(filtered_true_sequence), len(filtered_pred_sequence))
                if min_length > 0:
                    all_true_phonemes.extend(filtered_true_sequence[:min_length])
                    all_predicted_phonemes.extend(filtered_pred_sequence[:min_length])

                    # Store hidden states for MDS plotting
                    all_hidden_states.extend(hidden_states[i, :min_length].cpu().numpy())
                    all_phoneme_labels.extend(filtered_true_sequence[:min_length])

                    # Phoneme Error Rate (PER)
                    total_phoneme_error_rate += compute_phoneme_error_rate(
                        [filtered_true_sequence[:min_length]],
                        [filtered_pred_sequence[:min_length]]
                    )
                total_sequences += 1

    # Convert lists to numpy arrays
    all_hidden_states = np.array(all_hidden_states)
    all_phoneme_labels = np.array(all_phoneme_labels)

    if len(all_hidden_states) == 0:
        print("No data to plot.")
        return

    # Average hidden states for each phoneme class
    phoneme_sum = defaultdict(lambda: np.zeros(all_hidden_states.shape[1]))
    phoneme_count = defaultdict(int)

    for i, label in enumerate(all_phoneme_labels):
        phoneme_sum[label] += all_hidden_states[i]
        phoneme_count[label] += 1

    average_hidden_states = np.array([phoneme_sum[label] / phoneme_count[label] for label in phoneme_sum.keys()])
    average_phoneme_labels = np.array(list(phoneme_sum.keys()))

    # Apply MDS to the averaged embeddings
    mds = MDS(n_components=2, random_state=42)
    hidden_states_2d = mds.fit_transform(average_hidden_states)

    # Calculate averages
    avg_per = total_phoneme_error_rate / total_sequences if total_sequences > 0 else 0

    print(f"Phoneme Error Rate (PER): {avg_per:.4f}")

    # # Plot the MDS result
    # plt.figure(figsize=(14, 8))

    # legend_handles = []

    # # Use phoneme groups to assign colors and prepare for the legend
    # for i, phoneme_idx in enumerate(average_phoneme_labels):
    #     phoneme_str = next((k for k, v in phoneme_vocab.items() if v == phoneme_idx), None)
    #     color = phoneme_color_map.get(phoneme_str, 'black')  # Default to black if not found
    #     plt.text(
    #         hidden_states_2d[i, 0],
    #         hidden_states_2d[i, 1],
    #         phoneme_str,
    #         fontsize=12,
    #         ha='center',
    #         va='center',
    #         alpha=0.8,
    #         color=color
    #     )

    #     group_name = phoneme_to_group.get(phoneme_str, 'Unknown')
    #     if group_name and all(handle.get_label() != group_name for handle in legend_handles):
    #         legend_handles.append(mpatches.Patch(color=color, label=group_name))

    # plt.title('MDS of Averaged Model Output Features with Phoneme Labels', fontsize=16)
    # plt.xlabel('MDS Dimension 1', fontsize=14)
    # plt.ylabel('MDS Dimension 2', fontsize=14)

    # # Add the legend outside the plot
    # plt.legend(
    #     handles=legend_handles,
    #     title="Phoneme Groups",
    #     fontsize=12,
    #     title_fontsize=14,
    #     loc='upper left',
    #     bbox_to_anchor=(1, 1)
    # )

    # plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the right margin to fit the legend
    # plt.show()

    # # Exclude 'sil' from consideration
    # phoneme_vocab_poped = phoneme_vocab.copy()
    # phoneme_vocab_poped.pop('sil', None)

    # plot_confusion_matrix(all_true_phonemes, all_predicted_phonemes, phoneme_vocab_poped, phoneme_groups)

def plot_confusion_matrix(true_labels, predicted_labels, phoneme_vocab, phoneme_groups):
    """
    Plots the confusion matrix for the predicted and true labels, focusing only on the relevant phonemes
    and highlighting similar phonemes together.

    Args:
    - true_labels (list): List of true phoneme indices.
    - predicted_labels (list): List of predicted phoneme indices.
    - phoneme_vocab (dict): Dictionary mapping phoneme strings to their corresponding indices.
    - phoneme_groups (dict): Dictionary where keys are group names and values are lists of phoneme names belonging to that group.
    """
    import matplotlib.pyplot as plt

    # Create a sorted list of all phonemes grouped by their similarity
    ordered_phonemes = []
    for group_name, phonemes in phoneme_groups.items():
        for phoneme in phonemes:
            if phoneme in phoneme_vocab:
                ordered_phonemes.append(phoneme)

    # Create a list of relevant phoneme indices in the order of the grouped phonemes
    ordered_classes = [phoneme_vocab[p] for p in ordered_phonemes]

    if not ordered_classes:
        raise ValueError("No relevant phonemes found in the dataset. 'ordered_classes' is empty.")

    # Compute confusion matrix for the relevant phonemes
    cm = confusion_matrix(true_labels, predicted_labels, labels=ordered_classes)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(12, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ordered_phonemes)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', ax=ax)
    plt.title('Confusion Matrix of Phoneme Predictions (Grouped by Similarity)')

    # Highlight similar phoneme groups
    for group_name, phonemes in phoneme_groups.items():
        group_indices = [ordered_classes.index(phoneme_vocab[p]) for p in phonemes if p in phoneme_vocab and phoneme_vocab[p] in ordered_classes]
        if not group_indices:
            continue

        min_index = min(group_indices)
        max_index = max(group_indices)

        # Draw rectangle to highlight group
        rect = plt.Rectangle(
            (min_index - 0.5, min_index - 0.5),
            max_index - min_index + 1,
            max_index - min_index + 1,
            fill=False,
            edgecolor='red',
            linewidth=2
        )
        ax.add_patch(rect)

    plt.show()

def phonetic_distance(seq1, seq2):
    """
    Calculate a phonetic distance between two phoneme sequences with weighted similarity.

    Args:
    - seq1 (list): First phoneme sequence.
    - seq2 (list): Second phoneme sequence.

    Returns:
    - float: Phonetic distance.
    """
    # Phonetic category map with some weight adjustments
    category_map = {
        'vowels': ['iy', 'ih', 'eh', 'ae', 'aa', 'ao', 'uh', 'uw', 'ah', 'er'],
        'diphthongs': ['ey', 'ay', 'aw', 'oy', 'ow'],
        'stops_plosives': ['p', 'b', 't', 'd', 'k', 'g'],
        'affricates': ['ch', 'jh'],
        'fricatives': ['f', 'v', 'th', 'dh', 's', 'z', 'sh', 'zh', 'hh'],
        'nasals': ['m', 'n', 'ng'],
        'liquids': ['l', 'r'],
        'semivowels': ['w', 'y'],
        'silence': ['sil'],
    }

    # Mapping within categories for weighted substitutions
    phoneme_weights = {
        ('iy', 'ih'): 0.5, ('eh', 'ih'): 0.6,  # Similar vowels
        ('p', 'b'): 0.4, ('t', 'd'): 0.4,     # Similar stops
        ('f', 'v'): 0.6, ('s', 'z'): 0.6,     # Similar fricatives
        # Add more nuanced mappings as needed
    }

    def get_category(phoneme):
        for category, phonemes in category_map.items():
            if phoneme == phoneme or phoneme in phonemes:
                return category
        return None

    # Initialize the distance
    distance = 0

    # Iterate over the sequences and compute distance
    for p1, p2 in zip(seq1, seq2):
        if p1 == p2:
            continue
        elif (p1, p2) in phoneme_weights:
            distance += phoneme_weights[(p1, p2)]  # Weighted similarity
        elif get_category(p1) != get_category(p2):
            distance += 1  # Full distance if in different categories
        else:
            distance += 0.8  # Partial distance if in the same category but different phoneme

    # Add extra cost for length differences
    distance += abs(len(seq1) - len(seq2))

    return distance

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_sequence = False
        self.index = None  # This will store the index of the word/sequence in your dictionary

class Trie:
    def __init__(self, phoneme_dict):
        self.root = TrieNode()
        self.phoneme_dict = phoneme_dict
        self.build_trie()

    def build_trie(self):
        """
        Build the Trie using the provided phoneme dictionary.
        """
        for index, (word, phoneme_sequences) in enumerate(self.phoneme_dict.items()):
            # Loop through each pronunciation for the word
            for phoneme_sequence in phoneme_sequences:
                self.insert(phoneme_sequence, index)

    def insert(self, phoneme_sequence, index):
        """
        Insert a phoneme sequence into the Trie.

        Args:
        - phoneme_sequence (list): A list of phonemes representing a word.
        - index (int): The index of the word/sequence in the dictionary.
        """
        current = self.root
        # Convert the phoneme sequence to a tuple to ensure it's hashable
        phoneme_sequence = tuple(phoneme_sequence)
        
        for phoneme in phoneme_sequence:
            if phoneme not in current.children:
                current.children[phoneme] = TrieNode()
            current = current.children[phoneme]
        current.is_end_of_sequence = True
        current.index = index

    def search_closest(self, phoneme_sequence):
        """
        Search for the closest matching phoneme sequence in the Trie using weighted phonetic distance.

        Args:
        - phoneme_sequence (list): The phoneme sequence to search for.

        Returns:
        - int: The index of the closest matching word/sequence in the dictionary.
        """
        min_distance = float('inf')
        closest_index = None

        for index, (word, seq) in enumerate(self.phoneme_dict.items()):
            # Calculate phonetic distance
            distance = phonetic_distance(phoneme_sequence, seq)
            if distance < min_distance:
                min_distance = distance
                closest_index = index

        return closest_index


def test_trie(trie, test_sequences, word_list_cmu):
    """
    Function to test the Trie and print results.

    Args:
    - trie (Trie): The Trie object.
    - test_sequences (list): List of test phoneme sequences.
    - word_list_cmu (list): List of words from the CMU dictionary.
    """
    for i, sequence in enumerate(test_sequences):
        closest_index = trie.search_closest(sequence)
        if closest_index is not None:
            print(f"Test Sequence {i+1}: {sequence} => Closest Word: {word_list_cmu[closest_index]}")
        else:
            print(f"Test Sequence {i+1}: {sequence} => No close match found.")

def load_cmu_dict(filepath="cmudict-0.7b.txt"):
    """
    Load the CMU Pronouncing Dictionary from a file.

    Args:
    - filepath (str): Path to the CMU Pronouncing Dictionary file.

    Returns:
    - cmu_dict (dict): Dictionary mapping words to lists of phonemes.
    """
    cmu_dict = {}

    with open(filepath, "r", encoding='utf-8') as f:
        for line in f:
            # Skip comments and empty lines
            if line.startswith(";;;") or line.strip() == "":
                continue

            # Split word and phonemes
            parts = line.strip().split("  ")
            if len(parts) != 2:
                continue  # Skip malformed lines
            word = parts[0].lower()  # Convert word to lowercase
            phonemes = parts[1].split()

            # Remove stress markers from phonemes and convert to lowercase
            phonemes = [ph.lower().strip('012') for ph in phonemes]

            # Handle multiple pronunciations (e.g., WORD(1), WORD(2))
            word = word.split('(')[0]

            # Add to the dictionary
            if word in cmu_dict:
                cmu_dict[word].append(phonemes)
            else:
                cmu_dict[word] = [phonemes]

    return cmu_dict