# data_loader.py

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import mat73  # For loading MATLAB v7.3 .mat files

# Import configurations
from config import (
    DEVICE,
    SEED,
    PHONEME_VOCAB,
    WORD_TO_PHONEME_DICT,
    MAX_SEQ_LEN,
    ROOT_DIR,
    BATCH_SIZE,
)

# Set random seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)


class EEGPhonemeDataset(Dataset):
    """
    Custom Dataset for loading EEG data and corresponding phoneme sequences.
    """

    def __init__(self, eeg_data, labels, phoneme_dict):
        self.eeg_data = eeg_data
        self.labels = labels
        self.phoneme_dict = phoneme_dict
        self.eos_index = PHONEME_VOCAB["sil"]

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        eeg = self.eeg_data[idx]
        label = self.labels[idx].item() if isinstance(self.labels[idx], torch.Tensor) else self.labels[idx]

        if label not in self.phoneme_dict:
            raise KeyError(f"Label {label} not found in phoneme_dict.")

        phoneme_seq = self.phoneme_dict[label]

        return torch.FloatTensor(eeg), torch.LongTensor(phoneme_seq), label


def zscore_norm(data, mean=None, std=None, epsilon=1e-8):
    """
    Applies Z-score normalization to the data.

    Args:
        data (torch.Tensor): The data to normalize.
        mean (torch.Tensor, optional): Precomputed mean. If None, computed from data.
        std (torch.Tensor, optional): Precomputed std. If None, computed from data.
        epsilon (float, optional): Small value to avoid division by zero.

    Returns:
        norm_data (torch.Tensor): Normalized data.
    """
    if mean is None or std is None:
        mean = data.mean(dim=0, keepdim=True)
        std = data.std(dim=0, keepdim=True)

    norm_data = (data - mean) / (std + epsilon)
    return norm_data


def load_data(root_dir, subject, session):
    """
    Loads EEG data and labels from a .mat file.

    Args:
        root_dir (str): Directory containing the data files.
        subject (int): Subject identifier.
        session (int): Session number.

    Returns:
        X (torch.Tensor): EEG data.
        Y (torch.Tensor): Labels.
    """
    filename = f"s{subject}_sess{session}.mat"
    file_path = os.path.join(root_dir, filename)

    file = mat73.loadmat(file_path)

    # Extract and preprocess data
    X = np.float32(file["X"])
    X = np.transpose(X, (2, 1, 0))  # Adjust dimensions if necessary
    Y = np.int_(file["y"]) - 1  # Adjust labels if necessary

    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)

    return X, Y


class CollateFn:
    """
    Collate function to be used with DataLoader.
    Pads phoneme sequences and stacks EEG data.
    """

    def __init__(self, phoneme_vocab):
        self.padding_value = phoneme_vocab["sil"]  # Ensure 'sil' exists in phoneme_vocab

    def __call__(self, batch):
        eeg_batch, phoneme_batch, label_batch = zip(*batch)

        # Pad phoneme sequences to the length of the longest sequence in the batch
        phoneme_batch_padded = pad_sequence(
            phoneme_batch, batch_first=True, padding_value=self.padding_value
        ).long()

        # Stack EEG data
        eeg_batch_stacked = torch.stack(eeg_batch, dim=0)

        # Convert label_batch to a tensor
        label_batch_stacked = torch.tensor(label_batch, dtype=torch.long)

        return eeg_batch_stacked, phoneme_batch_padded, label_batch_stacked


def train_test_set(X, Y, target_labels=None, group_size=4, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets.

    Args:
        X (torch.Tensor): EEG data.
        Y (torch.Tensor): Labels.
        target_labels (list, optional): Specific labels to include.
        group_size (int, optional): Group size for balanced splitting.
        test_size (float, optional): Proportion of data to use as test set.
        random_state (int, optional): Random seed for shuffling.

    Returns:
        X_train, Y_train, X_test, Y_test
    """
    if target_labels is None:
        target_labels = torch.unique(Y)

    X_train_list = []
    Y_train_list = []
    X_test_list = []
    Y_test_list = []

    for label in target_labels:
        indices = (Y == label).nonzero(as_tuple=True)[0]

        # Trim indices to be multiple of group_size
        indices = indices[: len(indices) // group_size * group_size]
        indices = indices.view(-1, group_size)  # Group samples

        # Shuffle indices
        generator = torch.Generator().manual_seed(random_state)
        indices = indices[torch.randperm(indices.size(0), generator=generator)]

        # Split into train and test
        split = int((1 - test_size) * indices.size(0))
        train_indices = indices[:split].flatten()
        test_indices = indices[split:].flatten()

        # Append to lists
        X_train_list.append(X[train_indices])
        Y_train_list.append(Y[train_indices])
        X_test_list.append(X[test_indices])
        Y_test_list.append(Y[test_indices])

    X_train = torch.cat(X_train_list, dim=0)
    Y_train = torch.cat(Y_train_list, dim=0)
    X_test = torch.cat(X_test_list, dim=0)
    Y_test = torch.cat(Y_test_list, dim=0)

    print(f"Train set: {X_train.shape}, {Y_train.shape}")
    print(f"Test set: {X_test.shape}, {Y_test.shape}")

    return X_train, Y_train, X_test, Y_test


def get_dataloaders(
    root_dir,
    subject,
    session,
    batch_size,
    phoneme_vocab,
    phoneme_dict,
    test_size=0.2,
    random_state=42,
):
    """
    Creates DataLoader objects for training and testing.

    Args:
        root_dir (str): Directory containing data files.
        subject (int): Subject identifier.
        session (int): Session number.
        batch_size (int): Batch size for DataLoader.
        phoneme_vocab (dict): Phoneme vocabulary.
        phoneme_dict (dict): Mapping from labels to phoneme sequences.
        test_size (float, optional): Proportion of data to use as test set.
        random_state (int, optional): Random seed.

    Returns:
        train_loader, test_loader
    """
    # Load and normalize data
    X, Y = load_data(root_dir, subject, session)

    # Split data into training and testing sets
    X_train, Y_train, X_test, Y_test = train_test_set(
        X, Y, test_size=test_size, random_state=random_state
    )

    # Normalize training data and compute mean and std
    X_train, mean, std = zscore_norm(X_train)

    # Normalize test data using training mean and std
    X_test, _, _ = zscore_norm(X_test, mean=mean, std=std)

    # Create datasets
    train_dataset = EEGPhonemeDataset(X_train, Y_train, phoneme_dict)
    test_dataset = EEGPhonemeDataset(X_test, Y_test, phoneme_dict)

    # Initialize collate function
    collate_fn = CollateFn(phoneme_vocab)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_loader, test_loader
