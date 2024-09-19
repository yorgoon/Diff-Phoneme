# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np  # Added missing import

from data import load_data, zscore_norm, EEGPhonemeDataset, CollateFn, train_test_set
from utils.helpers import update_vocabulary_and_dict, Trie, evaluate_model_with_mds, load_cmu_dict
from models import EEGPhonemeSeq2Seq, EEGNet

from config import (
    DEVICE,
    SEED,
    PHONEME_VOCAB,
    PHONEME_GROUPS,
    PHONEME_TO_GROUP,
    WORD_LIST,
    WORD_TO_PHONEME_DICT,
    MAX_SEQ_LEN,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    SUBJECT,
    SESSION,
    TARGET_LABELS,
    MODEL_SAVE_PATH,
    OUTPUT_PATH,
    ROOT_DIR,
)

# Set random seed
torch.manual_seed(SEED)
np.random.seed(SEED)

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Data directory and batch size
img_dir = ROOT_DIR
batch_size = BATCH_SIZE

# Load CMU Pronouncing Dictionary
cmu_dict = load_cmu_dict(filepath="cmudict-0.7b.txt")  # Ensure the file path is correct

# Loop over subjects
for subject in SUBJECT:
    print(f"Processing Subject {subject}")

    # Load data for the current subject
    X, Y = load_data(root_dir=img_dir, subject=subject, session=SESSION)

    # Normalize data
    X = zscore_norm(X)
    Y = Y.long()  # Ensure labels are of type long

    # Define train and test labels (using all labels)
    train_labels = TARGET_LABELS
    test_labels = TARGET_LABELS

    # Update the vocabulary and phoneme dictionary
    updated_word_list, updated_phoneme_vocab, updated_phoneme_dict = update_vocabulary_and_dict(
        train_labels,
        test_labels,
        WORD_LIST,
        PHONEME_VOCAB,
        WORD_TO_PHONEME_DICT
    )

    print(type(updated_word_list))
    print(type(updated_phoneme_vocab))
    print(type(updated_phoneme_dict))

    # Ensure words are lowercase for comparison with CMU dictionary keys
    common_words_lower = [word.lower() for word in updated_word_list]

    # Create a smaller CMU dictionary for the common words
    common_cmu_dict = {word: cmu_dict[word] for word in common_words_lower if word in cmu_dict}

    print(common_cmu_dict)

    # Initialize the Trie with the common CMU dictionary
    trie = Trie(phoneme_dict=common_cmu_dict)

    # Create the train and test sets using the custom train_test_set function
    X_train, Y_train, X_test, Y_test = train_test_set(
        X, Y, target_labels=train_labels, test_size=0.2, random_state=SEED
    )

    # Create datasets
    train_dataset = EEGPhonemeDataset(X_train, Y_train, updated_phoneme_dict)
    test_dataset = EEGPhonemeDataset(X_test, Y_test, updated_phoneme_dict)

    # Initialize collate function
    collate_fn = CollateFn(updated_phoneme_vocab)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Model parameters
    num_channels = X.shape[1]  # Number of EEG channels
    hidden_size = 128
    output_size = len(updated_phoneme_vocab)
    num_layers = 2
    max_output_length = MAX_SEQ_LEN

    # Initialize models
    encoder_model = EEGNet(output_features=hidden_size).to(DEVICE)

    seq_model = EEGPhonemeSeq2Seq(
        input_size=hidden_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        max_output_length=max_output_length
    ).to(DEVICE)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = optim.Adam([
        {'params': encoder_model.parameters(), 'lr': LEARNING_RATE},
        {'params': seq_model.parameters(), 'lr': LEARNING_RATE}
    ])

    # Training loop
    for epoch in range(NUM_EPOCHS):
        encoder_model.train()
        seq_model.train()
        total_loss = 0.0

        for eeg_batch, phoneme_batch, label_batch in train_loader:
            eeg_batch = eeg_batch.to(DEVICE)  # Shape: (batch_size, num_channels, seq_length)
            phoneme_batch = phoneme_batch.to(DEVICE)  # Shape: (batch_size, seq_length_phoneme)
            label_batch = label_batch.to(DEVICE)

            # Forward pass
            # Pass the EEG data through the encoder model
            eeg_batch = eeg_batch.unsqueeze(1)  # (batch_size, 1, num_channels, seq_length)
            eeg_features = encoder_model(eeg_batch)  # Output shape: (batch_size, hidden_size, seq_length_eeg)

            # Permute eeg_features to match expected input shape for seq_model
            # eeg_features = eeg_features.permute(0, 2, 1)  # Now shape: (batch_size, seq_length_eeg, hidden_size)

            # Pass the features through the sequence model
            seq_outputs, hidden_states = seq_model(eeg_features)

            # seq_outputs: (batch_size, target_seq_length, output_size)
            # phoneme_batch: (batch_size, target_seq_length)

            # Assuming seq_outputs has shape (batch_size, seq_length, num_classes)
            batch_size, seq_length, num_classes = seq_outputs.shape

            # Ensure seq_outputs and phoneme_batch have matching sequence lengths
            target_seq_length = phoneme_batch.size(1)  # Target sequence length

            # Truncate seq_outputs to match target_seq_length, if necessary
            if seq_length > target_seq_length:
                seq_outputs = seq_outputs[:, :target_seq_length, :]
            else:
                # Optionally, pad phoneme_batch if it's shorter (if this case applies)
                phoneme_batch = phoneme_batch[:, :seq_length]

            # After truncation or padding, shapes should match
            assert seq_outputs.size(1) == phoneme_batch.size(1), "Mismatch in sequence lengths after truncation or padding."

            # Flatten seq_outputs and phoneme_batch to 2D
            seq_outputs = seq_outputs.reshape(-1, num_classes)  # (batch_size * target_seq_length, num_classes)
            phoneme_batch = phoneme_batch.reshape(-1)           # (batch_size * target_seq_length,)

            # Compute the loss
            loss = criterion(seq_outputs, phoneme_batch)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        print(f"Subject {subject}, Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")

        # Optional: Evaluate on validation set and save the best model
        # (Add your validation code here if needed)

        # After training, evaluate the model
        evaluate_model_with_mds(
            encoder_model,
            seq_model,
            test_loader,
            updated_phoneme_vocab,
            PHONEME_GROUPS,
            PHONEME_TO_GROUP,
            DEVICE
        )
