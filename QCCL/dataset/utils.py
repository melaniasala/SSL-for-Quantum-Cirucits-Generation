import numpy as np

def split_dataset(X, train_size= 0.8, val_size=0.1, use_pre_paired=False, composite_transforms_size=7):
    print('\nStarting dataset split...')
    
    total_size = len(X)
    print(f"Total dataset size: {total_size}")
    print(f"Composite transformations size: {composite_transforms_size}")
    
    test_size = 1.0 - train_size - val_size
    test_size = int(test_size * total_size)
    val_size = int(val_size * total_size)
    train_size = total_size - val_size - test_size

    print(f"Splitting into {train_size} training, {val_size} validation, and {test_size} test circuits...")

    composite_circuits = X[-composite_transforms_size:]  # Last n circuits (composite transformations)
    single_transform_circuits = X[:-composite_transforms_size]  # Remaining circuits (single transformations)

    shuffling_mask = np.random.permutation(len(composite_circuits))
    composite_circuits = [composite_circuits[i] for i in shuffling_mask]

    val_data = composite_circuits[:val_size]
    remaining_composite = composite_circuits[min(val_size, composite_transforms_size):]

    test_data = remaining_composite[:min(test_size, len(remaining_composite))]
    if len(test_data) < test_size:  # If not enough composite circuits, take from single transformation circuits
        print("Not enough composite circuits for test set, adding single transformation circuits...")
        additional_test_data = single_transform_circuits[:(test_size - len(test_data))]
        test_data = test_data + additional_test_data

    remaining_composite = remaining_composite[min(test_size, len(remaining_composite)):]
    remaining_single = single_transform_circuits[(test_size - len(test_data)):]
    train_data = remaining_single + remaining_composite
    shuffling_mask = np.random.permutation(len(train_data))
    train_data = [train_data[i] for i in shuffling_mask]

    print('Data split completed:')
    print(f'Training set: {len(train_data)} samples, ({len(train_data)/total_size*100:.2f}%)')
    print(f'Validation set: {len(val_data)} samples, ({len(val_data)/total_size*100:.2f}%)')
    print(f'Test set: {len(test_data)} samples, ({len(test_data)/total_size*100:.2f}%)')
    
    return train_data, val_data, test_data