import pickle
import os
import json
import numpy as np
import traceback
import contextlib
from io import StringIO
from datetime import datetime
from QCCL.transformations import NoMatchingSubgraphsError, CompositeTransformation, TransformationError
import sys

# Define your available transformations
transformation_pool = [
                "add_identity", 
                "remove_identity", 
                "swap_ctrl_trgt", 
                "cnot_decomp", 
                "change_basis", 
                "parallel_x", 
                "parallel_z", 
                "commute_cnot_rot", 
                "commute_cnots", 
                "swap_cnots"
                ]

def generate_augmented_dataset(input_file, transformations=None, output_file=None, save_interval=1, metadata_file=None):
    """Main function to generate augmented dataset."""
    sys.path.append("../../Data")
    
    if transformations is None:
        transformations = transformation_pool

    # Load dataset
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    n_qubits = list(data["generation_progress"].keys())[0]
    dataset_size = data["generation_progress"][n_qubits]
    dataset = data["dataset"]
    statevectors = data["statevectors"]

    # Save statevectors and free memory
    statevectors_dir = save_statevectors(statevectors, n_qubits)
    del statevectors

    # Load or initialize metadata
    metadata_file = metadata_file or f"metadata_{n_qubits}_qubits.json"
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as metadata_f:
            metadata = json.load(metadata_f)
        metadata = json.loads(metadata)
    else:
        metadata = {
            "dataset_info": {
                "num_qubits": n_qubits,
                "num_samples": 0,
                "num_views": 0,
                "timestamp": datetime.now().isoformat(),
                "statevector_dir": statevectors_dir
            },
            "logs": {
                "processed_count": 0,
                "samples": {}
            },
            "configuration": {
                "transformations": transformations,
                "save_interval": save_interval
            }
        }

    # Load or initialize augmented dataset
    output_file = output_file or f"augmented_dataset_{n_qubits}_qubits.pkl"
    if os.path.exists(output_file):
        with open(output_file, "rb") as f:
            augmented_dataset = pickle.load(f)
    else:
        augmented_dataset = []

    # Process samples
    for idx, sample in enumerate(dataset):
        print(f"Processing sample {idx}/{dataset_size}...")
        if str(idx) in metadata["logs"]["samples"]:
            print(f"Sample {idx} already processed. Skipping...")
            continue

        # Initialize sample log and views
        metadata["logs"]["samples"][f"{idx}"] = {"successfully_applied": [], "successful_count": 0}
        sample_views = {}

        # Save base sample
        sample_views[0] = {
            "graph": sample.graph,
            "quantum_circuit": sample.quantum_circuit,
            "transformations": []
        }
        metadata["dataset_info"]["num_views"] += 1

        # Apply transformations
        successful_transformations = augment(sample, transformations, sample_views)

        # Save statevector and views to the augmented dataset, and free memory
        statevector_path = os.path.join(statevectors_dir, f"statevector_{idx}.npy")
        augmented_dataset.append({
            "sample_id": idx,  # Add a unique identifier for each sample
            "statevector": np.load(statevector_path).tolist(),
            "views": sample_views
        })
        # Delete the statevector file to free up memory
        os.remove(statevector_path)

        # Update logs in metadata
        metadata["logs"]["samples"][f"{idx}"]["successfully_applied"] = successful_transformations
        metadata["logs"]["samples"][f"{idx}"]["successful_count"] += len(successful_transformations)
        metadata["logs"]["processed_count"] += 1
        metadata["dataset_info"]["num_samples"] += 1
        metadata["dataset_info"]["num_views"] += len(successful_transformations)

        print(metadata)
        # Save progress periodically
        if idx % save_interval == 0:
            save_progress(output_file, augmented_dataset, metadata_file, metadata)

    # Final save
    save_progress(output_file, augmented_dataset, metadata_file, metadata)
    print("Augmented dataset generation completed.")

    # Cleanup
    if os.path.exists(statevectors_dir) and not os.listdir(statevectors_dir):
        os.rmdir(statevectors_dir)

    return augmented_dataset, metadata
    


def save_statevectors(statevectors, n_qubits):
    """Save statevectors as a compressed .npz file and separate .npy files."""
    # Save all statevectors as a single compressed .npz file
    filename = f"statevectors_{n_qubits}_qubits.npz"
    np.savez_compressed(filename, *statevectors)
    print(f"Statevectors saved to {filename} (compressed)")

    # Save individual statevectors as .npy files in a directory
    directory = f"statevectors_{n_qubits}_qubits"
    os.makedirs(directory, exist_ok=True)
    for idx, sv in enumerate(statevectors):
        np.save(os.path.join(directory, f"statevector_{idx}.npy"), sv)
    print(f"Statevectors saved separately to {directory}")

    return directory


def augment(sample, transformations, dictionary):
    """Apply transformations to a sample and log results."""
    successful_transformations = {}
    view_idx = 1

    for t_1 in transformations:
        try:
            with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
                transformed = CompositeTransformation(sample, [t_1]).apply()
            dictionary[view_idx] = {
                "graph": transformed.graph,
                "circuit": transformed.quantum_circuit,
                "transformations": [t_1]
            }
            successful_transformations[f"{view_idx}"] = t_1
            view_idx += 1
        except NoMatchingSubgraphsError:
            print(f"No matching subgraphs found for {t_1}: moving on to the next transformation.")
            continue
        except TransformationError as e:
            print(f"Transformation error for {t_1}.")
            print(f"Exception details: {str(e)}")
            continue

        for t_2 in transformations:
            try:
                with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
                    transformed = CompositeTransformation(sample, [t_1, t_2]).apply()
                dictionary[view_idx] = {
                    "graph": transformed.graph,
                    "circuit": transformed.quantum_circuit,
                    "transformations": [t_1, t_2]
                }
                successful_transformations[f"{view_idx}"] = f"{t_1}, {t_2}"
                view_idx += 1
            except NoMatchingSubgraphsError:
                print(f"No matching subgraphs found for {t_1}, {t_2}: moving on to the next transformation.")
                continue
            except TransformationError as e:
                print(f"Transformation error for {t_1}, {t_2}.")
                print(f"Exception details: {str(e)}")
                continue

    return successful_transformations


def save_progress(file_path, data, metadata_path, metadata):
    """Save the augmented dataset and progress log."""

    # Use a temporary file for safer incremental saving
    temp_file_path = file_path.replace('.pkl', '.tmp')

    try:
        save_to_file(file_path, data, '.pkl', temp_file_path)
        save_to_file(metadata_path, metadata, '.json')

        print(f"Both {file_path} and {metadata_path} saved successfully.")

        return True

    except Exception as e:
        # Cleanup in case of failure
        print("Error: Failed to save files atomically.")
        print(f"Exception details: {str(e)}")
        print(traceback.format_exc())

        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)

        return False

def save_to_file(file_path, data, format=None, temp_file_path=None):
    """Save data to file based on the format specified."""
    print(f"Saving data to {file_path}...")
    try:
        if format == '.json':
            json_str = json.dumps(data, indent=4)
            with open(file_path, 'w') as file:
                json.dump(json_str, file, indent=4)
        elif format == '.pkl':
            # Save to a temporary file first to ensure atomicity
            temp_file_path = temp_file_path or file_path.replace('.pkl', '.tmp')
            with open(temp_file_path, 'wb') as file:
                pickle.dump(data, file)
            
            # Rename temporary file to final destination
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_path = f"{timestamp}_{file_path}"
            if os.path.exists(file_path): # Backup existing file
                os.rename(file_path, unique_path)
            os.rename(temp_file_path, file_path) # Rename temporary file
            if os.path.exists(unique_path): # Remove backup
                os.remove(unique_path)
        else:
            with open(file_path, 'w') as file:
                file.write(str(data))  # Fall back to a text-based save
        return True  # Indicate successful save
    except Exception as e:
        print(f"Error saving dataset: {str(e)}")
        return False  # Indicate failed save

