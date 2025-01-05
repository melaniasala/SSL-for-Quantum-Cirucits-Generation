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

def generate_augmented_dataset(input_file, transformations=None, save_interval=1, output_dir=None):
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

    # Create output directory if it doesn't exist
    if not output_dir:
        output_dir = f"augmented_dataset_{n_qubits}_qubits"
    os.makedirs(output_dir, exist_ok=True)

    # Save statevectors and free memory
    statevectors_dir = save_statevectors(statevectors, n_qubits, output_dir)
    del statevectors

    # Load or initialize metadata
    metadata_file = f"metadata_{n_qubits}_qubits.json"
    metadata_path = os.path.join(output_dir, metadata_file) if output_dir else metadata_file
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as metadata_f:
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
    data_file = f"augmented_dataset_{n_qubits}_qubits.pkl"
    data_path = os.path.join(output_dir, data_file) if output_dir else data_file

    # Process samples
    for idx, sample in enumerate(dataset):
        print("-" * 80)
        print(f"Processing sample {idx+1}/{dataset_size} (sample_id: {idx})...")
        if str(idx) in metadata["logs"]["samples"]:
            print(f"Sample {idx} already processed. Skipping...")
            continue
        else:
            print()

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
        statevector_path = os.path.join(statevectors_dir, f"statevector_{idx}.npz")
        with np.load(statevector_path, allow_pickle=True) as sv:
            sample_data = {
                "sample_id": idx,
                "statevector": sv,
                "views": sample_views
            }

        # Delete the statevector file to free up memory
        os.remove(statevector_path)

        # Append sample to file
        append_sample_to_file(data_path, sample_data)

        # Update logs in metadata
        metadata["logs"]["samples"][f"{idx}"]["successfully_applied"] = successful_transformations
        metadata["logs"]["samples"][f"{idx}"]["successful_count"] += len(successful_transformations)
        metadata["logs"]["processed_count"] += 1
        metadata["dataset_info"]["num_samples"] += 1
        metadata["dataset_info"]["num_views"] += len(successful_transformations)

        # Save progress periodically
        if idx % save_interval == 0:
            save_progress(data_path, sample_data, metadata_path, metadata)

    print("Augmented dataset generation completed.")

    # Cleanup
    if os.path.exists(statevectors_dir) and not os.listdir(statevectors_dir):
        os.rmdir(statevectors_dir)

    return metadata
    


def save_statevectors(statevectors, n_qubits, directory):
    """Save statevectors as a compressed .npz file and separate .npy files."""
    # Save all statevectors as a single compressed .npz file
    filename = f"statevectors_{n_qubits}_qubits.npz" 
    file_path = os.path.join(directory, filename) if directory else filename
    reduced_statevectors = [np.array(statevector, dtype=np.float16) for statevector in statevectors]
    np.savez_compressed(file_path, *reduced_statevectors)
    print(f"Statevectors saved to file {file_path} (compressed)")

    # Save individual statevectors as .npy files in a directory
    directory = os.path.join(directory, f"statevectors_{n_qubits}_qubits") if directory else f"statevectors_{n_qubits}_qubits"
    os.makedirs(directory, exist_ok=True)
    for idx, sv in enumerate(statevectors):
        compressed_statevector = np.array(sv, dtype=np.float16)
        np.savez_compressed(os.path.join(directory, f"statevector_{idx}.npz"), compressed_statevector)
    print(f"Statevectors saved separately to directory {directory}\n")

    # Free up memory
    del statevectors, compressed_statevector, reduced_statevectors, sv

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


def save_progress(file_path, data_to_append, metadata_path, metadata):
    """Save the augmented dataset and progress log."""

    # Use a temporary file for safer incremental saving
    temp_file_path = file_path.replace('.pkl', '.tmp')

    try:
        print("\nSaving progress...")
        append_sample_to_file(file_path, data_to_append)
        print(f"Saved {file_path} successfully.")
        save_to_file(metadata_path, metadata, '.json')

        print(f"Both {file_path} and {metadata_path} saved successfully.\n")

        return True

    except Exception as e:
        # Cleanup in case of failure
        print(f"Error: Failed to save files. Exception details: {str(e)}")
        print(traceback.format_exc())

        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        raise e

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
            unique_path = f"{file_path.replace('.pkl', '')}_{timestamp}.pkl"
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
        # Stop execution and raise the exception
        raise e
    

def append_sample_to_file(file_path, sample):
    """Append a single sample to the dataset file incrementally."""
    try:
        if not os.path.exists(file_path):
            # Create a new file if it doesn't exist
            with open(file_path, "wb") as f:
                pickle.dump([sample], f)
        else:
            # Append to the existing file
            with open(file_path, "rb") as f:
                existing_data = pickle.load(f)
            existing_data.append(sample)
            with open(file_path, "wb") as f:
                pickle.dump(existing_data, f)
    except Exception as e:
        print(f"Error while appending sample to file: {e}")
        raise

