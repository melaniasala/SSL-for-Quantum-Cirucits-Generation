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
from tqdm import tqdm

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

def generate_augmented_dataset(input_file, transformations=None, save_interval=1, output_dir=None, chunk_size=None, start_idx=0, end_idx=None, find_all=True):
    """Main function to generate augmented dataset."""
    sys.path.append('../../Data')

    if transformations is None:
        transformations = transformation_pool

    # Load dataset
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    n_qubits = list(data["generation_progress"].keys())[0]
    dataset_size = data["generation_progress"][n_qubits]
    statevectors = data["statevectors"]

    # Load dataset in chunks if specified
    if chunk_size:
        dataset = data["dataset"][start_idx : start_idx + chunk_size].copy()
    else:
        dataset = data["dataset"].copy()
    del data

    # Create output directory if it doesn't exist
    if not output_dir:
        output_dir = f"augmented_dataset_{n_qubits}_qubits"
    os.makedirs(output_dir, exist_ok=True)

    # Save statevectors and free memory
    statevectors_dir = save_statevectors(statevectors, n_qubits, output_dir, start_idx)
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
                "transformations": transformations,
                "timestamp": datetime.now().isoformat(),
            },
            "logs": {
                "processed_count": 0,
                "processed_samples": {}
            },
            "samples": {}
        }

    # Process samples
    end_idx = end_idx or dataset_size
    for idx in range(start_idx, end_idx):
        # Load next chunk if needed
        shifted_idx = idx - start_idx
        if chunk_size and shifted_idx > 0 and shifted_idx%chunk_size == 0:
            chunk_start = idx
            chunk_end = min(idx + chunk_size, dataset_size)
            print("-" * 80)
            print(f"Loading chunk {idx//chunk_size + 1}/{dataset_size//chunk_size} (from sample {chunk_start+1} to {chunk_end+1})...")
            with open(input_file, 'rb') as f:
                data = pickle.load(f)
            dataset = data["dataset"][chunk_start:chunk_end].copy()
            del data
        sample = dataset[shifted_idx%chunk_size] if chunk_size else dataset[idx]

        sample_id = f"q{n_qubits}_s{idx:02}"
        print("-" * 80)
        print(f"Processing sample {idx+1}/{dataset_size} (sample_id: {sample_id})...")
        if sample_id in metadata["samples"]:
            print(f"Sample {sample_id} already processed. Skipping...")
            # Delete statevector file if it exists
            if os.path.exists(os.path.join(statevectors_dir, f"statevector_{idx}.npz")):
                os.remove(os.path.join(statevectors_dir, f"statevector_{idx}.npz"))
            continue
        else:
            print()

         # Create folder for the sample
        sample_dir = os.path.join(output_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        # Initialize sample logs
        metadata["logs"]["processed_samples"][f"{idx}"] = {"successfully_applied": [], "successful_count": 0}
        # Initialize sample metadata
        metadata["samples"][sample_id] = {
            "statevector_file": os.path.join(sample_dir, f"{sample_id}_statevector.npz"),
            "views": {}
        }

        # Save statevector 
        statevector_path = os.path.join(statevectors_dir, f"statevector_{idx}.npz")
        with np.load(statevector_path, allow_pickle=True) as sv:
            sample_statevector = sv
        statevector_file = os.path.join(sample_dir, f"{sample_id}_statevector.npz")
        np.savez_compressed(statevector_file, sample_statevector)
        metadata["samples"][sample_id]["statevector_file"] = statevector_file

        # Free memory
        del sample_statevector
        os.remove(statevector_path)

        # Save base sample
        view_idx = 0
        view_id = f"{sample_id}_v{view_idx:03}"
        base_view_filename = f"{view_id}.pkl"
        base_view_path = os.path.join(sample_dir, base_view_filename)
        with open(base_view_path, "wb") as f:
            pickle.dump({
                "graph": sample.graph,
                "quantum_circuit": sample.quantum_circuit,
                "transformations": []
            }, f)
        metadata["samples"][sample_id]["views"][view_id] = {
            "file": base_view_path,
            "transformations": []
        }
        metadata["dataset_info"]["num_views"] += 1

        # Apply transformations
        successful_transformations = augment(sample, transformations, sample_id, sample_dir, metadata, find_all)

        # Update logs in metadata
        metadata["logs"]["processed_samples"][f"{idx}"]["successfully_applied"] = successful_transformations
        metadata["logs"]["processed_samples"][f"{idx}"]["successful_count"] += len(successful_transformations)
        metadata["logs"]["processed_count"] += 1
        metadata["dataset_info"]["num_samples"] += 1

        # Save progress periodically
        if idx % save_interval == 0:
            save_metadata(metadata_path, metadata)

    print("Augmented dataset generation completed.")

    # Cleanup
    if os.path.exists(statevectors_dir) and not os.listdir(statevectors_dir):
        os.rmdir(statevectors_dir)

    return metadata
    


def save_statevectors(statevectors, n_qubits, directory, start):
    """Save statevectors as a compressed .npz file and separate .npy files."""
    # Save individual statevectors as .npz files in a directory
    directory = os.path.join(directory, f"statevectors_{n_qubits}_qubits") if directory else f"statevectors_{n_qubits}_qubits"
    os.makedirs(directory, exist_ok=True)
    if isinstance(statevectors, list):
        pass
    elif isinstance(statevectors, dict):
        statevectors = list(statevectors.values())[0]
    else:
        raise TypeError("The 'statevectors' object must be either a list or a dictionary.")

    for idx, sv in enumerate(statevectors):
        if idx < start:
            continue
        compressed_statevector = np.array(sv, dtype=np.float16)
        np.savez_compressed(os.path.join(directory, f"statevector_{idx}.npz"), compressed_statevector)
      
    print(f"Statevectors saved separately to directory {directory}\n")

    # Free up memory
    del statevectors, compressed_statevector, sv

    return directory


def augment(sample, transformations, sample_id, sample_dir, metadata, find_all=True):
    """Apply transformations to a sample and log results."""
    successful_transformations = {}
    view_idx = 1
    unsuc_log = str()

    for t_1 in tqdm(transformations, desc="Producing augmented views"):
        try:
            with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
                transformed = CompositeTransformation(sample, [t_1], find_all=find_all).apply()
            view_id = f"{sample_id}_v{view_idx:03}"
            view_filename = f"{view_id}.pkl"
            view_path = os.path.join(sample_dir, view_filename)
            with open(view_path, "wb") as f:
                pickle.dump({
                    "graph": transformed.graph,
                    "quantum_circuit": transformed.quantum_circuit,
                    "transformations": [t_1]
                }, f)
            metadata["samples"][sample_id]["views"][view_id] = {
                "file": view_path,
                "transformations": [t_1]
            }
            metadata["dataset_info"]["num_views"] += 1
            successful_transformations[f"{view_id}"] = t_1
            view_idx += 1
        except NoMatchingSubgraphsError:
            unsuc_log += f"No matching subgraphs found for {t_1}: moving on to the next transformation.\n"
            continue
        except TransformationError as e:
            unsuc_log += f"Transformation error for {t_1}. Exception details: {str(e)}\n"
            continue

        for t_2 in transformations:
            try:
                with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
                    transformed = CompositeTransformation(sample, [t_1, t_2], find_all=find_all).apply()
                view_id = f"{sample_id}_v{view_idx:03}"
                view_filename = f"{view_id}.pkl"
                view_path = os.path.join(sample_dir, view_filename)
                with open(view_path, "wb") as f:
                    pickle.dump({
                        "graph": transformed.graph,
                        "quantum_circuit": transformed.quantum_circuit,
                        "transformations": [t_1, t_2]
                    }, f)
                metadata["samples"][sample_id]["views"][view_id] = {
                    "file": view_path,
                    "transformations": [t_1, t_2]
                }
                metadata["dataset_info"]["num_views"] += 1
                successful_transformations[f"{view_id}"] = [t_1, t_2]
                view_idx += 1
            except NoMatchingSubgraphsError:
                unsuc_log += f"No matching subgraphs found for {t_1}, {t_2}: moving on to the next transformation.\n"
                continue
            except TransformationError as e:
                unsuc_log += f"Transformation error for {t_1}, {t_2}. Exception details: {str(e)}\n"
                continue

    if unsuc_log:
        print(unsuc_log) 

    return successful_transformations


def save_metadata(metadata_path, metadata):
    """Save metadata to a JSON file."""

    try:
        print("\nSaving progress...")
        save_to_file(metadata_path, metadata, '.json')

        print(f"Metadata saved to {metadata_path} successfully.\n")

        return True

    except Exception as e:
        print(f"Error: Failed to save file. Exception details: {str(e)}")
        print(traceback.format_exc())
        # Stop execution and raise the exception
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
    
