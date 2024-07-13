import torch
from . import data_preprocessing


def encode_sequence(graph, sequence_length=None, end_index=None, use_padding=True, padding_value=0.0):
    """
    Encode a sequence of nodes into a tensor representation: each node is represented by its feature vector.

    :param graph: QuantumCircuitGraph object
    :param sequence_length: Maximum size of the sequence to encode (default is the full sequence)
    :param end_index: Index of the last node in the sequence (exclusive)
    :param use_padding: Whether to pad the sequence to the specified length
    :param padding_value: Value to use for padding
    :return: Tensor representation of the sequence
    """
    # Ensure graph is a QuantumCircuitGraph object
    if not isinstance(graph, data_preprocessing.QuantumCircuitGraph):
        raise ValueError("graph parameter must be a QuantumCircuitGraph object")

    total_nodes = graph.node_feature_matrix.shape[0]
    if sequence_length is None:
        sequence_length = total_nodes

    if end_index is None:
        end_index = len(graph.node_ids)  # end node index (exclusive)
    start_index = max(end_index - sequence_length, 0)  # start node index (inclusive)
    actual_length = end_index - start_index

    # Initialize the encoded sequence with padding if needed
    encoded_sequence = []
    if use_padding and actual_length < sequence_length:
        padding_tensor = torch.ones((1, graph.n_node_features)) * padding_value
        encoded_sequence.extend([padding_tensor] * (sequence_length - actual_length))

    for node_id in graph.to_sequence()[start_index:end_index]:
        node_idx = graph.node_mapping[node_id]
        node_feature = graph.node_feature_matrix[node_idx]
        encoded_sequence.append(node_feature)

    # Convert the encoded sequence to a PyTorch tensor
    encoded_sequence = torch.stack(encoded_sequence) 
    return encoded_sequence