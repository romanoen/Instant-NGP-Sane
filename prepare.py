import os
import torch
import numpy as np
from tqdm import tqdm

# =============================
# Configuration Parameters
# =============================

CONFIG = {
    'hash_encoding': {
        'num_levels': 16,             # Number of levels in the hash encoding
        'level_dim': 2,               # Feature dimension at each level
        'input_dim': 3,               # Dimensionality of the input coordinates (e.g., x, y, z)
        'log2_hashmap_size': 19,      # Log2 of the hash map size (2^19 entries)
        'base_resolution': 16         # Resolution of the first (lowest) level
    }
}

# =============================
# Utility Functions
# =============================

def load_torch_weights(file_path):
    """
    Load model weights from a checkpoint (.pth) file.

    Args:
        file_path (str): Path to the .pth checkpoint file.

    Returns:
        dict or None: Returns the 'model' state dictionary if successful,
                      otherwise prints an error and returns None.
    """
    try:
        weights = torch.load(file_path, map_location='cpu')
        # Assuming the checkpoint stores the model under the key 'model'
        return weights['model']
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def extract_hash_encoding_structure(model_weights, config, levels_to_extract=None):
    """
    Extract per-level embeddings from the hash encoder in a NeRF model checkpoint.

    Args:
        model_weights (dict): State dictionary of the loaded model.
        config (dict): Configuration for the hash encoding (num_levels, level_dim, etc.).
        levels_to_extract (list[int], optional): Specific level indices to extract (0-based).
                                                 If None, extract all levels.

    Returns:
        dict: A dictionary mapping level names ('level_0', 'level_1', etc.) to their weight tensors.
    """
    # Access the flattened embeddings tensor from the state dict
    embeddings = model_weights['_orig_mod.grid_encoder.embeddings']

    num_levels = config['num_levels']
    level_dim = config['level_dim']
    input_dim = config['input_dim']
    log2_hashmap_size = config['log2_hashmap_size']
    base_resolution = config['base_resolution']

    # Calculate maximum number of parameters per level (hashed) based on log2 size
    max_params = 2 ** log2_hashmap_size
    # Compute the scale factor between levels, so that resolutions grow exponentially from base_resolution to 2048
    per_level_scale = np.exp2(np.log2(2048 / base_resolution) / (num_levels - 1))

    hash_structure = {}
    offset = 0

    # Iterate over each level index to slice out the corresponding weights
    for level in range(num_levels):
        # Compute the current resolution for this level
        resolution = int(np.ceil(base_resolution * (per_level_scale ** level)))
        # Number of parameters in this level is either resolution^input_dim or max_params, whichever is smaller
        params_in_level = min(max_params, resolution ** input_dim)
        # Round up to the nearest multiple of 8 for alignment
        params_in_level = int(np.ceil(params_in_level / 8) * 8)

        # Slice out the embeddings corresponding to this level
        level_weights = embeddings[offset:offset + params_in_level]

        # If the user did not specify specific levels, or this level is requested, store it
        if levels_to_extract is None or level in levels_to_extract:
            hash_structure[f'level_{level}'] = level_weights

        # Move the offset forward by the number of parameters read
        offset += params_in_level

    return hash_structure

def preprocess_to_tokens_and_positions(base_dict):
    """
    Flatten the per-level hash encoding dictionary into two tensors:
    - tokens: the embedding vectors themselves
    - positions: a coordinate tensor [global_index, level_index, position_within_level]

    Args:
        base_dict (dict): Dictionary with keys like 'level_0', 'level_1', ... and tensor values.

    Returns:
        (torch.Tensor, torch.Tensor):
            tokens: Tensor of shape [total_tokens, level_dim].
            positions: Tensor of shape [total_tokens, 3] with (global_index, level_index, position_in_level).
    """
    tokens = []
    positions = []
    global_index = 0

    # Sort keys by numeric level index to ensure consistent ordering
    for key in sorted(base_dict.keys(), key=lambda x: int(x.split("_")[1])):
        layer_index = int(key.split("_")[1])
        layer_tensor = base_dict[key]
        num_tokens = layer_tensor.shape[0]

        # For each embedding in this level, append its vector and create a position triplet
        for pos_in_layer in range(num_tokens):
            token = layer_tensor[pos_in_layer]
            tokens.append(token)
            # (global_index, which level, position within that level)
            positions.append(torch.tensor([global_index, layer_index, pos_in_layer]))
            global_index += 1

    # Stack all token vectors into a single tensor, and same for positions
    return torch.stack(tokens), torch.stack(positions)

def save_processed_model(output_dir, model_id, tokens, positions):
    """
    Save the preprocessed token and position tensors to disk.

    Args:
        output_dir (str): Directory where the output files will be saved.
        model_id (str): Identifier for the model (used as filename prefix).
        tokens (torch.Tensor): Tensor of shape [num_tokens, level_dim].
        positions (torch.Tensor): Tensor of shape [num_tokens, 3].
    """
    os.makedirs(output_dir, exist_ok=True)
    torch.save(tokens, os.path.join(output_dir, f"{model_id}_tokens.pt"))
    torch.save(positions, os.path.join(output_dir, f"{model_id}_positions.pt"))

# =============================
# Crawler: Batch Preprocessing
# =============================

def run_crawler(model_root_dir, output_dir, levels_to_extract=None):
    """
    Traverse a directory tree to find .pth model files, extract their hash encoding
    embeddings, and save them as token/position pairs for downstream processing.

    Args:
        model_root_dir (str): Root directory containing model .pth files.
        output_dir (str): Directory where processed token/position files will be saved.
        levels_to_extract (list[int] or None): Specific level indices to extract (0–15).
                                               If None, extract all levels.
    """
    model_paths = []

    # Walk through the directory tree to find all .pth files
    for root, dirs, files in os.walk(model_root_dir):
        for file in files:
            if file.endswith('.pth'):
                full_path = os.path.join(root, file)
                model_paths.append(full_path)

    print(f"Found {len(model_paths)} models. Starting preprocessing...")

    # Process each model checkpoint found
    for path in tqdm(model_paths):
        # Create a unique model ID by making the path relative to model_root_dir,
        # replacing separators with '__' and removing the '.pth' extension
        model_id = os.path.relpath(path, model_root_dir).replace(os.sep, '__').replace('.pth', '')

        # Load the checkpoint weights
        weights = load_torch_weights(path)
        if weights is None:
            continue  # Skip this model if loading failed

        # Extract the hash encoding structure (only selected levels if provided)
        base_dict = extract_hash_encoding_structure(weights, CONFIG['hash_encoding'], levels_to_extract)
        if not base_dict:
            continue  # Skip if nothing was extracted

        # Convert the per-level embeddings into flat token & position tensors
        tokens, positions = preprocess_to_tokens_and_positions(base_dict)
        # Save them to the output directory
        save_processed_model(output_dir, model_id, tokens, positions)

    print(f"Processing complete. Results saved to: {output_dir}")

# =============================
# Usage
# =============================

levels_to_extract = [0,1,2,3]

run_crawler(
    model_root_dir='shared_data',            # Directory containing .pth model files
    output_dir=f'prepared_objects_first_{max(levels_to_extract)+1}_levels/',  # Where processed data will be stored
    levels_to_extract=levels_to_extract 
)
