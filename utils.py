from pypdf import PdfReader
import regex as re
import random
import torch
import pandas as pd
from tqdm import tqdm


def pdf_to_text(file_path: str) -> str:
    """
    Reads a PDF file and converts its content into text.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        str: The extracted text from the PDF.
    """
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"An error occurred while reading the PDF: {e}")
        return None
    
    
def clean_and_split_text(text: str) -> list[str]:
    """
    Cleans the input text and splits it into sentences by full stops.

    Args:
        text (str): The input text to clean and split.

    Returns:
        list: A list of sentences split by full stops.
    """
    try:
        # Remove extra whitespace and newlines
        cleaned_text = re.sub(r'\s+', ' ', text.strip())
        
        # Split the text by full stops
        sentences = [sentence.strip() for sentence in cleaned_text.split('.') if sentence.strip()]
        
        return sentences
    except Exception as e:
        print(f"An error occurred while cleaning and splitting the text: {e}")
        return None


def get_random_members(input_list: list, n: int, min_size: int = 50) -> list:
    """
    Selects n random members from the input list.

    Args:
        input_list (list): The list to select members from.
        n (int): The number of random members to return.

    Returns:
        list: A list containing n random members from the input list.
    """
    try:
        if n > len(input_list):
            raise ValueError("n cannot be greater than the length of the input list.")
        
        output_list = [i for i in input_list if len(i) > min_size]

        return random.sample(output_list, n)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def tokenize_instructions(tokenizer, instructions: list[str]) -> torch.Tensor:
    """
    Tokenizes the given instructions using the standard Hugging Face tokenizer.

    Args:
        tokenizer: The tokenizer object.
        instructions: The instructions to tokenize.

    Returns:
        torch.Tensor: Tokenized input IDs.
    """
    try:
        # Ensure the tokenizer has a pad token
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # Tokenize the instructions
        return tokenizer(
            instructions,
            padding=True,          # Add padding to ensure consistent sequence lengths
            truncation=True,       # Truncate sequences that exceed the model's max length
            return_tensors="pt",   # Return PyTorch tensors
        ).input_ids
    except Exception as e:
        print(f"Error during tokenization: {e}")
        return None
    

def analyse_hidden_layers(
        comment_df: pd.DataFrame,
        column: str,
        model,
        tokenizer,
        index: int = -1
        ) -> dict[int, dict]:
    """
    Analyzes hidden layers for each comment in the DataFrame.

    Args:
        comment_df (pd.DataFrame): DataFrame containing comments and tokens.
        model: Pretrained DialoGPT model.
        tokenizer: Tokenizer associated with the model.

    Returns:
        dict[int, dict]: Dictionary containing hidden layer embeddings for each comment.
    """
    output_dict = {}

    for i in tqdm(comment_df.index):
        tokens = tokenizer(comment_df.loc[i, column], return_tensors="pt")
        output_dict[i] = get_hidden_embeddings(tokens, model, index)

    return output_dict


def get_hidden_embeddings(tokens, model, index: int = -1) -> torch.Tensor:
    """
    Extracts the final hidden layer embeddings for the given tokens.

    Args:
        tokens: Tokenized input (PyTorch tensors).
        model: Pretrained DialoGPT model.

    Returns:
        torch.Tensor: Embeddings from the final hidden layer.
    """
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
        output_dict = outputs.hidden_states[index]  

    return output_dict


def analyse_hidden_layers(
        comment_df: pd.DataFrame,
        column: str,
        model,
        tokenizer,
        index: int = -1
        ) -> dict[int, torch.Tensor]:
    """
    Analyzes the final hidden layer for each comment in the DataFrame and aggregates it into a single vector.

    Args:
        comment_df (pd.DataFrame): DataFrame containing comments.
        column (str): Column name containing the comments.
        model: Pretrained DialoGPT model.
        tokenizer: Tokenizer associated with the model.
        index (int): Index of the hidden layer to extract (default is -1 for the final layer).

    Returns:
        dict[int, torch.Tensor]: Dictionary containing the aggregated embeddings for each comment.
    """
    output_dict = {}

    for i in tqdm(comment_df.index):
        # Tokenize the comment
        tokens = tokenizer(comment_df.loc[i, column], return_tensors="pt")

        # Get the final hidden layer
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
            final_hidden_layer = outputs.hidden_states[index]  # Shape: [1, sequence_length, hidden_size]

        # Aggregate the embeddings across the token dimension
        aggregated_embedding = final_hidden_layer.mean(dim=1)  # Mean pooling, Shape: [1, hidden_size]

        # Store the aggregated embedding
        output_dict[i] = aggregated_embedding.squeeze(0)  # Shape: [hidden_size]

    return output_dict

def get_hidden_embeddings(tokens, model, index: int = -1) -> torch.Tensor:
    """
    Extracts the final hidden layer embeddings for the given tokens and aggregates them into a single vector.

    Args:
        tokens: Tokenized input (PyTorch tensors).
        model: Pretrained DialoGPT model.
        index (int): Index of the hidden layer to extract (default is -1 for the final layer).

    Returns:
        torch.Tensor: Aggregated embeddings from the final hidden layer.
    """
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
        final_hidden_layer = outputs.hidden_states[index]  # Shape: [1, sequence_length, hidden_size]

    # Aggregate the embeddings across the token dimension
    aggregated_embedding = final_hidden_layer.mean(dim=1)  # Mean pooling, Shape: [1, hidden_size]

    return aggregated_embedding.squeeze(0)  # Shape: [hidden_size]


def get_comment_representation(hidden_layer):
    """
    Converts the hidden layer into a single vector representation for the comment.

    Args:
        hidden_layer (torch.Tensor): Hidden layer with shape [1, sequence_length, hidden_size].

    Returns:
        torch.Tensor: Pooled vector representation with shape [hidden_size].
    """
    # Mean pooling across the token dimension
    pooled_representation = hidden_layer.mean(dim=1)  # Shape: [1, 1024]
    return pooled_representation.squeeze(0)  # Shape: [1024]


def get_comment_representation_flat(hidden_layer):
    """
    Flattens the hidden layer into a single long vector representation for the comment.

    Args:
        hidden_layer (torch.Tensor): Hidden layer with shape [1, sequence_length, hidden_size].

    Returns:
        torch.Tensor: Flattened vector representation with shape [sequence_length * hidden_size].
    """
    flattened_representation = hidden_layer.view(-1)  # Shape: [sequence_length * hidden_size]
    return flattened_representation


def aggregate_tensors(tensor_list: list[torch.Tensor], method: str = "mean") -> torch.Tensor:
    """
    Aggregates a list of tensors into a single vector using the specified method.

    Args:
        tensor_list (list[torch.Tensor]): List of tensors with shape [1024].
        method (str): Aggregation method, either "mean" or "max" (default is "mean").

    Returns:
        torch.Tensor: Aggregated tensor with shape [1024].
    """
    if not tensor_list:
        raise ValueError("The tensor list is empty.")

    # Stack tensors into a single matrix
    stacked_tensors = torch.stack(tensor_list)  # Shape: [num_tensors, 1024]

    if method == "mean":
        # Mean pooling across the first dimension
        aggregated_tensor = stacked_tensors.mean(dim=0)  # Shape: [1024]
    elif method == "max":
        # Max pooling across the first dimension
        aggregated_tensor = stacked_tensors.max(dim=0).values  # Shape: [1024]
    else:
        raise ValueError("Invalid aggregation method. Use 'mean' or 'max'.")

    return aggregated_tensor