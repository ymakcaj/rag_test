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