"""
Extraction of Sentence-Level Embeddings with SBERT models.
"""
from typing import List, Optional, Union

import torch
from sentence_transformers import SentenceTransformer
from torch.nn.utils.rnn import pad_sequence


def make_sentence_model(
    model_name: str = "all-mpnet-base-v2",
    root_path: Optional[str] = None,
) -> SentenceTransformer:
    """
    Setup SentenceTransformer model.

    Args:
        model_name (str): name of the model to use.
        root_path (Optional[str]): path to the root directory of the model.
    """
    if root_path is not None:
        model = SentenceTransformer(root_path + model_name)
    else:
        model = SentenceTransformer(model_name)
    return model


def extract_sentence_embeddings(
    model: SentenceTransformer,
    sentences: Union[str, List[str]],
    device: torch.device,
) -> torch.Tensor:
    """
    Extract sentence embeddings.

    Args:
        model (SentenceTransformer): SentenceTransformer model.
        sentences (Union[str, List[str]]): list of sentences to encode.
        device (torch.device): device to use for the model.

    Returns:
        torch.Tensor: tensor of sentence-level embeddings.
    """
    batch_size = 1
    if isinstance(sentences, List):
        batch_size = len(sentences)
    try:
        embeddings = model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
            device=device,
        )
    except AttributeError:
        embeddings = model.module.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
            device=device,
        )
    return embeddings


def extract_token_embeddings(
    model: SentenceTransformer,
    sentences: Union[str, List[str]],
    device: torch.device,
) -> torch.Tensor:
    """
    Extract embeddings at token level.

    Args:
        model (SentenceTransformer): SentenceTransformer model.
        sentences (Union[str, List[str]]): list of sentences to encode.
        device (torch.device): device to use for the model.

    Returns:
        torch.Tensor: tensor of token-level embeddings (padded).
    """
    batch_size = len(sentences) if isinstance(sentences, List) else 1
    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        output_value="token_embeddings",
        show_progress_bar=False,
        device=device,
    )
    return pad_sequence(embeddings, batch_first=True, padding_value=0)
