"""
Python file to extract features from the trained model for evaluation.
"""
import os
import pickle
from glob import glob
from operator import itemgetter
from typing import Optional, Union

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig
from tqdm import tqdm

from utils.instantiate_model import instantiate_model
from utils.synonyms import fix_synonyms_dict, synonym_combine


def load_model(cfg: DictConfig):
    """
    Load model from checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.swin:
        # need to load swin model
        chkpt = torch.load(
            cfg.checkpoint,
            map_location=device,
        )["state_dict"]
        W = chkpt["module.generator.weight"]
        b = chkpt["module.generator.bias"]
        model = {"W": W, "b": b}
    else:
        model = instantiate_model(cfg)
        model.to(device)
        model.eval()
        # load checkpoint
        checkpoint = torch.load(cfg.checkpoint, map_location=device)
        # remove module. prefix
        model_state_dict = checkpoint["model_state_dict"]
        model.state_dict = {
            k.replace(
                "module.", ""
            ).replace(
                "self_attn", "self_attention"
            ): v for k, v in model_state_dict.items()
        }
        # !!! Be careful with strict=False
        # some weights might not be loaded
        # and the model might not work as expected
        # this will happen silently
        model.load_state_dict(model.state_dict, strict=False)
        print(f"Loaded checkpoint {cfg.checkpoint}")
    return model, device


def load_text_files(
    cfg: DictConfig,
    model: Optional[Union[nn.Module, dict]] = None,
    device: Optional[torch.device] = None,
):
    """
    Load word embeddings, vocabulary and synonyms.
    """
    word_embds = None
    if model is not None:
        # load word embeddings
        word_embds = pickle.load(open(cfg.paths.word_embds_pkl, "rb"))
        if isinstance(word_embds, dict):
            # need to convert to list
            word_embds = [val for _, val in word_embds.items()]
        word_embds = torch.stack(word_embds).to(device)
        if not cfg.swin:
            word_embds = model.project_word_embeddings(word_embds)

    # load vocab
    vocab = pickle.load(open(cfg.paths.vocab_pkl, "rb"))
    if "words_to_id" in vocab.keys():
        vocab = vocab["words_to_id"]
    id2word = {v: k for k, v in vocab.items()}
    # load synonyms
    synonyms = pickle.load(open(cfg.paths.synonyms_pkl, "rb"))
    synonyms = fix_synonyms_dict(synonyms)
    return word_embds, vocab, id2word, synonyms


def create_dirs(
    out_dir: str,
):
    """
    Create directories for saving features.
    """
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    feats_save_dir_path = os.path.join(out_dir, "features")
    if not os.path.isdir(feats_save_dir_path):
        os.makedirs(feats_save_dir_path)
    classif_save_dir_path = os.path.join(out_dir, "classification")
    if not os.path.isdir(classif_save_dir_path):
        os.makedirs(classif_save_dir_path)
    nn_save_dir_path = os.path.join(out_dir, "nn")
    if not os.path.isdir(nn_save_dir_path):
        os.makedirs(nn_save_dir_path)
    return feats_save_dir_path, classif_save_dir_path, nn_save_dir_path


def update_syn_combine(
    logits: torch.Tensor,
    pred_dict: dict,
    dict_key: str,
    synonyms: dict,
    id2word: dict,
    vocab: dict,
    synonym_grouping: bool,
) -> dict:
    """
    Update prediction dictionaries with synonym combine if needed.

    Args:
        logits (torch.Tensor): logits tensor
        pred_dict (dict): prediction dict to update
        dict_key (str): key to update
        synonyms (dict): synonym dict
        id2word (dict): id2word dict
        vocab (dict): vocab dict
        synonym_grouping (bool): whether to use synonym grouping or not

    Returns:
        pred_dict (dict): updated prediction dict
    """
    top5_probs, top5_labels = torch.topk(
        logits, k=5, dim=-1
    )
    top5_probs = top5_probs.cpu().numpy()
    top5_labels = top5_labels.cpu().numpy()
    if synonym_grouping:
        labels = rearrange(top5_labels, "t k -> (t k)")
        words = itemgetter(*labels)(id2word)
        words = rearrange(
            np.array(words), "(t k) -> t k", k=5
        )
        new_words, new_probs = [], []
        for word, prob in zip(words, top5_probs):
            new_prob, new_word = synonym_combine(
                word, prob, synonyms,
            )
            new_words.append(new_word)
            new_probs.append(new_prob)
        new_words = np.array(new_words)
        new_words = rearrange(new_words, "t k -> (t k)")
        labels = itemgetter(*new_words)(vocab)
        labels = rearrange(
            np.array(labels), "(t k) -> t k", k=5
        )
        top5_labels = labels
        top5_probs = np.array(new_probs)

    # update pred_dict
    pred_dict[dict_key] = {
        "labels": [top5_labels],
        "probs": [top5_probs],
        "logits": [],
    }
    return pred_dict


def save_dicts(
    features: dict,
    feats_save_dir_path: str,
    classification: dict,
    classif_save_dir_path: str,
    nn_classification: dict,
    nn_save_dir_path: str,
    vid_name: str,
) -> None:
    """
    Save predictions that are stored in dictionaries.

    Args:
        features (dict): features dict
        feats_save_dir_path (str): features save dir path
        classification (dict): classification dict
        classif_save_dir_path (str): classification save dir path
        nn_classification (dict): nn classification dict
        nn_save_dir_path (str): nn classification save dir path
        vid_name (str): video name
    """
    vid_name = f"{vid_name.split('.')[0]}.pkl"
    with open(os.path.join(feats_save_dir_path, vid_name), "wb") as feats_f:
        pickle.dump(features, feats_f)
    with open(os.path.join(classif_save_dir_path, vid_name), "wb") as classif_f:
        pickle.dump(classification, classif_f)
    with open(os.path.join(nn_save_dir_path, vid_name), "wb") as nn_f:
        pickle.dump(nn_classification, nn_f)


@hydra.main(version_base=None, config_path="config", config_name="cslr2_eval.yaml")
def main(cfg: Optional[DictConfig] = None) -> None:
    """
    Main Funcion to extract features from trained model for evaluation.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if cfg.swin:
        out_dir = cfg.checkpoint
    else:
        out_dir = os.path.dirname(
            os.path.dirname(cfg.checkpoint)
        )

    # load model
    model, device = load_model(cfg)
    # load other files
    word_embds, vocab, id2word, synonyms = load_text_files(
        cfg, model, device,
    )

    # extract features (first loop on splits and then on csv roots)
    for split in ["train", "val", "test"]:
        # load dataset
        setname = split if split != "test" else "public_test"
        dataset = hydra.utils.instantiate(
            cfg.dataset,
            setname=setname,
            subtitles_max_duration=1000000.0,
            subtitles_min_duration=0.0,
        )
        for csv_root in [cfg.paths.misaligned_csv_root, cfg.paths.heuristic_aligned_csv_root]:
            # 0 is the tolerance
            split_csv_root = os.path.join(csv_root, f"0/{split}")
            all_csvs = list(glob(os.path.join(split_csv_root, "*.csv")))
            print(f"Found {len(all_csvs)} csvs in {split_csv_root}")
            # create the save directory
            save_dir_path = os.path.join(
                out_dir,
                os.path.join(csv_root.split("/")[-2], "eval")
            )
            feats_save_dir_path, classif_save_dir_path, nn_save_dir_path = create_dirs(
                save_dir_path
            )
            for csv in tqdm(all_csvs):
                gt_df = pd.read_csv(csv, delimiter=",")
                starts = gt_df["start_sub"].tolist()
                ends = gt_df["end_sub"].tolist()
                subs = gt_df["english sentence"].tolist()
                vid_name = os.path.basename(csv)
                features = {}
                classification = {}
                nn_classification = {}
                assert len(starts) == len(ends) and len(starts) == len(subs)
                for start, end, sub in zip(starts, ends, subs):
                    start, end = float(start), float(end)
                    start = max(0.0, start)
                    end = min(
                        end,
                        dataset.subtitles.length[
                            dataset.subtitles.info_file_idx[
                                vid_name.replace(".csv", ".mp4")
                            ]
                        ] / 25 - 0.32,
                    )
                    try:
                        src = dataset.features.load_sequence(
                            episode_name=vid_name,
                            begin_frame=int(start * 25),
                            end_frame=int(end * 25),
                        ).to(device).unsqueeze(0)
                        if cfg.swin:
                            # swin model
                            with torch.no_grad():
                                src = src.squeeze(0)
                                logits = src @ model["W"].T + \
                                    model["b"][None, :]
                                logits = torch.nn.Softmax(dim=-1)(logits)
                            dict_key = f"{round(start, 3)}--{round(end, 3)}"
                            classification = update_syn_combine(
                                logits, classification, dict_key, synonyms,
                                id2word, vocab, cfg.synonym_grouping,
                            )

                        else:
                            with torch.no_grad():
                                cls_tokens, output_tensor = model.video_encoder(
                                    src)
                                tokens = cls_tokens[:, 1:] if not model.no_video_encoder \
                                    else cls_tokens
                                feats = tokens.cpu().numpy()
                                if model.video_token_ll is not None:
                                    video_tokens = model.project_token_embeddings(
                                        tokens
                                    )
                                else:
                                    # normalise
                                    video_tokens = F.normalize(tokens, dim=-1)
                                video_tokens = video_tokens.squeeze(0)
                                # compute sim matrix between video tokens and word embeddings
                                sim_matrix = video_tokens @ word_embds.T
                                if cfg.synonym_grouping:
                                    sim_matrix = torch.nn.Softmax(
                                        dim=-1
                                    )(sim_matrix / cfg.temp)
                            dict_key = f"{round(start, 3)}--{round(end, 3)}"
                            nn_classification = update_syn_combine(
                                sim_matrix, nn_classification, dict_key, synonyms,
                                id2word, vocab, cfg.synonym_grouping,
                            )
                            features[dict_key] = feats

                            # classification layer (from logits)
                            output_tensor = output_tensor.squeeze(0)
                            classification = update_syn_combine(
                                output_tensor, classification, dict_key, synonyms,
                                id2word, vocab, cfg.synonym_grouping,
                            )
                    except AttributeError:
                        print(f"Some error with {vid_name} at {start}--{end}")
                        print(f"Sub: {sub}")
                save_dicts(
                    features, feats_save_dir_path,
                    classification, classif_save_dir_path,
                    nn_classification, nn_save_dir_path,
                    vid_name,
                )


if __name__ == "__main__":
    main()
