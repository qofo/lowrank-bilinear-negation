"""CSV-backed MCQ dataset used to score the NegBench COCO / VOC2007 / CheXpert MCQ benchmarks.

Only ``CsvMCQDataset`` is kept from the upstream NegBench ``training/data.py``; the class body is
unchanged. The training loaders (webdataset, video, ImageNet) that lived alongside it are not
needed to reproduce the paper and are omitted together with their dependencies.
"""
import random

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CsvMCQDataset(Dataset):
    # Canonical caption order as stored in the CSV files.
    # caption_0 is always the correct answer; captions 1-3 are distractors.
    CAPTION_TYPES = ['gt', 'hybrid', 'positive', 'negative']
    SYNTHETIC_CAPTION_TYPES = ['positive', 'hybrid', 'hybrid', 'negative']

    def __init__(self, csv_file, transforms, num_answers=4, path="image_path",
                 tokenizer=None, shuffle_options=False, seed=42, is_synthetic=False):
        """
        Dataset for MCQ task evaluation.

        Args:
            csv_file (string): Path to the csv file with annotations.
            transforms (callable): Transform to be applied on a sample.
            num_answers (int): Number of answer choices (captions) given to the model.
            path (string): Column name for image file paths.
            tokenizer (callable): Tokenizer to be applied on the captions.
                Needs to be passed if dataset is used for training.
            shuffle_options (bool): If True, randomly permute the order of answer
                choices for each sample.  The correct_answer index and caption_types
                list are updated accordingly so downstream metrics remain correct.
                Essential for generative models (e.g. LLaVA) to eliminate position
                bias; has no effect on CLIP-like embedding models.
            seed (int): Random seed used for reproducible shuffling.
        """
        self.df = pd.read_csv(csv_file, sep=',')
        self.transforms = transforms
        
        # Dynamically detect num_answers from CSV columns if caption_* exist
        caption_cols = [c for c in self.df.columns if c.startswith("caption_")]
        if caption_cols:
            self.num_answers = len(caption_cols)
        else:
            self.num_answers = num_answers if num_answers is not None else 4

        # Dynamically check path column name
        if path in self.df.columns:
            self.path = path
        elif "filepath" in self.df.columns:
            self.path = "filepath"
        elif "image_path" in self.df.columns:
            self.path = "image_path"
        else:
            self.path = self.df.columns[0]

        self.tokenizer = tokenizer
        self.shuffle_options = shuffle_options
        self.is_synthetic = is_synthetic
        # Use a per-dataset Random instance so shuffling is independent of the
        # global numpy/torch RNG, and fully reproducible given the same seed.
        self._rng = random.Random(seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = str(row[self.path])
        captions = [str(row[f"caption_{i}"]) for i in range(self.num_answers)]
        correct_answer = int(row["correct_answer"]) if "correct_answer" in row else 0
        correct_answer_template = str(row["correct_answer_template"]) if "correct_answer_template" in row else "mcq"

        # caption_types tracks the semantic role of each caption slot.
        # Original order depends on whether the dataset is synthetic or not.
        if self.is_synthetic:
            caption_types = list(self.SYNTHETIC_CAPTION_TYPES[:self.num_answers])
        else:
            caption_types = list(self.CAPTION_TYPES[:self.num_answers])
        while len(caption_types) < self.num_answers:
            caption_types.append(f"option_{len(caption_types)}")

        if self.shuffle_options:
            perm = list(range(self.num_answers))
            self._rng.shuffle(perm)
            captions = [captions[i] for i in perm]
            caption_types = [caption_types[i] for i in perm]
            # correct_answer was at position 0 ('gt'); find its new position.
            correct_answer = perm.index(0)

        image = self.transforms(Image.open(image_path))

        if self.tokenizer is not None:
            captions = [self.tokenizer([str(caption)])[0] for caption in captions]
            captions = torch.stack(captions)  # (num_answers, max_seq_len)

        return (
            image,
            captions,
            correct_answer,
            correct_answer_template,
            image_path,
            caption_types,   # list[str] of length num_answers, e.g. ['hybrid','gt','negative','positive']
        )
