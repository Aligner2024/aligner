# Copyright Authors of Aligner. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# This file is modified from:
# https://github.com/PKU-Alignment/safe_rlhf/datasets/raw/safe_rlhf.py
# ==============================================================================
# Copyright 2023-2024 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Safe-RLHF preference datasets."""

from __future__ import annotations

from typing import ClassVar

from datasets import load_dataset
from safe_rlhf.datasets.base import RawDataset, RawSample
import json

__all__ = [
    'SafeRLHFDataset',
    'SafeRLHFTrainDataset',
    'SafeRLHFTestDataset',
    'SafeRLHF30KTrainDataset',
    'SafeRLHF30KTestDataset',
    'SafeRLHF10KTrainDataset',
    'TreeRLHFFULLTrainDataset',
    'TreeRLHFALLTrainDataset',
    'TreeRLHFPPODataset',
    'Baichuan3RM/train',
    'Baichuan3RM/test',
]


class SafeRLHFDataset(RawDataset):
    SPLIT: ClassVar[str]
    PATH: ClassVar[str]

    def __init__(self, path: str | None = None) -> None:
        self.data = json.load(open(path or self.PATH, 'r', encoding='utf-8'))

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(
            input=data.get('prompt'),
            answer=data.get('response_0'),
            other_answer=data.get('response_1'),
            better=int(data.get('better_response_id', 0)) == 0,
            safer=int(data.get('better_response_id', 0)) == 0,
            is_safe=True,
            is_other_safe=True,
        )

    def __len__(self) -> int:
        return len(self.data)

class TreeRLHFFULLTrainDataset(SafeRLHFDataset):
    NAME: str = 'TreeRLHFFULL/train'
    ALIASES: tuple[str, ...] = ('PKU-Alignment/TreeRLHFFULL/train',)
    PATH: str = '/root/data/tree/TreeBasedRLHF/datasets/training_data/alpaca/full/original_full_new.json'
    SPLIT: str = 'train'
    
class TreeRLHFALLTrainDataset(SafeRLHFDataset):
    NAME: str = 'TreeRLHFALL/train'
    ALIASES: tuple[str, ...] = ('PKU-Alignment/TreeRLHFALL/train',)
    PATH: str = '/root/data/tree/TreeBasedRLHF/datasets/training_data/alpaca/all/original_all_new.json'
    SPLIT: str = 'train'

class TreeRLHFTestDataset(SafeRLHFDataset):
    NAME: str = 'TreeRLHF/test'
    ALIASES: tuple[str, ...] = ('PKU-Alignment/TreeRLHFPPO/test',)
    PATH: str = '/root/data/tree/TreeBasedRLHF/datasets/training_data/alpaca/test_1000.json'
    SPLIT: str = 'test'
    
class SafeRLHFTrainDataset(SafeRLHFDataset):
    NAME: str = 'PKU-SafeRLHF/train'
    ALIASES: tuple[str, ...] = ('PKU-Alignment/PKU-SafeRLHF/train',)
    PATH: str = 'PKU-Alignment/PKU-SafeRLHF'
    SPLIT: str = 'train'

class SafeRLHFTestDataset(SafeRLHFDataset):
    NAME: str = 'PKU-SafeRLHF/test'
    ALIASES: tuple[str, ...] = ('PKU-Alignment/PKU-SafeRLHF/test',)
    PATH: str = 'PKU-Alignment/PKU-SafeRLHF'
    SPLIT: str = 'test'

class SafeRLHF30KTrainDataset(SafeRLHFDataset):
    NAME: str = 'PKU-SafeRLHF-30K/train'
    ALIASES: tuple[str, ...] = ('PKU-Alignment/PKU-SafeRLHF-30K/train',)
    PATH: str = 'PKU-Alignment/PKU-SafeRLHF-30K'
    SPLIT: str = 'train'


class SafeRLHF30KTestDataset(SafeRLHFDataset):
    NAME: str = 'PKU-SafeRLHF-30K/test'
    ALIASES: tuple[str, ...] = ('PKU-Alignment/PKU-SafeRLHF-30K/test',)
    PATH: str = 'PKU-Alignment/PKU-SafeRLHF-30K'
    SPLIT: str = 'test'


class SafeRLHF10KTrainDataset(SafeRLHFDataset):
    NAME: str = 'PKU-SafeRLHF-10K/train'
    ALIASES: tuple[str, ...] = ('PKU-Alignment/PKU-SafeRLHF-10K/train',)
    PATH: str = 'PKU-Alignment/PKU-SafeRLHF-10K'
    SPLIT: str = 'train'

class Baichuan3RMTrainDataset(SafeRLHFDataset):
    NAME: str = 'Baichuan3RM/train'
    PATH: str = '/root/tree/safe-rlhf/dataset/train.json'
    SPLIT: str = 'train'

class Baichuan3RMTestDataset(SafeRLHFDataset):
    NAME: str = 'Baichuan3RM/test'
    PATH: str = '/root/tree/safe-rlhf/dataset/test.json'
    SPLIT: str = 'test'