# ./src/cnn_spam_classifier.py

"""
PyTorch 기반 스팸 메일 분류 및 작성 스크립트 파일
SMS 문장을 숫자 시퀀스로 변환한 뒤,
Embedding + Conv1D + MaxPooling + Linear 구조로 정상/스팸을 분류 하는 모델 작성
"""

import os  # 파일 존재 여부
import random  # 난수 seed 고정

# 인터넷에 있는 csv 데이터 파일을 내려받기 위해 사용
import urllib.request  # 인터넷 접속을 코드상에서 해야 될 때

# 파이썬이 제공하는 표준 모듈
# 단어 빈도 수를 쉽게 계산하기 위해 사용
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# 난수 시드 고정 함수 정의
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    # CPU 사용시 PyTorch 난수 시드 고정
    torch.manual_seed(seed)

    # GPU 가 사용 가능한 경우 GPU 난수 시드 고정
    torch.cuda.manual_seed(seed)


# 간단한 영어 토큰화 수행하는 함수 정의
def simple_tokenize(text: str) -> list[str]:
    # 전달받은 문장을 문자열로 변환한 뒤에 소문자로 변환
    text = str(text).lower()

    # 문장을 공백 기준으로 나누어서 단어 리스트를 만듦
    tokens = text.split()

    return tokens


# 훈련 데이터에 대한 단어 사전 만드는 함수 정의
def build_vocab(texts: pd.Series, min_freq: int = 1) -> tuple[dict[str, int], Counter]:
    # 모든 단어의 등장 횟수를 저장할 Counter 객체 생성
    counter = Counter()

    # 훈련 데이터의 각 문장을 하나씩 반복 처리
    for text in texts:
        # 현재 문장을 토큰화한 뒤, Counter 에 단어 빈도를 누적
        counter.update(simple_tokenize(text))

    # 0번은 패딩 토큰으로 사용
    word_to_index = {"<PAD>": 0}

    # 1번은 사전에 없는 단어를 표현하는 미등록 단어 토큰으로 사용한다.
    word_to_index["<UNK>"] = 1

    # 빈도수가 높은 단어부터 정렬해서 단어 사전에 추가
    for word, freq in counter.most_common():
        # min_freq 이상 등장한 단어만 단어 사전에 포함시킨다.
        if freq >= min_freq:
            # 아직 사전에 없는 단어라면 새로운 번호를 부여
            if word not in word_to_index:
                word_to_index[word] = len(word_to_index)

    # 단어-번호 사전과 단어 빈도 Counter 를 반환
    return word_to_index, counter


# 하나의 문장을 정수 인덱스 시퀀스로 바꾸고, 고정 길이로 패딩하는 함수
def encode_end_pad(text: str, word_to_index: dict[str, int], max_len: int) -> list[int]:
    # 문장을 단어 단위로 나눔
    tokens = simple_tokenize(text)

    # 각 단어를 사전 번호를 변환하고, 사전에 없으면 <UNK> 번호 1을 사용
    encoded = [word_to_index.get(token, word_to_index["<UNK>"]) for token in tokens]

    # 문장이 max_len 보다 길면 앞에서 max_len 위치까지 자른다.
    encoded = encoded[:max_len]

    # 문장이 max_len 보다 짧으면 뒤쪽에 <PAD> 번호 0을 추가해서 길이를 맞춘다.
    padded = encoded + [word_to_index["<PAD>"]] * (max_len - len(encoded))

    # 길이가 맞춰진 단어 번호 정수 리스트가 반환
    return padded


# SMS 문장과 레이블(ham | spam, 정답)을 Torch DataLoader가 읽을 수 있도록 구성하는 Dataset 클래스 정의
class SMSDataset(Dataset):
    # 생성자가 해당 클래스 객체 생성시, 문장, 레이블, 단어 사전, 최대 길이를 전달받도록 함
    def __init__(self, texts: pd.Series, labels: pd.Series, word_to_index: dict[str, int], max_len: int):
        # 입력 문장을 리스트로 변환하고 저장
        self.texts = list(texts)

        # 정답 레이블도 정수 리스트로 변환하고 저장
        self.labels = list(labels.astype(int))

        # 단어를 정수 번호로 바꾸기(인코딩) 위한 사전 저장
        self.word_to_index = word_to_index

        # 모든 문장을 맞출 최대 길이로 저장
        self.max_len = max_len

    # Dataset의 전체 샘플 갯수 반환하는 메소드(멤버함수 : 클래스에 소속된 함수) 정의
    def __len__(self) -> int:  # 연산자 오버로딩
        # 저장된 문장의 갯수를 반환
        return len(self.texts)

    # 특정 인덱스 하나에 해당하는 입력 텐서와 정답 텐서를 반환하는 메소드 정의
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # index 위치의 문장을 정수 시퀀스로 변환하고 패딩 처리함
        x = encode_end_pad(self.texts[index], self.word_to_index, self.max_len)

        # index 위치의 정답 레이블도 꺼냄
        y = self.labels[index]

        # 입력 문장은 정수 시퀀스이므로, LongTensor 로 변환함
        # Embedding 계층은 정수 인덱스를 입력으로 받기 때문
        x_tensor = torch.tensor(x, dtype=torch.long)

        # 이진 분류 솔실함수 BCEWithLogitsLoss 에 맞추기 위해 정답 레이블 float 텐서로 변환
        # logits (점수) : ex) 0.0123 (0에 가까움), 0.978 (1에 가까움)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # 입력 텐서와 정답 텐서를 반환
        return x_tensor, y_tensor


# 모델 설계
# Embedding - Conv1D - GlobalMaxPooling - Linear 구조의 CNN 문장 분류 모델
class TextCNN(nn.Module):
    # 모델 계층 초기화
    def __init__(self, vocab_size: int,
                 embedding_dim: int = 32,
                 num_filters: int = 32,
                 kernel_size: int = 5,
                 dropout_ratio: float = 0.3):
        # 부모 클래스 초기화 생성자 호출 (후손 생성자 내부에서만 호출할 수 있음, 생성자 내부 첫줄에 기입할 것)
        super().__init__()

        # 신경망 계층에 대한 설정 선언
        # 1. Embedding 계층
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0  # 0인 이유는?
        )

        # 2. 과적합 줄이기 위해 Dropout 계층
        self.dropout1 = nn.Dropout(p=dropout_ratio)

        # 3. 1차원 합성곱 계층 : 문장 안의 연속된 단어 패턴을 추출
        self.conv1d = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
        )

        # ReLu 활성화함수 : 음수를 0으로 바꿈 => 양수 특징만 통과시킴
        self.relu = nn.ReLU()

        self.dropout2 = nn.Dropout(p=dropout_ratio)

        # 출력층
