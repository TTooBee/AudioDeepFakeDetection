import torch
from torch.utils.data import Dataset
import numpy as np
import os

# preprocess.py에서 load_features 함수를 import 합니다.
from preprocess import load_features

class AudioFeaturesDataset(Dataset):
    def __init__(self, base_folder_real=None, base_folder_fake=None):
        # 기본 경로 설정
        if base_folder_real is None:
            base_folder_real = 'features_real_temp'
        if base_folder_fake is None:
            base_folder_fake = 'features_fake_temp'

        # 실제 데이터를 로드하고 레이블을 생성합니다.
        self.features_real = load_features(base_folder_real)
        self.labels_real = np.ones(len(self.features_real))

        # 가짜 데이터를 로드하고 레이블을 생성합니다.
        self.features_fake = load_features(base_folder_fake)
        self.labels_fake = np.zeros(len(self.features_fake))

        # 데이터와 레이블을 하나의 리스트로 통합합니다.
        self.data = np.concatenate((self.features_real, self.features_fake), axis=0)
        self.labels = np.concatenate((self.labels_real, self.labels_fake), axis=0)

    def __len__(self):
        # 데이터셋의 총 데이터 수를 반환합니다.
        return len(self.data)

    def __getitem__(self, idx):
        # 지정된 인덱스의 데이터와 레이블을 반환합니다.
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


if __name__ == "__main__":
    # 사용 예
    dataset = AudioFeaturesDataset('features_real_temp', 'features_fake_temp')
    print(len(dataset))
    feature, label = dataset[0]
    print(feature.shape, label)