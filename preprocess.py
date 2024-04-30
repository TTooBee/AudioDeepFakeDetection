import numpy as np
import os

def load_and_pad_matrix(feature_path, target_length=324, feature_dim=40):
    # 텍스트 파일에서 행렬을 읽어옵니다.
    with open(feature_path, 'r') as file:
        matrix = np.array([list(map(float, line.split())) for line in file])

    # 행 길이에 따라 자르거나 패딩을 추가합니다.
    if matrix.shape[0] > target_length:
        matrix = matrix[:target_length, :]  # 행 길이가 너무 길면 자릅니다.
    elif matrix.shape[0] < target_length:
        # 행 길이가 짧은 경우 패딩을 추가합니다.
        padding = np.zeros((target_length - matrix.shape[0], feature_dim))
        matrix = np.vstack((matrix, padding))
    
    # 행렬을 transpose하여 (40, 324) 형태로 만듭니다.
    return matrix.T

def load_features(base_folder):
    # 모든 행렬을 저장할 리스트입니다.
    all_features = []
    
    # features_real 폴더 내의 모든 서브폴더를 순회합니다.
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        feature_path = os.path.join(folder_path, 'features(evs_enc).txt')
        if os.path.isfile(feature_path):
            matrix = load_and_pad_matrix(feature_path)
            all_features.append(matrix)
    
    # 모든 행렬을 하나의 numpy 배열로 변환합니다.
    return np.array(all_features)


if __name__ == "__main__":
    # features 변수에 결과를 저장합니다.
    features_real_temp = load_features('features_real_temp')
    print(features_real_temp.shape)  # 출력 결과는 (파일 수, 40, 324) 형태가 됩니다.

    # features_fake_temp = load_features('features_fake_temp')

    # print(features_real_temp[0].shape)

    # for i in range(1, 7):
    #     mean_real_temp = np.mean(features_real_temp[i])
    #     variance_real_temp = np.var(features_real_temp[i], ddof=0)
    
    #     mean_fake_temp = np.mean(features_fake_temp[i])
    #     variance_fake_temp = np.var(features_fake_temp[i], ddof=0)
    
    