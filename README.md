# train.py 실행 명령어 

```bash
python train.py --feature evs --feature_dim 12 --real LibriSpeech/train-clean-360/all_audio --fake asvspoof2019/LA/ASVspoof2019_LA_train/flac --batch_size 32 --epochs 100000 --model lstm
```


# inference.py 실행 명령어

```bash
python inference.py --model model_weights.pt --in features_fake_for_test.txt --out result.txt
```

결과 results.txt에 저장됨. 1에 가까울수록 real, 0에 가까울수록 fake.

# 저장된 데이터
features_real과 features_fake에 모든 feature 파일 존재. 

# 깃허브 사용법
