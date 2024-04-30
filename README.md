# 깃허브 푸시할때 비밀번호

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


# train.py 실행 명령어 

```bash
python train.py --real features_real_temp --fake features_fake_temp --batch_size 1 --epochs 1
```
실제로 학습 코드 실행할때는 batch_size 128, epochs 20으로 설정해서 해보자.


# inference.py 실행 명령어

```bash
python inference.py --model model_weights.pt --in features_fake_for_test.txt --out result.txt
```

결과 results.txt에 저장됨. 1에 가까울수록 real, 0에 가까울수록 fake.

# 저장된 데이터
features_real과 features_fake에 모든 feature 파일 존재. 