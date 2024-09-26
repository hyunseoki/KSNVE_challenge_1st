## 학습 구현 프로세스

### 1. 데이터 준비: CSV 파일을 NPY로 변환

**데이터셋 폴더 구조:**
```
./dataset
├── track1
│ ├── eval
│ ├── test
│ └── train
└── track2
├── eval
├── test
└── train
```


**변환 실행:**

```bash
bash ./script/make_dataset.sh
```

**2. 모델 학습**
```bash
### 시드 바꿔서 두 번 반복
bash ./script/train.sh
```

**3. 추론 학습**
```bash
bash ./script/eval.sh
```

