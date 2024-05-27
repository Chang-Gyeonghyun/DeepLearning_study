## Benchmark_Architecture

이 디렉토리에는 CIFAR-100과 MNIST 데이터셋에 대한 벤치마크 아키텍처가 포함되어 있습니다. 각 데이터셋 폴더는 데이터, 모델, 유틸리티 및 학습과 테스트를 위한 스크립트를 조직화하는 구조를 가지고 있습니다.

### 디렉토리 구조

```
Benchmark_Architecture/
├── Cifar-100/
│ ├── data/
│ ├── models/
│ ├── utils/
│ ├── test.py
│ └── train.py
└── Minist/
├── data/
├── models/
├── utils/
├── test.py
└── train.py
```

### 하위 디렉토리 및 파일 설명

- data/
    
    이 폴더는 학습 및 테스트에 필요한 데이터를 로드합니다.
    
- models/
    
    이 폴더에는 모델 정의가 포함되어 있습니다. 이 디렉토리에서 신경망의 아키텍처를 찾을 수 있습니다.
    
- utils/
    
    유틸리티 함수 및 보조 스크립트가 이 폴더에 저장되어 있습니다.
    
- test.py
    
    이 스크립트는 학습된 모델을 테스트 데이터셋에서 평가하는 역할을 합니다. 모델을 로드하고, 테스트 데이터를 처리하며, 성능 지표를 계산합니다.
    
- train.py
    
    이 스크립트는 모델을 학습시키는 데 사용됩니다. 데이터 로딩, 모델 학습 및 추후 사용을 위한 학습된 모델 저장을 처리합니다.
    

## 사용 방법

스크립트 실행

`train.py` 스크립트는 모델을 학습시키는 데 사용됩니다. 이 스크립트는 다양한 모델을 선택하여 학습할 수 있도록 구성되어 있습니다. 기본적으로 `cnn` 모델을 사용하지만, `resnet` 또는 `mobilenet` 모델을 선택할 수도 있습니다. 다음은 스크립트를 실행하는 방법입니다.

```bash
# 기본적으로 `cnn` 모델을 사용하여 학습을 시작
python train.py

# 특정 모델(ResNet 또는 MobileNet)로 학습
# --model 인자를 사용하여 모델 이름을 지정
python train.py --model resnet
python train.py --model mobilenet
```

## GenerativeModels/DCGAN

이 디렉토리에는 DCGAN (Deep Convolutional Generative Adversarial Network) 모델을 위한 코드와 관련된 파일들이 포함되어 있습니다. DCGAN은 이미지 생성을 위한 강력한 생성 모델입니다.

### 디렉토리 구조

```
GenerativeModels/
└── DCGAN/
├── data/
├── models/
│ ├── discriminator.py
│ └── generator.py
├── saved/
│ └── generator.pth
├── utils/
├── test.py
└── train.py
```

### 하위 디렉토리 및 파일 설명

- models/
    
    이 폴더에는 DCGAN의 생성자와 판별자 모델 정의가 포함되어 있습니다.
    

## Knowledge Distillation - TAKD

이 디렉토리에는 TAKD (Teacher Assistant Knowledge Distillation) 모델을 위한 코드와 관련된 파일들이 포함되어 있습니다. 이 프로젝트에서는 간단히 CNN 모델만 직접 구현하여 사용합니다.

### 디렉토리 구조

```
Knowledge_Distillation/
└── TAKD/
├── data/
├── models/
├── saved/
├── test.py
└── train.py
```

### 하위 디렉토리 및 파일 설명

### models/

이 폴더에는 CNN 모델 정의가 포함되어 있습니다.

### saved/

이 폴더는 학습된 모델 파라미터를 저장하는 곳입니다. 10 layer teacher model에서 2 layer student model로 지식을 전달하는 모든 경로에 대해 최종적으로 2 layer student model의 성능이 저장되어 있습니다.

### 사용 방법

**스크립트 실행**

`train.py` 스크립트는 CNN 모델을 학습시키는 데 사용됩니다.

```bash
python train.py --epochs 150 --lr 0.1 --momentum 0.9 --teacher 10 --student 2 --device cpu --checkpoint 10
```

1. **명령줄 인자 파싱:**
    - **`argparse`** 모듈을 사용하여 명령줄 인자를 파싱합니다.
    - 인자 설명:
        - **`-epochs`**: 총 학습 에포크 수 (기본값: 150)
        - **`-lr`**: 학습률 (기본값: 0.1)
        - **`-momentum`**: 모멘텀 값 (기본값: 0.9)
        - **`-teacher`**: teacher model의 layer (기본값: '10')
        - **`-student`**: student model의 layer (기본값: '2')
        - **`-device`**: CUDA 사용 여부 및 학습을 GPU에서 수행할지 여부 (기본값: 'cpu')
        - **`-checkpoint`**: checkpoint model state (기본값: '10')