### 기존 방법

반지도 학습(Semi-Supervised Learning, SSL)과 도메인 적응(Domain Adaptation, DA)에서는 다음과 같은 기법들이 사용되어 왔다:

- **FixMatch**: 약한 증강(weak augmentation)과 강한 증강(strong augmentation)을 통해 가짜 라벨(pseudo-label)을 생성하고, 이를 사용하여 모델을 학습.
- **ReMixMatch**: 다양한 데이터 증강과 분포 정렬(distribution alignment)을 통해 모델의 일반화 성능을 높임.

## AdaMatch Contribution

![adamatch.png](https://raw.githubusercontent.com/google-research/adamatch/master/media/AdaMatch.png)

### Random Logit Interpolation

source 예제만을 사용하여 얻은 source logit과 source와 target 예제를 결합하여 얻은 logit간의 연결된 공간에서 동일한 레이블을 생성하는 목적이다. 

source와 target 도메인의 로짓(logits)을 무작위로 보간하여 두 도메인의 분포를 혼합한다. 같은 source data에 대한 logit도 후자와 같이 source와 target이 같은 batch안에 들어가게 되면, batch norm에 의해 온전히 source만 넣었을 때와 logit의 차이가 생기게 된다. 이를 통해 모델이 두 도메인에서 일관된 예측을 할 수 있도록 합니다.

### Relative Confidence Threshold

target 도메인의 의사 라벨을 생성할 때, 약한 증강을 통해 얻은 예측 확률이 특정 신뢰도 임계값을 초과하는 경우에만 해당 값을 사용한다. 이를 통해 모델이 높은 신뢰도를 갖는 예측만을 학습하게 한다. FixMatch와 같은 방식이다. 

### Modified Distribution Alignment

target 도메인의 예측 분포를 source 도메인의 분포에 맞추기 위해 정규화한다. domain adaptation을 위한 작업.

## 알고리즘 상세 과정

1. **데이터 결합 (Data Combination)**:
    - Source_weak, Source_strong, Target_weak, Target_strong 네 가지로 나눈 데이터를 합쳐 data_combined와 source augmentation끼리 합친 source_combine으로 구분한다.
2. **로짓 계산 (Logit Calculation)**:
    - data_combined를 인코딩하고 분류기까지 넣어 logits 확률 값을 계산.
    - 이 중 source_data의 수 만큼을 따로 떼어 소스의 logit을 얻는다.
    - source_combine에서도 동일하게 인코딩과 분류기를 통해 logit 값을 계산.
3. **랜덤 로짓 보간 (Random Logit Interpolation)**:
    - 두 개의 다른 logit 값을 적절한 비율(lambda)로 합쳐 최종 source logit을 만든다.
    - source loss는 두 개의 logit을 비슷하게 만들어주는 역할.
4. **의사 라벨링 (Pseudo-Labeling)**:
    - Source_weak에 해당하는 logit을 따로 뽑아 pseudo-labeling을 한다.
    - Target_weak에 대해서도 동일하게 pseudo-labeling을 진행.
    - target loss는 Target_weak와 Source_weak의 pseudo-labeling을 합작하여 만들어진 label이 Target_weak의 logit을 softmax하여 예측할 때의 값과 일치하도록 만든다.

## 결론

**FixMatch**의 경우 의사 라벨링(pseudo-labeling)에서 약한 증강(weak augmentation)과 강한 증강(strong augmentation)을 적용하여 두 분포를 비슷하게 만든다. 그 이유는 모델이 데이터의 불확실성에 대해 더 강건해지도록 하기 위해서이다. 즉, 레이블이 없는 데이터(unlabeled data)에 대한 예측의 신뢰성을 강화하기 위함이다.

**AdaMatch**에서도 이러한 의사 라벨링 방식이 사용된다. 이는 target 도메인을 두고 봤을 때 해당 도메인에 대한 예측의 확실성을 높이기 위한 것이다. 

또한 단순히 약한 증강만을 사용하여 의사 라벨링을 하는 대신, source도메인의 분포 정렬(distribution alignment)을 통해 target 도메인의 분포를 source 도메인의 분포 쪽으로 이동시키는 방식입니다. 이를 통해 모델은 source와 target 도메인 간의 분포 차이를 줄이고, target 도메인에서도 더 정확한 예측을 할 수 있게 됩니다.