## Domain-Adversarial Training of Neural Networks

도메인 적대적 학습(Domain-Adversarial Training)은 source와 target의 distribution이 다를 때 발생하는 문제를 완화하기 위한 방법론이다. 이 방법은 두 도메인 간의 간격을 줄여, 모델이 다양한 도메인에서 일관된 성능을 유지하도록 돕는다.

![Untitled](https://jamiekang.github.io/media/2017-06-05-domain-adversarial-training-of-neural-networks-fig1.jpg)

### Input Image로부터 Feature Vector 추출

먼저, 입력 이미지 x로부터 feature vector를 추출한다. 이 과정은 일반적인 이미지 처리 과정과 유사하다. 이미지가 입력되면, 이를 뉴럴 네트워크를 통해 feature vector로 변환한다.

### **label predictor**와 **domain classifier**로의 분기

추출된 feature vector는 두 가지 경로로 분기된다

- **Label Predictor (라벨 예측기)**: 이 경로는 기존 모델과 동일하게 작동하며, 입력 이미지의 클래스 라벨을 예측하는 역할을 한다. 이 부분은 학습 성능을 최대한 높이는 것이 목표이다.
- **Domain Classifier (도메인 분류기)**: 새로운 방법론에 해당하는 부분이다. 도메인 분류기의 역할은 해당 입력 이미지가 어떤 도메인에서 왔는지(source인지 target인지)를 구분하는 것이다. 그러나 도메인 분류기는 성능이 좋지 않게 만드는 것이 목표이다. 이는 도메인 분류기가 입력 이미지의 출처 도메인을 구분하지 못하게 하여, 모델이 두 도메인 간의 분포 차이를 인식하지 못하도록 하기 위함이다.

### 적대적 학습

도메인 적대적 학습의 핵심은 두 가지 목표가 서로 반대되는 방향으로 학습하는 것이다:

1. **Label Predictor의 성능을 최대화**: 이 경로는 입력 이미지의 클래스 라벨을 정확하게 예측하는 것을 목표로 한다. 따라서 일반적인 학습 과정과 동일하게 진행된다.
2. **Domain Classifier의 성능을 최소화**: 이 경로는 입력 이미지의 출처 도메인을 정확하게 예측하지 못하게 하는 것을 목표로 한다. 이를 위해, 도메인 분류기의 gradient는 음수를 붙여 반대로 backpropagation이 진행된다. 이렇게 함으로써, 도메인 분류기의 학습 성능이 떨어지게 된다.

이러한 적대적 학습을 통해, 라벨 예측기는 입력 이미지의 클래스 라벨을 잘 구분하게 되지만, 그 출처 도메인(source인지 target인지)은 구분하지 못하게 된다. 결과적으로, 두 도메인 간의 분포 차이가 줄어들게 된다.

### 결론

도메인 적대적 학습(Domain-Adversarial Training)은 두 도메인 간의 분포 차이를 줄여 모델의 일반화 성능을 높이는 강력한 방법론이다. 이를 통해, 도메인 간의 차이로 인해 발생하는 문제를 완화하고, 다양한 도메인에서 일관된 성능을 유지할 수 있다.