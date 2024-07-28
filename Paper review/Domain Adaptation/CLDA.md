## 지도 학습 (Supervised Training)

- 소스 및 타겟 도메인의 라벨링된 샘플에서 교차 엔트로피 손실을 최소화
- **손실 함수**:
    
    $$
    L_{sup} = - \sum_{k=1}^K (y_i)k \log(F(G(x{il}))_k)
    
    $$
    

![clda.png](https://d3i71xaburhd42.cloudfront.net/e78775b0afa70755fdf31121be6300fe82c61011/4-Figure2-1.png)

## 도메인 간 대조 정렬 (Inter-Domain Contrastive Alignment)

### 클러스터 중심 계산

- **소스 도메인**:
    
    $$
    C_s^k = \frac{\sum_{i=1}^{B} 1\{y_s^i = k\} F(G(x_s^i))}{\sum_{i=1}^{B} 1\{y_s^i = k\}}
    
    $$
    
- **타겟 도메인**: 의사 라벨 $\hat{y}_t$를 사용하여 동일한 방법으로 계산. 따로 임계값이 존재하지 않음.

### 클러스터 손실

- NT-Xent 대조 손실을 사용하여 소스와 타겟 도메인의 각 클래스 중심 간의 유사성을 최대화
    
    $$
    L_{clu}(C_t^i, C_s^i) = - \log \frac{h(C_t^i, C_s^i)}{h(C_t^i, C_s^i) + \sum_{r=1}^K \sum_{q \in \{s, t\}} 1\{r \neq i\} h(C_t^i, C_q^r)}
    $$
    

## 인스턴스 대조 정렬 (Instance Contrastive Alignment)

### 강한 증강

- 라벨링되지 않은 타겟 이미지의 강하게 증강된 버전 생성
- **NT-Xent 손실**:
    
     $L_{ins}(\tilde{x}_t^i, x_t^i) = - \log \frac{h(F(G(\tilde{x}t^i)), F(G(x_t^i)))}{\sum{r=1}^B h(F(G(\tilde{x}t^i)), F(G(x_t^r))) + \sum{r=1}^B 1\{r \neq i\} h(F(G(\tilde{x}_t^i)), F(G(\tilde{x}_t^r)))}$
    

## 전체 프레임워크와 훈련 목표

### 전체 손실 함수

- 지도 학습 손실, 도메인 간 대조 정렬 손실, 인스턴스 대조 정렬 손실을 결합한 전체 손실 함수
    
    $$
    L_{tot} = L_{sup} + \alpha L_{clu} + \beta L_{ins}
    $$
    

이 프레임워크를 통해 CLDA는 소스 도메인과 타겟 도메인 간의 도메인 격차를 줄이고, 타겟 도메인의 라벨링되지 않은 데이터를 효과적으로 활용하여 성능을 향상시킨다.