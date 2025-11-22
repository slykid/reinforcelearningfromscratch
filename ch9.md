# Chapter 9: 정책경사법 (Policy Gradient Methods)

---

## 목차

**CHAPTER 9. 정책경사법 (Policy Gradient Methods)**

1. [9.1 가장 간단한 정책경사법](#91-가장-간단한-정책경사법)
2. [9.2 REINFORCE](#92-reinforce)
3. [9.3 베이스라인](#93-베이스라인-baseline)
4. [9.4 Actor-Critic](#94-actor-critic)
5. [9.5 정책 기반 방법의 장점](#95-정책-기반-방법의-장점)
6. [9.6 정리](#96-정리)

---

## 9.1 가장 간단한 정책경사법

### 가치 기반 vs 정책 기반

#### 가치 기반 방법 (Value-based Methods)
- Q학습, SARSA, 몬테카를로 방법 등
- 가치 함수를 모델링한 후, 간접적으로 정책 도출
- 예: $Q(s,a)$를 학습 → 정책: $\arg\max Q(s,a)$

#### 정책 기반 방법 (Policy-based Methods)
- **가치 함수를 거치지 않고 정책을 직접 표현**
- 신경망을 사용하여 정책 $\pi_\theta(a|s)$ 모델링 ($\theta$는 파라미터)
- 경사 상승법으로 기대 누적 보상 최대화

### 정책 경사법의 기본

**목적**: 파라미터 $\theta$를 조정하여 궤적 $\tau$의 기대 누적 보상 $J(\theta)$를 최대화

$$J(\theta) = \mathbb{E}[G(\tau)]$$

여기서 $G(\tau)$는 궤적의 누적 할인 보상

**경사 상승법**:

$$\theta \leftarrow \theta + \alpha \cdot \nabla_\theta J(\theta)$$

### 로그 기울기 트릭 (Log-Gradient Trick)

환경 동역학을 모르더라도 기울기 계산 가능:

$$\nabla_\theta \log \pi_\theta(a|s)$$

### 가장 간단한 정책경사법

전체 궤적의 누적 보상 $G_0$을 모든 시점에 사용:

$$\nabla_\theta J(\theta) = \sum_{t=0}^{T} G_0 \cdot \nabla_\theta \log \pi_\theta(a_t|s_t)$$

여기서 $G_0 = \sum_{t=0}^{T} r_t$는 전체 에피소드의 총 보상

**문제점**:
- 모든 시점에 동일한 가중치 사용
- 과거의 보상까지 현재 행동 평가에 포함 → 분산 증가

---

## 9.2 REINFORCE

### REINFORCE 알고리즘

**핵심 개선**: 시점 $t$ 이후의 보상만 고려

$$\nabla_\theta J(\theta) = \sum_{t=0}^{T} G_t \cdot \nabla_\theta \log \pi_\theta(a_t|s_t)$$

**수익 계산**:

$$G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$$

**업데이트 수식**:

$$\theta \leftarrow \theta + \alpha \cdot G_t \cdot \nabla_\theta \log \pi_\theta(a_t|s_t)$$

**개선 효과**:
- 이전 시점의 보상은 현재 행동 평가에서 제외
- 불필요한 노이즈 감소

---

## 9.3 베이스라인 (Baseline)

### 분산 문제

REINFORCE의 주요 문제:
- **높은 분산**: Monte Carlo 추정으로 인한 노이즈
- 학습이 불안정하고 느림

### 베이스라인 도입

**핵심 아이디어**: 보상 $G_t$에서 기준값 $b(s)$를 빼도 정책 경사의 기댓값은 변하지 않음

$$\nabla_\theta J(\theta) = \mathbb{E}_\pi[\nabla_\theta \log \pi_\theta(a|s) \cdot (G_t - b(s))]$$

**증명**:

$$\begin{align}
\mathbb{E}_\pi[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)] &= \sum_a \pi_\theta(a|s) \cdot \nabla_\theta \log \pi_\theta(a|s) \cdot b(s) \\
&= b(s) \cdot \nabla_\theta[\sum_a \pi_\theta(a|s)] \\
&= b(s) \cdot \nabla_\theta(1) = 0
\end{align}$$

### 최적 베이스라인

상태 가치 함수 $V^\pi(s)$를 베이스라인으로 사용하면 Advantage 함수:

$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$

**업데이트 수식**:

$$\theta \leftarrow \theta + \alpha \cdot (G_t - b(s_t)) \cdot \nabla_\theta \log \pi_\theta(a_t|s_t)$$

---

## 9.4 Actor-Critic

### Actor-Critic 구조

**두 개의 네트워크**:
- **Actor (행위자)**: 정책 $\pi_\theta(a|s)$ 학습
- **Critic (평가자)**: 가치 함수 $V_w(s)$ 학습

**특징**:
- 가치 기반 + 정책 기반 방법의 결합
- TD 방식으로 에피소드 완료를 기다리지 않고 학습 가능

### TD Error를 이용한 Advantage 추정

TD error:

$$\delta_t = r_t + \gamma V_w(s_{t+1}) - V_w(s_t)$$

이를 Advantage의 추정값으로 사용:

$$\delta_t \approx A^\pi(s,a)$$

### 업데이트 수식

**Critic 업데이트** (가치 함수):

$$w \leftarrow w + \alpha_w \cdot \delta_t \cdot \nabla_w V_w(s_t)$$

**Actor 업데이트** (정책):

$$\theta \leftarrow \theta + \alpha_\theta \cdot \delta_t \cdot \nabla_\theta \log \pi_\theta(a_t|s_t)$$

---

## 9.5 정책 기반 방법의 장점

### 가치 기반 방법과의 차이

**정책 기반 방법의 장점**:

1. **연속 행동 공간 처리**
   - 가우시안 분포 등으로 연속 행동 자연스럽게 표현
   - 로봇 제어 등에 적합

2. **확률적 정책 학습**
   - 최적 정책이 확률적인 경우에 적합
   - 각 행동의 확률을 직접 모델링

3. **부드러운 정책 업데이트**
   - Softmax 확률이 점진적으로 변화
   - 안정적인 학습

---

## 9.6 정리

### 알고리즘 발전 과정

```
가장 간단한 정책경사법
    ↓ (그 시점 이후의 보상만 고려)
REINFORCE
    ↓ (베이스라인 추가로 분산 감소)
REINFORCE with Baseline
    ↓ (신경망으로 베이스라인 학습)
Actor-Critic
```

### 각 방법의 업데이트 수식 비교

1. **가장 간단한 정책경사법**:
   $$\theta \leftarrow \theta + \alpha \cdot G_0 \cdot \nabla_\theta \log \pi_\theta(a_t|s_t)$$

2. **REINFORCE**:
   $$\theta \leftarrow \theta + \alpha \cdot G_t \cdot \nabla_\theta \log \pi_\theta(a_t|s_t)$$

3. **REINFORCE with Baseline**:
   $$\theta \leftarrow \theta + \alpha \cdot (G_t - b(s_t)) \cdot \nabla_\theta \log \pi_\theta(a_t|s_t)$$

4. **Actor-Critic**:
   $$\begin{align}
   \delta_t &= r_t + \gamma V_w(s_{t+1}) - V_w(s_t) \\
   w &\leftarrow w + \alpha_w \cdot \delta_t \cdot \nabla_w V_w(s_t) \\
   \theta &\leftarrow \theta + \alpha_\theta \cdot \delta_t \cdot \nabla_\theta \log \pi_\theta(a_t|s_t)
   \end{align}$$

---

**참고**: 이 자료는 『밑바닥부터 시작하는 딥러닝 4 - 강화학습편』 Chapter 9의 핵심 내용을 정리한 것입니다.
