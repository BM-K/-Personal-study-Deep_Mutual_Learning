# Deep Mutual Learning
## Model Distillation 이란 Teacher(큰) 모델로부터 Student(작은) 모델을 학습시키는 것이다. 이를 통해 상대적으로 매우 큰 교사 모델을 통해 네트워크가 작은 학생 모델을 훈련함으로서 더 적은 파라미터로 빠르고 비슷한 성능을 내는 학생 모델을 얻어낼 수 있다. Deep Mutual Learning은 이와 다르게 학생 모델들로만 구성하여 서로 협력해 나가는 방식으로 모델이 훈련된다. 이를 통해 더 좋은 성능을 이끌어 낼 수 있다 한다. 
# Paper
DML 논문에서는 아래와 같이 학생 네트워크가 구성된다. <br>
<img src = https://user-images.githubusercontent.com/55969260/72857528-357f0580-3d01-11ea-9732-e2132c8261e4.png> <br>
Loss function은 모델 p들 사이의 KL Divergence를 사용하여 모델들 사이의 확률분포의 차이를 일반적인 loss 함수에 더하여 역전파를 수행한다. <br>
<img src = https://user-images.githubusercontent.com/55969260/72857719-d79eed80-3d01-11ea-82cc-a07031425818.png> <br>
<br>
실험은 CIFAR-100으로 하였을 때 DML - Inedependent가 양수인 것으로 보아 성능이 다 높아졌음을 알 수 있다. <br>
<img src = https://user-images.githubusercontent.com/55969260/72857859-38c6c100-3d02-11ea-91af-203d0dec0f79.png><br>
# My Experiment
