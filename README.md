# Deep Mutual Learning
Model Distillation 이란 Teacher(큰) 모델로부터 Student(작은) 모델을 학습시키는 것이다. 이를 통해 상대적으로 매우 큰 교사 모델로 네트워크가 작은 학생 모델을 훈련함으로서 더 적은 파라미터로 빠르고 비슷한 성능을 내는 학생 모델을 얻어낼 수 있다. Deep Mutual Learning은 이와 다르게 학생 모델들로만 구성하여 서로 협력해 나가는 방식으로 모델이 훈련된다. 이를 통해 더 좋은 성능을 이끌어 낼 수 있다 한다. 
# Paper
DML 논문에서는 아래와 같이 학생 네트워크가 구성된다. <br>
<img src = https://user-images.githubusercontent.com/55969260/72857528-357f0580-3d01-11ea-9732-e2132c8261e4.png> <br>
Loss function은 모델 p들 사이의 KL Divergence를 사용하여 모델들 사이의 확률분포의 차이를 일반적인 loss 함수에 더하여 역전파를 수행한다. <br>
<img src = https://user-images.githubusercontent.com/55969260/72857719-d79eed80-3d01-11ea-82cc-a07031425818.png> <br>
<br>
실험은 CIFAR-100으로 하였을 때 DML - Inedependent가 양수인 것으로 보아 성능이 다 높아졌음을 알 수 있다. <br>
<img src = https://user-images.githubusercontent.com/55969260/72857859-38c6c100-3d02-11ea-91af-203d0dec0f79.png><br>
# My Experiment
## Data
Naver 영화평과 IMDB를 통한 Sentiment Analysis 그리고 MNIST Classifier. SA는 Bi-LSTM 네트워크를 사용하였고 MNIST는 CNN을 사용하였다. 
## 결과
- Naver 영화평 : 한국어 Embedding을 사용하여 SA를 수행하였을 경우 단일 모델과 성능에 큰 차이가 없음을 확인했다. 단일 모델의 Acc가 80.26% 나온다면 2개의 네트워크를 동시에 학습 시 평균 80.24%가 나온다. 네트워크의 수를 3개로 늘렸을시 단일모델은 80.15, 3개 모델의 평균은 80.35로 성능이 조금 높아짐을 확인했다. KLD loss를 분석해본 결과 4.35e-5등 매우 작은 확률분포 차이를 보여주었다. 그렇기에 일반적인 loss에 KLD loss가 더해질 시 매우 작은 값이 더해져서 성능에 큰 영향을 끼치지 않았음을 확인할 수 있었다. 
- IMDB : 영어 데이터로 SA수행시 네트워크 2개, 3개로 DML적용 결과는 아래와 같다. <br>
<img src = https://user-images.githubusercontent.com/55969260/72858404-09b14f00-3d04-11ea-9a67-6d6a489625e9.png> *두 개 네트워크<br>*
<img src = https://user-images.githubusercontent.com/55969260/72858420-1cc41f00-3d04-11ea-9ed7-cc34f1825955.png> *세 개 네트워크<br>*
결과가 보여주듯 성능이 모두 증가됨을 확인 할 수 있었다. KLD loss를 분석해본결과 Iterator가 진행될수록 KLD loss값이 증가함을 보였고 (0.01~0.02), 이는 위의 Naver 영화평 SA의 loss와 매우 차이나는 값으로 많은 분포 차이에서의 데이터가 일반적인 loss함수에 영향을 꽤 끼쳤다고 할 수 있다. 다만 모델을 추가적으로 돌려봤을 때 단일 네트워크가 더 잘하는 모델이 만들어질 때도 있었다. 
- MNIST : MNIST의 결과는 아래와 같다. <br>
<img src =https://user-images.githubusercontent.com/55969260/72858718-0f5b6480-3d05-11ea-8096-494231c8c848.png> <br>
단일 네트워크에 비해 DML을 적용하여 네트워크의 수를 늘렸을시 Acc가 높아짐을 볼 수 있었다. 
## Plus
기존 논문에 대해 Pytorch로 구현한 코드를 돌려봤고 성능이 잘 나옴을 확인했다. 이때 KLD loss를 분석했을시 0.1~0.3 사이로 분포의 차이를 일반적인 loss에 잘 반영함을 봤다. 이에 NLP Task를 수행시 성능이 왔다 갔다 한 이유는 KLD loss가 상대적으로 작아서 일반 loss에 잘 반영되지 못하여 이러한 결과를 냄이라 생각되고 논문에선 데이터를 다 이미지로 구성하였는데 뭔가 Iterator가 진행될 수록 이미지 데이터는 서로간의 확률분포가 다름을 즉, 서로 다른 네트워크들이 다른 부분을 보고 있어 자신이 본 결과 값을 다른 네트워크에 잘 알려주어 이러한 loss가 잘 반영되지 않았나 생각된다.
<br>
