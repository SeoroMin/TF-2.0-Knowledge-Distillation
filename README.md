# TF 2.0 Knowledge Distillation

## Fork 

from https://github.com/tripdancer0916/keras-knowledge-distillation

## Experiment Result

https://tobigs.gitbook.io/knowledge-distilation/

## How to use

Metric Evaluation baseline.py vs knowledge_distillation.py vs teacher_model.py




# khu_nlp_lab

작성 날짜 : 2021-05-07

논문 제목 : Distilling the Knowledge in a Neural Network

-----

## Background

오버피팅을 피하기 위해 앙상블 기법을 사용.
하지만 앙상블은 계산시간이 많이 걸린다는 단점이 있어, 앙상블만큼의 성능과 '적은 파라미터 수'를 가진 nn모델이 필요.

Knowledge + Distillation -> 지식 증류
증류 : 액체를 가열하여 생긴 기체를 냉각하여 다시 액체로 만드는 일.

![image](https://user-images.githubusercontent.com/57586314/117425320-a7627900-af5d-11eb-8cd4-53e9734c11de.png)

즉 딥러닝에서 지식 증류는 큰 모델(teacher network)로부터 증류한 지식을 작은 모델(student network)로 transfer하는 과정이다. 

## How to Knowledge Distillation

1) Soft Label
![image](https://user-images.githubusercontent.com/57586314/117425493-c95bfb80-af5d-11eb-97c9-c598a497df9d.png)
어떠한 label을 예측한다고 가정했을 때, 값이 [0, 0, 1, 0]보다 [0.1, 0.1, 0.7, 0.1]로 soft하게 함으로써 다른 label을 예측할 때 도움을 줄 수 있음.
![image](https://user-images.githubusercontent.com/57586314/117425608-e42e7000-af5d-11eb-83be-da7aa986c7f7.png)

2) distilliation loss 
큰 모델을 학습 시킨 후, 작은 모델을 다음과 같은 손실함수를 통해서 학습함.
![image](https://user-images.githubusercontent.com/57586314/117428038-a0893580-af60-11eb-86b9-4a76563c8512.png)

Teacher Network 학습 -> Student Network 학습
Student Network soft prediction + Teacher Network soft label -> distillation loss 구성
Student Network (hard) prediction + Original (hard) label -> classification loss 구성

Teacher 모델의 손실값과 Student 모델의 cross entropy 값을 더해 갱신해나감. 

## Experiment

Mnist 데이터 셋

