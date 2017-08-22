# Keras KPI Distribution Estimator

## What is this.
これは、深層学習で不十分なデータの分布から背景を推定するテクノロジーです  

アプローチとしては、ベイズや深層学習が考えれらますが、実装の難易度やノウハウの関係から、深層学習を用います  

## 問題点
様々なクロスでのデータのサンプリングを行い、ターゲティングを行う必要があるのですが、現状は全てをサンプリングすることはできません　　　
主にコスト的な側面の問題や、実験に対するROI（投資対効果）が、最大化できるかの懸念があるためです　　

提案として、サンプル数が少ない状態でも深層学習を用いると、求めたい分布が得られることを示します  

## 理論
任意の多次元データに対して、全てのデータの情報が埋まっているわけでないが、ある程度、欠落を何らかの周辺情報から機械学習のアルゴリズムで予想できるならば、予想可能であろう、という仮説（物事の事象を説明しうる多様体が存在するという仮説）を立てます  

この仮説が正しいとする際、ある程度のサンプリングで、ディープラーニングは多様体を獲得し、未知の欠けてしまったデータに対して予測性能を発揮できることになります　　

## ネットワークの定義 
```python
inputs = Input(shape=(2,))
x      = Dense(100, activation='relu')(inputs)
x      = Dropout(0.7)(x)
x      = Dense(100, activation='relu')(x)
x      = Dropout(0.7)(x)
x      = Dense(1, activation='sigmoid')(x)
est    = Model(inputs, x)
est.compile(optimizer=Adam(), loss='mse')  
```
<p align="center">
  <img width="550px" src="https://user-images.githubusercontent.com/4949982/29544869-021e17b4-8726-11e7-8c80-c46e1d8700ff.png">
</p>
