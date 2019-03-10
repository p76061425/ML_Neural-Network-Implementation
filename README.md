# <center> Neural Network Implementation </center>
* 手刻神經網路作Digits Classification
* 手刻Autoencoder並作denoise及dropout處理

## Usage

### Neural Network
```sh
$ python3 nn_model.py [-h] 
```

| optional Options | Description |
| ---              | --- |
| -h --help       | show this help message and exit |
| -b BATCH_SIZE        | batch size,(default=64) |
| -lr LEARNING_RATE  | learning rate,(default=0.1) |
| -ep EPOCH     | iters num,(default=10000)|

* 結果圖片會存在result/1資料夾內


### Autoencoder
```sh
$ python3 autoencoder.py [-h] 
```

| optional Options | Description |
| ---              | --- |
| -h --help       | show this help message and exit |
| -mode MODE    | train=training and show result, show=show result,(default=train) |
|  -b BATCH_SIZE  | batch size,(default=64) |
| -lr LEARNING_RATE     | learning rate,(default=0.1)|
| -ep EPOCH     | iters num,(default=10000)|
| -dn DENOISE |denoise, 0=False, 1=True, (default=1)|
| -drop DROPOUT | dropout, 0=False, else=dropout_rate,(default=0.5) |

* 結果圖片及模型會存在result/2資料夾內

## Report

## Digits Classification

### __Implement two neural networks with </br> (a) wide hidden layer and </br> (b) deep hidden layer to classify the digits in MNIST dataset.  </br>then show the accuracy and loss curve of the testing data for each model.__

#### __Model parameters:__
* __batch_size = 64__
* __learning_rate = 0.1__
* __epoch = 20__


#### __(a) wide hidden layer__

|Wide Model | Neurons | Activation |
|---        |---      |---         |
|Input Layer  | 784 | -       |
|Hidden Layer | 256 |  ReLU   |
|Output Layer | 10  | Softmax |


* __accuracy__
![](https://i.imgur.com/q0VoOUX.png)
![](https://i.imgur.com/t0Q6SZh.png)
* __loss__
![](https://i.imgur.com/IvgZiGz.png)
![](https://i.imgur.com/9zVQKzk.png)


#### __(b) deep hidden layer__

|Wide Model | Neurons | Activation |
|---        |---      |---         |
|Input Layer   | 784 | -       |
|Hidden Layer 1| 204 |  ReLU   |
|Hidden Layer 2| 202 |  ReLU   |
|Output Layer  | 10  | Softmax |


![](https://i.imgur.com/knAn5V3.png)
![](https://i.imgur.com/Ef7AyXA.png)
![](https://i.imgur.com/3Yz1wr0.png)
![](https://i.imgur.com/gh34EKJ.png)

* __上圖為訓練了20個epoch的accuracy和loss，在實作了wide hidden layer network和deep hidden layer network的比較之後，可以觀察到deep hidden layer network在最終的精準度上是略較 wide hidden layer network還要高的，神經網路在一般情況下深度越深的效果越好。__

## Autoencoder
### __Implement an autoencoder (AE) to learn the representation of the MNIST datasets.__

#### __Model parameters:__
* __batch_size = 64__
* __learning_rate = 0.01__
* __epoch = 20__

|Wide Model | Neurons | Activation |
|---        |---      |---         |
|Input Layer  | 784 | -       |
|Hidden Layer | 128 |  ReLU   |
|Output Layer | 784 | Sigmoid |

#### __a. Show the results of the AE-based dimension reduction__ 

![](https://i.imgur.com/FaKcpim.png)


#### __b. Visualize the reconstruction results and the filters.__ 

* __reconstruction results__
![](https://i.imgur.com/EKjHGe7.png)

* __filters__
![](https://i.imgur.com/rt7Pcvp.png)


#### __c. Apply denoise and dropout mechanism, and visualize the reconstruction results and the filters.__ 
* __denoise:True__
##### __dropout rate=0.3__ </br>

* __reconstruction results__ </br>
![](https://i.imgur.com/pwx6GOZ.png)

* __filters__ </br>
![](https://i.imgur.com/8smoCif.png)

* __loss__</br>
__118.84398303229302 </br>
105.82029540333674 </br>
101.58097784242122 </br>
99.54880212478625 </br>
97.61855675552452 </br>
96.34956672822764 </br>
95.64395347190118 </br>
95.28998787251157 </br>
94.39119154332509 </br>
94.0588533673074 </br>
93.67034662743141 </br>
93.54857179649814 </br>
93.11056580151774 </br>
93.0339604673621 </br>
92.71219558239682 </br>
92.68021584747451 </br>
92.58595073038595 </br>
92.37826216701657 </br>
92.16426406065743__


##### __dropout rate=0.5__ </br>

* __reconstruction results__ </br>
![](https://i.imgur.com/kdCQWuO.png)


* __filters__ </br>
![](https://i.imgur.com/f1lQefP.png)

* __loss__</br>
__134.10843117407825 </br>
119.65076758300864 </br>
114.10359240038784 </br>
111.07228044783366 </br>
109.1115876641319 </br>
107.74818291241075 </br>
107.21445799533033 </br>
106.31078543866472 </br>
105.74565855582188 </br>
105.1224298176707 </br>
104.70423828210544 </br>
103.89410422469419 </br>
103.62096139459072 </br>
103.15376267250367 </br>
102.82374519382584 </br>
102.8229997676354 </br>
102.61613930906108 </br>
102.12456704162484 </br>
102.14564543005211__


##### __dropout rate=0.7__ </br>

* __reconstruction results__ </br>
![](https://i.imgur.com/XueMjWV.png)


* __filters__ </br>
![](https://i.imgur.com/lPi55Eq.png)

* __loss:__</br>
__155.60497814717988 </br>
139.40536520966398 </br>
132.8257615246033 </br>
129.34472207673667 </br>
126.92866727153346 </br>
125.25035193704254 </br>
124.04271310670009 </br>
122.93852882845252 </br>
122.28391630149758 </br>
121.41698687108557 </br>
121.02655336449139 </br>
120.43690609493795 </br>
120.13572135149354 </br>
119.66361225781348 </br>
119.22806662300013 </br>
119.31152860529934 </br>
118.87773935171258 </br>
118.39480283097203 </br>
118.4303557903164__


#### __從上面的結果圖可以觀察到，有作denoise及dropout時，reconstruction results的圖片會比沒有作denoise及dropout的結果還要來得模糊一些，但filters visualize出來後的結果輪廓特徵是比較明顯的，而且dropout rate越高，reconstruction results的圖片越模糊，loss也會較高。使用此兩個技術，可以增加抗噪能力和防止overfitting的問題。__

