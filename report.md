## 摘要



### Sec. 2 介绍

在交通拥堵指数预测的题目中，需要根据过去的交通拥堵指数和某一时间段内的网约车GPS数据预测某一时刻的交通拥堵指数TTI。经过预处理和特征工程之后，我们将问题转化为一个回归问题。回归问题是机器学习中一个基本问题，受到了很多研究人员的关注，该问题的解决将有助于提升机器学习技术在多个应用场景中的表现，比如在城市管理中对交通数据进行预测、推荐系统中对客户进行个性化推荐。

目前，回归任务常用集成学习算法，比如随机森林、XGBoost、lightGBM这三种。

其中，随机森林[1]是一种传统方法，提出时间较早。Leo Breiman于2001年在Random Forests中给出了详细介绍，该算法的独特之处在于对训练集进行随机自助采样来训练每一棵决策树和训练决策树时引入随机属性划分。引入这两点的随机性之后，随机森林不仅在准确性上和AdaBoost不相上下，而且对噪音更加鲁棒。

XGBoost[2]由。。。。。提出，。。。。。。。。。（**TODO**）

LightGBM是一种新兴的方法。为了处理在训练集特征维数过高和数据规模过大时GBDT的效率和可扩展性问题，微软团队在2017年提出了lightGBM算法[2]。该算法的独特之处在于提出了两种新技术GOSS和EFB：其中GOSS保留梯度大的样本，随机去掉梯度小的样本；EFB将互斥特征进行Bundling，将该问题归约至图着色问题，并使用贪心算法解决。和传统的GBDT算法比较，lightGBM在保持几乎相同的准确率的前提下，将速度提高了约20倍。

针对深圳北站交通拥堵指数预测的具体问题，我们分别使用这三种算法实验，但均存在超过预期的泛化误差。我们针对已有方法存在的问题，采用集成学习，提出将这三个模型加权结合，找到了解决预测TTI的更加准确的方法。我们利用华为云的评测平台，对新方法进行了测试，和原有方法相比，结果提高了近1个百分点。*（？表述感觉有点问题）*

本报告结构安排如下：第3-5部分详细介绍了我们在本次比赛中使用的模型和算法，第6-8部分对算法进行分析，第9-10部分对实验中的数值结果和实验过程进行展示分析以及讨论，最后在第11部分进行实验总结。

### Sec. 3 模型

观察数据，得到每个文件中数据的基本内容：

* 201901_201903.csv / 201910_11.csv / 20191201_20191220.csv / toPredict_train_gps.csv: 网约车订单追踪数据，每一条记录是一个网约车订单，其中包含了订单编号id_order、用户编号id_user和数量不定的五元组gps_records ( 经度，纬度，速度，方向，时间戳 )
* train_TTI.csv: 记录了每个路口从2019/1/1 0:00 - 2019/12/21 23:50的交通信息，每十分钟一条记录，包括 ( 路段代号id_road，交通拥堵指数TTI，路过车辆平均速度speed，标准时间time)
* toPredict_train_TTI.csv: 记录了每个路口从2019/12/21 7:30 - 2020/1/1 20:50的交通信息，具体有记录的时间段为 ( 奇数时:30 - 偶数时:20 ) ，例如 ( 7:30 - 8:20, 9:30 - 10:20 ) ，每十分钟一条记录，包括 ( 路段代号id_road，交通拥堵指数TTI，路过车辆平均速度speed，标准时间time ) .
* toPredict_noLabel.csv: 是需要预测交通拥堵指数的样本. ( 样本号id_sample，路段代号id_road，标准时间time ) . 时间段一般为某小时中的前半小时或后半小时，与toPredict_train_TTI.csv中的时间段几乎没有重合.

&emsp;我们需要先对 201901_201903.csv / 201910_11.csv / 20191201_20191220.csv 中的数据进行预处理，得到processed_train_data，对toPredict_train_gps.csv预处理得到processed_test_data. 利用processed_train_data和train_TTI.csv中的数据可以训练模型，由于需要用前1小时的数据对后10-30分钟进行预测，因此我们选择将所有数据按照时间窗口切分，分为训练集和验证集，并训练时间序列模型.

<img src="C:\Users\98061\AppData\Roaming\Typora\typora-user-images\image-20200615102328117.png" alt="image-20200615102328117" style="zoom:67%;" />

在数据预处理的部分，基本想法是遍历训练集，根据网约车gps的信息、运动方向和各路口的位置统计出各时间段经过各路口的车辆数。 由于训练集中已有路过车辆的平均速度，因此暂时不考虑统计网约车的速度情况。由于路段具有一定长度，并且不是一条直的线段，所以我们用平行四边形近似表示路段区域，将网约车GPS看作一个点的坐标，用点是否在区域中判断车辆是否经过这一路段。再通过比较网约车的方向和路段整体走向，得出网约车具体经过哪一个方向的路段。网约车gps数据清理的代码在 /data_processing 中。经过数据预处理，得出每个路口在各个时刻的车流量、TTI和通过路口的平均车速，存入train_#name.csv中。

在特征工程的部分，基本思路是使用前60分钟的数据预测后30分钟的数据，即用前60分钟的数据构成一个24维的向量，预测出一个三维向量。这里的TimeSlice是当天的第i个10分钟。获取输入和输出向量的方法如下，：
$$
x_i \leftarrow [Speed1, Car1, TTI1, TimeSlice1,Speed2, Car2, TTI2, TimeSlice2,Speed3, Car3, TTI3, TimeSlice3,Speed4, Car4, TTI4, TimeSlice4,Speed5, Car5, TTI5, TimeSlice5,Speed6, Car6, TTI6, TimeSlice6]
\\x_i \in \mathbb{R}^{24}
$$

$$
y_i \leftarrow [TTI7,TTI8,TTI9]
\\y_i \in \mathbb{R}^3
$$



我们希望训练出一个h：
$$
h:\mathbb{R}^{24} \rightarrow \mathbb{R}^3
$$

### Sec. 4 基本理论和方法

集成学习通过构建并结合多个学习器来完成学习任务。根据集成学习“好而不同”的原理，使用随机森林、XGBoost、LightGBM作为个体学习器，将他们进行结合，以获得比单一学习器更好的泛化能力。

### Sec. 5 方法



### Sec. 12 致谢

感谢助教老师提供了针对数据竞赛的阅读材料。

#### 参考文献

[1] Leo Breiman. Random Forests. In Machine Learning, 45(1):5-32, 2001. 

[2] Tianqi Chen and Carlos Guestrin. XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 785–794. ACM, 2016.

[3] Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-yan Liu. LightGBM: A highly efficient gradient boosting decision tree. In Advances in Neural Information Processing Systems, pages 3149–3157, 2017. 