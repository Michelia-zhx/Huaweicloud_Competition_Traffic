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

集成学习通过构建并结合多个学习器来完成学习任务。根据集成学习“好而不同”的原理，使用随机森林、XGBoost、LightGBM作为个体学习器，将他们进行结合，以获得比单一学习器更好的泛化能力。集成学习主要有两种方法：boosting和bagging。boosting算法的主要思想是首先根据初始训练集训练出一个基学习器，然后根据基学习器的表现调整样本权重，使得基学习器预测较差的样本在后续受到更多关注，重复这个过程，直到基学习器数量到达预先设定的T为止。bagging算法的主要思想是采用自主采样的方法从训练集中采样得到T个样本集合，使用每个采样集训练出一个学习器，然后把这T个学习器的结果相结合。

随机森林是一种bagging的方法，它和传统bagging的区别在于它又引入了一种随机性：属性选择的随机性。随机森林算法对于决策树的每个节点，先从该节点的属性集合种随机选择包含k个属性的子集，然后从中选出最优属性用于划分。

xgboost（**TODO**）

LightGBM基于XGBoost算法，再引入一种随机性：保留梯度大于阈值的向量，但是随机保留梯度小于阈值的向量。Ke[3]通过理论分析和实验证明这种随机性的引入可以使得结果优于随机采样。

(随机森林，boosting的伪代码？)

### Sec. 5 方法

在经过预处理和特征工程之后，输入向量是24维，输出向量是3维
$$
x_i \in \mathbb{R}^{24}
\\ y_i \in \mathbb{R}^3
$$


### Sec. 6 方法分析

rader.html

我们比较了不同算法性能、收敛速度、稳定性、适用性四方面。对于性能，我们用训练耗费的时间（单位秒）度量；对于收敛速度，我们用训练耗费的迭代轮数来度量；对于稳定性，我们用5次不同训练验证集划分下的mae的方差来度量；对于适用性，在TTI预测问题中，这两种模型都适用，所以我们设其为1。绘制出雷达图。

可以看出lightgbm的训练速度快于XGBoost，这和Ke[3]论文中提出来的论证结果是一致的，也体现了lightgbm在面对大量训练数据、高维特征的优越性，尽管这个问题的数据量比较大，但是lightgbm通过GOSS的加入，大大提高了运行速度。

在迭代轮数方面，两者相差很少。

在稳定性方面，实验次数不是很多，初步体现出xgboost优于lightGBM，但这一点还有待更多实验。

### Sec. 7 算法

*presedo.png中写了一下算法*



### Sec. 8 算法分析

### Sec. 9 数值结果

实验环境：windows + Intel Core i5

line.html

line.html中绘制了三个模型（lightgbm, xgboost, random_forest）在各个路口验证集上的mae误差。



### Sec. 10 讨论

在讨论部分中，我们主要研究了在TTI回归问题中降维、聚类、继承有效性的问题。首先是KMeans聚类。最初我们根据交通拥堵指数问题实际情况，考虑TTI可能会在不同时间段有不同分布模型，所以我们对输入的向量先进行了聚类，然后在 每一类中进行回归预测。但是，效果并不理想，最终我们放弃了先进行聚类的想法。实验结果见下表格。我们希望找到该问题不适合采用Kmeans聚类的原因。

| 簇数 | 验证集误差 |
| :--: | :--------: |
|  1   |  0.118500  |
|  2   |  0.147335  |
|  3   |  0.121700  |
|  5   |  0.131066  |

然后我们考虑使用降维进行数据可视化，我们对回归问题中每一个路口输入的24维向量进行PCA降维，降到2维，然后以此作为x,y轴，绘制出图像（PCA.png）。容易看出，在每个路口的图像中，图形都近似团成一簇，只有少量点散落在簇外，直观上并没有多簇的情况。再结合kmeans的原理，我们认为kmeans会把原先的一团分成几部分，反而破坏了模型的泛化能力，所以导致加入聚类效果不佳。同时由于降维之后我们发现有很少的点散落在簇外，我们考虑了异常点的情况。由于PCA降维有利有弊，既可以通过取较大的特征值对应的特征向量来去除噪音，也会丢失一部分特征信息，所以我们用降维前后的数据分别训练2种xgboost模型，然后根据问题实际特点将它们的预测结果加权相加，我们希望通过这种方法能够尽可能保留有效特征又去除一些噪音影响。

最后，我们根据各个模型在各路口预测结果的mae误差图，可以看出这三条线总体走势相同，但是在不同路口三种模型各有优劣。所以，我们希望按照不同权重将它们结合起来，以得到更高的泛化能力。经过实验，最终确定权重见伪代码。通过这一步，可以将泛化误差从0.1144提高到0.1078. 

### Sec. 11 结论



### Sec. 12 致谢

感谢助教老师提供了针对数据竞赛的阅读材料和撰写论文的建议。

### Sec. 13 参考文献

[1] Leo Breiman. Random Forests. In Machine Learning, 45(1):5-32, 2001. 

[2] Tianqi Chen and Carlos Guestrin. XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 785–794. ACM, 2016.

[3] Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-yan Liu. LightGBM: A highly efficient gradient boosting decision tree. In Advances in Neural Information Processing Systems, pages 3149–3157, 2017. 

### Sec. 14 附录

使用git来管理我们的代码，github项目地址：https://github.com/Michelia-zhx/Huaweicloud_Competition_Traffic

项目结构：项目树状图，（tree一下截屏）

**algorithm** : 不同算法的应用于本次TTI预测问题的代码

* train_array: 读入训练集的中间结果
* gen_train_npy.py ： 产生中间结果，暂时存到train_array中
* lightgbm.py: 使用lightGBM处理回归问题
  * train(train_df, test_df, params)   使用lightgbm进行训练
  * predict(road_id,timestamp,train_TTI,test_gps, models )使用训练出的lightgbm模型进行预测
* lstm.py： 使用LSTM模型处理回归问题
* random_forest2.py：使用随机森林处理回归问题
  * train(df,rf1)  使用随机森林在训练集上训练
  * predict(road_id, timestamp, train_TTI, test_gps,lst) 利用lst中的模型对road_id在timestamp时的TTI进行预测
* with_cluster_xgboost_alg.py：使用带聚类的XGBoost处理回归问题
  * gen_train：把数据经过特征工程变为24维的输入向量
  * gen_test：把测试数据经过特征工程变为24维的输入向量
  * train：训练模型
  * evaluate：使用model在X上进行评价
  * predict：使用model对测试集进行预测
* xgboost_alg.py：使用XGBoost处理回归问题
  * train(train_X, train_y, eval_X, eval_y, road_index)： 对road_index路口，使用train_X，train_y作为训练集，使用eval_X, eval_y作为验证集，进行训练。
  * gen_test(model, pred_df)： 进行特征工程，并对测试集产生预测结果
  * evaluate(model,X,y): 评价model在X上的mae误差

**data_preprocessing** : 数据预处理的代码

* count_car.py：通过对每10分钟经过各个路口的网约车计数，得到以十分钟为粒度的车流量数据
* gen_testset.py：产生每个路口的测试集
* gen_trainset.py：产生每个路口的训练集

**datasets**： 将训练数据和测试数据按照路口划分之后保存的csv结果

**photo**：根据20191201-20191220.csv中部分网约车的GPS信息，绘制出的GPS图形

* draw_road.py：绘制出20191201-20191220.csv中一部分网约车GPS图形，以经度作为横轴，维度作为纵轴

**processed_test_data**： 根据测试集上网约车GPS整理出的各个10分钟内每个路口的车流量

**processed_train_data**：根据训练集上网约车GPS整理出的各个时刻每个路口的车流量

**traffic**：部分训练集和测试集数据

**README.md**: 项目背景和题目要求

### Sec 15. 小组分工

* 181220067 张含笑：GPS轨迹绘制和观察，部分数据清洗，数据集划分，XGBoost回归，报告部分内容撰写
* 181220038 吕志存：部分数据清洗，RF回归，LightGBM回归，集成学习参数调整，报告部分内容撰写