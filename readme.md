# 对数几率模型在Iris数据集上的实现
---
本代码属于机器学习的基础内容, 适合初学者使用.

本代码包含了：

- 数据的简单可视化

 通过箱线图观察数据分布特点:
<div align="center">
  <img src="img/01.png"  />
  <p>箱线图</p>
</div>

 通过多变量图发现属性间的相关性:

<div align="center">
  <img src="img/02.png"  />
  <p>多变量图</p>
</div>

- 二分类情况

 将species属性进行编码, 属于Setosa类型的样本标记为1, 否则标记为0, 则问题转化为一个二分类问题.
 
<div align="center">
  <img src="img/03.png"  />
  <p>损失函数以及准确率变化</p>
</div>

- 多分类情况

 将数据集中的species属性重新进行编码, 采用one-hot编码方式, 并使模型输出三个预测值. 然后取最大值作为预测结果. 则问题转化为一个多分类问题.
 

<div align="center">
  <img src="img/04.png"  />
  <p>损失函数以及准确率变化</p>
</div>

- 获取属性的重要程度

 获取各属性重要程度可以筛选掉不重要因素, 利于模型可视化.
<div align="center">
  <img src="img/05.png"  />
  <p>各属性重要程度</p>
</div>

- 模型动态可视化

 从可视化的角度阐述模型
 <div align="center">
  <img src="img/demo.gif"  />
  <p>动态描述</p>
</div>



<div align="center">
  <img src="img/06.png"  />
  <p>静态描述</p>
</div>
