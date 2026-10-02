# MLE

# 0911

persistent Scheduling

* `grid`，GPU最高维度的调度单位，可以理解为SM
* 普遍`grid`可以使用一维，如果是计算的形状为二维，也可以使用二维的`grid`
  * 比方说
  * 二维矩阵乘，每个`grid`负责输出矩阵的一个`tile`
  * 图像的处理

回到persistent scheduleing中，我什么persistent schedule可以提高效率？

1. launch 一次keneral有固定的成本，这种实现lanuch一次keneral，可以多次复用，减少成本，最重要的是初始化 `barrier`和`tmem`
2. 大幅度改善L2缓存的命中率？主要作用是让使用差不多内存的任务，任务编号也相邻，从而在调度时间上也相邻。

一个warp group 共享一个tmem，和4个 tensor core，但是不是1:1，而是4:4的全乱序



# 0914

今日尝试在sm89上实现GEMM

今天重点收获：sm89 不支持 `mma.async`，然后`mma.sync`实际上，不是意味着发射命令后，一定要同步等到`mma`任务执行完成，只是意味着`warp`中的所有线程并行发起即可

# 0915

## `mbarrier`详解

`mbarrier`是放在shared memory 的，按照轮次工作的同步对象。

* 到达次数
* 当前轮次是否完成
* 等待者何时继续执行

8字节



逻辑状态

* `phase`：当前轮次 G
* `count expected`：E 下一轮需要多少次到达
* `count pending`：P 当前轮还缺多少次到达
* `signed count` ：T 当前轮尚未抵消的异步事务完成量

完成条件:

` p==0 && t == 0`

满足条件后，原子进入下一轮

```
G = G + 1
P = E
T = 0
```



？数据可能会比 arrive.expect_tx 到达？这是为什么，不可以避免吗？

好吧，经历了漫长的追问，终于搞懂了`mbarrier`的基本语义

mbarrier 本质上是一个让

* 异步硬件调用的发起者
* 异步硬件

之间进行同步的一个机制，通常前者是我们的thread，后者是cp.sync，mma，tma...

我们的目标是，

1. 所有的硬件调用都成功发起了
2. 所有的异步硬件的操作都完成了

**Hopper**

四个字段，Phase，Expected，Pending，Tx

异步硬件发起者：让Pending - 1, Tx + ，代表某个操作已经发起，并且等待异步硬件的效果完成

异步硬件完成 Tx -

**SM80x**

没有了Tx字段，怎么保证之前的语义呢？

初始化 P = 1

生产者发起：P + 1 = 2
生产者自己完成： P - 1= 1
异步硬件完成：p -  1 = 0
