霍夫直线法（straight）和滑动窗口法（curve）分别实现车道线检测

代码运行环境：Anaconda下面的spyder4.0.1

代码运行错误可根据提示，在Anaconda Prompt下安装各类库，pip install opencv-python

不出现图片输出结果，可以在spyder中设置Tools-Preferences-IPython console-Graphics-Backend-Automatic

主要思路：

gray.py 

基于Spyder4.0.1和OpenCV库，通过Python仿真进行上述不同灰度化处理方法的比较，如图1所示。

![image](http://github.com/zhaohaowu/Graduation-Project/raw/master/Images/1.jpg)

 (a)  R分量法             (b)  G分量法            (c)  B分量法
   
 (d)  最大值法            (e)  平均值法           (f)  加权平均法
