# 简介
* 使用多线程异步操作rknn模型, 提高rk3588/rk3588s的NPU使用率, 进而提高推理帧数
* 使用**yolov5s**和**新宝岛**进行演示, 理论上修改代码后可适用于大部分模型与应用场景 (这里只测试过resnet, 其中resnet50推理速度约为280帧) 
* rk3568之类的应该也能借此提高NPU使用率, 但是作者本人并没有rk3568开发板......

# 使用说明
### 演示
  * 将仓库拉取至开发板后运行main.py查看演示示例
### 部署应用
  * 修改main.py下的modelPath为你自己的模型所在路径
  * 修改main.py下的cap为你想要运行的视频/摄像头
  * 修改main.py下的TPEs为你想要的线程数, 具体可以看下文
  * 修改func.py为你自己需要的推理函数, 具体可以查看myFunc函数

# 多线程模型帧率测试
* 测试模型来源 [yolov5s](https://github.com/rockchip-linux/rknn-toolkit2/tree/master/examples/onnx/yolov5)
* yolov5s测试视频为 [新宝岛](https://www.bilibili.com/video/BV1j4411W7F7/?spm_id_from=333.337.search-card.all.click)

|  模型\线程数   | 1  |  2   | 3  |  4   | 5  | 6  |
|  ----  | ----  |  ----  | ----  |  ----  | ----  | ----  |
| Yolov5s - silu  | 10.4537 | 21.3908  | 31.5161 | 40.0544  | 42.0195 | 43.7535 |
| resnet26  |  |   |  |   |  |  |
| resnet50  |  |   |  |   |  |  |

# 补充
* 测试模型激活函数为silu, 此激活函数量化类型为float16, 导致推理过程中使用CPU进行计算, 量化效果较糟。 将激活函数换为relu, 可以在牺牲一点精度的情况下获得巨大性能提升, 群友测试约为**75 - 80帧**, 理论或许有上百? 详情可看[蓝灵风](https://www.bilibili.com/video/BV1sM4y1D7Q1/?spm_id_from=333.337.search-card.all.click)大佬的演示视频
* 性能劣化原因猜想：
    1.  python的GIL为伪多线程, 换为c++或许在8线程前仍有较大提升
    2.  rk3588的CPU性能跟不上, 对OpenCV绘框部分做c++优化或许有提升

# Acknowledgements
https://github.com/ultralytics/yolov5
https://github.com/rockchip-linux/rknn-toolkit2