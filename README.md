# 简介
* 使用多线程异步操作rknn模型, 提高rk3588/rk3588s的NPU使用率, 进而提高推理帧数
* 使用**yolov5s**和**新宝岛**进行演示, 理论上修改代码后可适用于大部分模型与应用场景 (这里只测试过resnet, 其中resnet50推理速度约为280帧) 
* rk3568之类的应该也能借此提高NPU使用率, 但是作者本人并没有rk3568开发板......

# 更新说明
* 修复rknnpool.py中NPU调用不均的BUG
* 修改了yolov5s的测试结果, 测试数据以定频后数据为准 
* 添加了resnet18, resnet26, resnet50的测试结果


# 使用说明
### 演示
  * 将仓库拉取至开发板后运行main.py查看演示示例
  * 运行rkcat.sh可以查看当前温度与NPU占用
  * 切换至root用户运行performance.sh可以进行定频操作(约等于开启性能模式)
### 部署应用
  * 修改main.py下的modelPath为你自己的模型所在路径
  * 修改main.py下的cap为你想要运行的视频/摄像头
  * 修改main.py下的TPEs为你想要的线程数, 具体可参考下表
  * 修改func.py为你自己需要的推理函数, 具体可查看myFunc函数

# 多线程模型帧率测试
* 使用performance.sh进行CPU/NPU定频尽量减少误差
* 测试模型来源: 
* [yolov5s-silu](https://github.com/rockchip-linux/rknn-toolkit2/tree/master/examples/onnx/yolov5) 
* yolov5s-relu 某群友
* [resnet18_for_rk3588](https://github.com/rockchip-linux/rknn-toolkit2/tree/master/rknn_toolkit_lite2/examples/inference_with_lite) 
* [resnet26](https://github.com/pprp/timm) 
* [resnet50](https://github.com/pprp/timm)
* yolov5s测试视频为 [新宝岛](https://www.bilibili.com/video/BV1j4411W7F7/?spm_id_from=333.337.search-card.all.click)

|  模型\线程数   | 1  | 三核并用  |  2   | 3  |  4  | 5  | 6  | 7  | 8  | 9  |
|  ----  | ----  |  ----  | ----  |  ----  | ----  | ----  | ----  | ----  | ----  | ----  |
| Yolov5s - silu  | 13.4159 | 14.9185  | 26.0832 | 35.8814  | 38.1729 | 43.2117 | 45.5549 | 45.2401 | 45.5819 | 46.4229 |
| Yolov5s - relu  |  |  |  |  |  | 83.8213 |  |  |  |  |
| resnet50  |  |  |  | 288.9171 |  |  | 483.8374 |  |  | 577.6006 |
| resnet26  |  |  |  | 233.1631 |  |  | 394.8324 |  |  | 420.1080 |
| resnet50  |  |  |  | 186.1753 |  |  | 259.8894 |  |  | 284.4917 |

# 补充
* 请注意由于多线程下CPU, NPU占用提高, **核心温度相应也会增高**, 请做好散热措施。 这里推荐开3线程, 30帧为大多数某宝摄像头的帧数, 实测小铜片散热下运行十分钟温度约为63°
* 测试模型激活函数为silu, 此激活函数量化类型为float16, 导致推理过程中使用CPU进行计算, 量化效果较糟。 将激活函数换为relu, 可以在牺牲一点精度的情况下获得巨大性能提升, 群友测试约为**80-83帧**, c++优化后理论或许有上百? 详情可看[蓝灵风](https://www.bilibili.com/video/BV1sM4y1D7Q1/?spm_id_from=333.337.search-card.all.click)的演示视频
* 性能劣化原因猜想：
    1.  python的GIL为伪多线程, 换为c++或许在8线程前仍有较大提升
    2.  rk3588的CPU性能跟不上, 对OpenCV绘框部分做c++优化或许有提升

# Acknowledgements
* https://github.com/ultralytics/yolov5
* https://github.com/rockchip-linux/rknn-toolkit2
* https://github.com/pprp/timm