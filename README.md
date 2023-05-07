# 简介
* 使用多线程异步操作rknn模型, 提高rk3588/rk3588s的NPU使用率, 进而提高推理帧数(rk3568之类修改后应该也能使用, 但是作者本人并没有rk3568开发板......)
* 此分支使用模型[yolov5s_relu_tk2_RK3588_i8.rknn](https://github.com/airockchip/rknn_model_zoo), 将yolov5s模型的激活函数silu修改为为relu,在损失一点精度的情况下获得较大性能提升,详情见于[rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo/tree/main/models/CV/object_detection/yolo)
* 此项目的[c++](https://github.com/leafqycc/rknn-cpp-Multithreading)实现

# 更新说明
* 无


# 使用说明
### 演示
  * 将仓库拉取至本地, 并将Releases中的演示视频720p60hz.mp4放于video目录下后运行main.py查看演示示例
  * 运行rkcat.sh可以查看当前温度与NPU占用
  * 切换至root用户运行performance.sh可以进行定频操作(约等于开启性能模式)
### 部署应用
  * 修改main.py下的modelPath为你自己的模型所在路径
  * 修改main.py下的cap为你想要运行的视频/摄像头
  * 修改main.py下的TPEs为你想要的线程数, 具体可参考下表
  * 修改func.py为你自己需要的推理函数, 具体可查看myFunc函数

# 多线程模型帧率测试
* 使用performance.sh进行CPU/NPU定频尽量减少误差
* 测试模型为[yolov5s_relu_tk2_RK3588_i8.rknn](https://github.com/airockchip/rknn_model_zoo)
* 测试视频见于Releases

|  模型\线程数   | 1    |  2   | 3  |  4  | 5  | 6  |
|  ----  | ----    | ----  |  ----  | ----  | ----  | ----  |
| yolov5s  | 27.4491 | 49.0747 | 65.3673  | 63.3204 | 71.8407 | 72.0590 |

# 补充
* 多线程下CPU, NPU占用较高, **核心温度相应增高**, 请做好散热。推荐开1, 2, 3线程, 实测小铜片散热下运行三分钟温度约为56°, 64°, 69°

# Acknowledgements
* https://github.com/ultralytics/yolov5
* https://github.com/rockchip-linux/rknn-toolkit2
* https://github.com/airockchip/rknn_model_zoo