# Sky Lab 代码仓库规范

这份文档写给要把自己项目放进 `sky-lab-uw` 组织的同学看。目的很简单:别人(包括半年后的你自己)拿到这个仓库,不用问你就能跑起来、看懂在干什么。

不需要写得多花哨,但下面几条是硬性要求,建组织的时候会照这个检查。

## 一、仓库里必须有什么

```
your-project/
├── README.md              必须,见下面的写法
├── requirements.txt        或 environment.yml / pyproject.toml,总之依赖要能装
├── configs/                参数、标定文件、超参配置
├── scripts/                跑起来用的脚本(train.sh, run_calib.py ...)
├── src/  或  <project>/     代码本体
├── data/                    只放小样例或 README 说明去哪下载数据,别传数据集
├── checkpoints/ 或 outputs/ 同上,只放 .gitignore,不传权重和日志
├── .gitignore
└── LICENSE                  没有的话默认按组内 MIT,问导师
```

不强制这个目录名一定叫 `src`,你原来的项目结构能看懂就行,但"代码、脚本、配置"这几类东西要能一眼分清楚,不能全堆在根目录一坨 `.py` 文件里。

## 二、README 写什么

README 只回答四件事:这是什么、怎么装、怎么跑、跑出来该长什么样。按这个顺序写,不用写成论文。

模板:

```markdown
# 项目名

一句话说这是干什么的,给谁用的。

维护人: @你的github用户名
状态: active / archived / 交接中

## 这是什么

2-4 句话说清楚背景和这个仓库解决的问题。如果依赖其他仓库(比如标定结果要喂给 vla 那个仓),在这里说清楚上下游关系。

## 环境

- Python 3.10 / ROS2 Humble / CUDA 12.1 (按实际写,不要写"最新版本"这种)
- 依赖: `pip install -r requirements.txt`
- 硬件依赖(如果有): 哪个相机型号、哪个机械臂,固件/驱动版本

## 怎么跑

给能直接复制粘贴执行的命令,不要写"运行主程序即可"这种。

    python scripts/run_calib.py --config configs/d435i.yaml

跑完应该看到什么(输出文件、终端打印、图),截图或贴一段示例输出。

## 数据 / 权重

数据集/预训练权重放在哪(NAS 路径、下载链接),不要指望别人问你要。

## 已知问题 / TODO

写实话。有坑就说是什么坑,免得下一个人重踩一遍。
```

举个例子,相机标定的仓库大概长这样:

```markdown
# camera-calibration

RealSense D435i 与机械臂末端的手眼标定流程。

维护人: @boyue
状态: active

## 这是什么

用于标定相机坐标系到机械臂末端坐标系的变换矩阵,标定结果给 arm-setup 仓库里的抓取流程用。

## 环境

- Python 3.10, OpenCV 4.9
- RealSense SDK (pyrealsense2==2.55.1)
- pip install -r requirements.txt

## 怎么跑

1. 采集标定板图像:
       python scripts/capture.py --output data/calib_0930/
2. 计算标定结果:
       python scripts/run_calib.py --input data/calib_0930/ --output configs/extrinsics.yaml
   跑完会在 configs/ 下生成 extrinsics.yaml,里面是 4x4 变换矩阵。

## 数据 / 权重

标定板图像不上传,示例数据在组内 NAS: /data/sky-lab/camera-calib-samples/

## 已知问题

- D435i 在近距离(<0.3m)畸变较大,标定板别放太近
- 目前只支持棋盘格标定板,ChArUco 还没接
```

这个例子不用照抄,但风格照抄:具体命令、具体路径、具体版本号,别写模糊的话。

## 三、命名和分支

- repo 名:`方向-内容`,小写加短横线,比如 `camera-calibration`、`vla-finetune`,不要把人名放进 repo 名。
- `main` 分支:受保护,只能通过 PR 合入,保证随时能跑。
- 自己开发用 `dev/你的名字` 或 `feature/具体描述`,比如 `dev/boyue`、`feature/charuco-support`。
- 阶段性成果打 tag,比如 `v0.1`、投论文对应的版本打 `icra26-submission`,别把版本号写进分支名(`camera/boyue/v1` 这种不要)。

## 四、绝对不能提交的东西

- 数据集、模型权重、rosbag、日志、图片压缩包 —— 这些体积大,进了 git 历史删不干净,统一走 NAS / HuggingFace / 对象存储,仓库里只放路径或下载脚本。
- API key、内网 IP、账号密码、实验室 wifi 密码 —— 写进 `.env`,加进 `.gitignore`,不要写死在代码里。
- 大文件如果确实要进 git,用 Git LFS,先问一下额度够不够。

提交前养成习惯看一眼 `git status` 和 `git diff --stat`,别一把梭 `git add .`。

## 五、检查清单

发 PR 或者第一次把项目挪进组织仓库前,自己过一遍:

- [ ] 别人 clone 下来,跟着 README 能装上环境
- [ ] 跟着 README 的命令能跑出结果,不用来问你
- [ ] 没有数据集、权重、密钥被提交进去(翻一下 git log,历史里有的话要清)
- [ ] requirements.txt / environment.yml 是最新的,不是半年前的
- [ ] README 顶部写了维护人和状态

就这些,不复杂,核心就是"README 让人不用问你就能跑起来"。有问题在组会上提。
