从0编写的llm强化学习全流程训练代码，用于：
1. 梳理从embedding开始的llm/transformer整体概念
2. 验证modelsight下的训推平台的全流程

training目录下是全量训练代码，支持cpu/gpu/mps（apple）三种不同的训练后端，通过training-cli调用。

training dashboard可以查看训练进度和指标。

最终通过游戏自对战，会训练出一个精通打升级的llm。
