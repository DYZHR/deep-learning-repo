# 一、论文



## 1. 标题+作者

作者即深度学习花书作者。

## 2. 摘要

一般的生成模型就一个，并且任务是尽量让生成数据贴近真实数据。

GAN包括2个模型，一个生成模型G来获取整个数据的分布，一个判别模型D来判别样本来自真实数据还是生成数据。生成模型的任务是尽量让判别模型犯错。这种观点来自博弈论。

GAN不需要markov依赖链等推理，使得模型简单，并且训练效果好。

## 3. 导言

深度学习不仅是深度网络，更是对数据分布的表示。

虽然深度学习在辨别模型上做得好，但在生成模型上还比较欠缺。其难点在于我们要去最大化似然函数时，要对概率分布进行很多近似，比如VAE要把隐变量分布近似为标准整体分布。

本文提出一个框架，称为GAN。这个框架有2个模型，每个模型由MLP组成。生成模型的作用就是产生虚拟数据，类似一个造假币的团队。判别模型就是警察，区分假币和真币。在对抗的过程中，二者的能力都会提升。

## 4. 相关工作

本论文arxiv版本相关工作其实不太相关，正式论文的相关工作还好。所以李沐推测，作者在想这篇论文时，其实没有找太多相关工作，就是自己突然想到了这个创新点。

之前的生成模型，总是想构造真实数据的分布函数，然后给这个函数提供一些参数，让它可以通过最大化似然函数来学习。这种方式一方面通常有复杂的似然函数，另一方面对分布做了预设的限制。

所以本文提出一种不假设分布函数的生成方法，直接拟合生成数据，算起来比较容易。

## 5. 模型

本框架最简单的应用是生成器和判别器都是MLP。为了让生成器在数据 $x$ 上学习一个名为 $P_g$ 的分布，我们为输入噪声变量定义了一个先验分布 $p_z(z)$ ，然后将数据空间的映射表示为 $G(z; \theta) $，其中 $G$ 是一个由参数 $\theta_g$ 表示的可微函数，该函数由多层感知机组成。 $G(z)$ 表示输入数据为 $z$ .

> 即：从先验分布 $p_z(z)$ 中采样出噪音 $z$ ， $z$ 输入 $\theta_g$ 表示的可微函数 $G$ ，得到生成数据 $x'$ ， $x'$ 的分布 $P_g$ 近似真实数据 $x$ 的分布。

同样定义第二个多层感知机 $D(x; \theta_d)$ ，输出一个单值标量。 $D(x)$ 表示 $x$ 为真实数据而非 $P_g$ 的概率。

$D$ 需要式(1)表示的值最**大**化：
$$
\max V(D) = \mathbb E_{x\sim p_{data}(x)}[\log D(x)] + \mathbb E_{z\sim p_z(z)}[\log (1-D(G(z)))]\tag{1}
$$
$G$ 需要式(2)表示的值最**小**化，或式(3)表示的值最**大**化，式(3)在训练初期能为 $G$ 提供更强的梯度：
$$
\min V(G) = \mathbb E_{z\sim p_z(z)}[\log (1-D(G(z)))]\tag{2}
$$

$$
\max V(G) = \mathbb E_{z\sim p_z(z)}[\log D(G(z))]\tag{3}
$$



整合起来就是：
$$
\underset{G}{\min}\underset{D}{\max} V(D, G) = \mathbb E_{x\sim p_{data}(x)}[\log D(x)] + \mathbb E_{z\sim p_z(z)}[\log (1-D(G(z)))]\tag{4}
$$

> 李沐以游戏举了个例子，比如我们要生成游戏的图片，最直接的办法就是反汇编游戏代码，得到真实代码那我们就知道这个游戏怎样生成的，这种方式代表寻找真实数据分布的一类生成模型。这种方式比较复杂。
>
> 另一种方法就是我们干脆就不管这个游戏程序怎样来的。虽然你这是个4K分辨率的游戏，有很多像素点，但背后就是一些代码控制，可能就是100个变量控制的，哪个人物出现在哪个位置，在干什么事。虽然我们不知道程序代码，但我估计就用个100维向量就能表示控制游戏图像的这些变量。然后用这100维向量通过MLP映射为图片，使它和真实游戏图片很像。这种方式比较简单，但缺点是难以知道生成数据的根源。

## 6. 理论

$D$ 和 $G$ 持续对抗，最终达到平衡，双方都不能更进一步，这在博弈论里被称为**纳什均衡**，这个节点就是 $p_g = p_{data}$ 。

![image-20250924234918870](./../../assets/image-20250924234918870.png)

图片的意思是：噪声 $z$ 取自一个均匀分布， $z$ 经过 $G$ 处理映射到 $x$ ，为绿色的线，大致是一个相对真实数据分布的均值偏大方差偏大的正态分布。真实分布 $x$ 是黑色的线，也是一个正态分布。判别器 $D$ 的输出值∈[0, 1]，较高值表示判别器认为输入数据倾向于真实数据，较低值表示倾向于生成数据。

刚开始真实数据在数据值较小的位置概率大，$D$ 也在数据值较小的位置倾向于认为输入数据为真实数据。随着训练，判别器减少振动，$G$ 输出的数据的数据分布 $p_g$ 也逐渐接近真实分布 $p_{data}$ 。到最后 $D$ 和 $G$ 收敛，$p_g$ 会接近 $p_{data}$ ，$D$ 也随之在所有位置都基本输出同一值(0.5)，即 $D$ 无法分辨输入数据是真实还是生成的。

**算法见原文。**

为什么先更新判别器再更新生成器？

> 判别器的更新依据，即真假样本标签，是先天就有的，所以判别器能直接开始更新。而生成器的更新依据是判别器的生成结果，需要有一定准确度的判别器，所以判别器先训练。

为什么判别器更新K次而生成器更新1次？

> 如果K值过小，判别器不能充分优化，判断力弱，那么梯度更新时 $D(G(z))$ 就会振荡，如图1第一张图，造成训练不稳定，生成样本质量差。
>
> 如果K值过大，判别器过于强大，那么生成器的目标函数就固定在最小值， $D(G(z))$ -> 0，梯度消失，无法学习。

> 回到李沐举的警察和假币商的例子。如果警察太差，抓人没有判断依据，那么假币商就没法找到规律，不能改进造假币技术。如果警察太强，假币商赚不到钱，一锅端走了，那假币商也没法改进技术了。

目标函数有个全局最优解，当且仅当 $p_g = p_{data}$ 时取得。这在数据科学上是一个常用的东西：如果要分别2个分布是否相同，就训练一个二分类分类器，如果这个分类器能分开这俩数据，就表示这2个分布不同，否则相同。

## 7. 评论

首先intro非常短，就是写一点故事性的东西，为什么做这个事情。

接下来就是写GAN在干什么。

在相关工作里，还是写了很多工作别人已经做过了。真正伟大的工作，不在乎别人是否已经做过，关键是你能给大家展示，你这个东西在这个应用上能取得很好的效果，让别人信服你，能跟着你继续做。

第三章介绍GAN的目标函数，以及怎样优化。

第四章讲证明，为什么目标函数能得到最优解，以及求解算法怎样得到最优解。

最后一章简单讲点实验，以及future work。

这样读下来就很舒服，假设你觉得你的工作在未来几年会被人反复读，那就这么写。

如果你的工作开创性不太高，那就一定要写清楚你跟别人的区别是什么，你的贡献是什么。

同时，GAN用一个监督学习的损失函数来做无监督学习，所以在训练上



# 二、代码



```python
import os
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image



# -------------------------- 2. 数据加载与预处理 --------------------------
# 数据预处理：转为Tensor + 归一化到[0,1]（MNIST原始像素值为0-255）
def flatten_img(x):
    return x.view(-1)



# -------------------------- 3. 定义GAN的核心网络（生成器+判别器） --------------------------
class Generator(nn.Module):
    """生成器G：输入100维隐变量z，输出784维图像向量（对应28x28灰度图）"""

    def __init__(self):
        super(Generator, self).__init__()
        # 全连接层，逐步将低位z映射到高维图像向量x
        self.model = nn.Sequential(
            # z(b, 100) -> (b, 256)
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),

            # (b, 256) -> (b, 512)
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),

            # (b, 512) -> (b, 1024)
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),

            # (b, 1024) -> (b, 784)
            nn.Linear(1024, 784),
            nn.Sigmoid()  # 归一化到(0, 1)，匹配MNIST像素范围
        )

    def forward(self, z):
        """前向传播，输入z -> 生成图像x"""
        img = self.model(z)
        return img


class Discriminator(nn.Module):
    """判别器D：输入784维图像向量，输出1维概率（0=假样本，1=真样本）"""

    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # (b, 784) -> (b, 1024)
            nn.Linear(img_size * img_size, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),  # 防止D过拟合，D太强导致G梯度消失

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, 1),
            nn.Sigmoid()  # 输出(0, 1)的概率
        )

    def forward(self, img):
        """前向传播：输入图像 -> 输出图像是真实样本的概率"""
        prob = self.model(img)
        return prob




# -------------------------- 5. 定义训练流程 --------------------------
def train_gan():
    # 固定生成器的输入z（用于每轮训练后生成样本，观察效果变化）
    fixed_z = torch.randn(32, latent_dim).to(device)  # 生成32个样本(2^5)

    for epoch in range(epochs):
        # 记录每轮的损失
        loss_D_total = 0.0  # 判别器总损失
        loss_G_total = 0.0  # 生成器总损失

        # 遍历训练集（批量训练）
        for batch_idx, (real_imgs, _) in enumerate(train_loader):
            # 准备数据：真是图像+标签
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            real_labels = torch.ones(batch_size, 1).to(device)  # 真实样本，标签1
            fake_labels = torch.zeros(batch_size, 1).to(device)  # 生成样本，标签0

            # -------------------------- 阶段1：训练判别器D（目标：最大化区分真假的能力） --------------------------
            for _ in range(K):
                D.train()  # 开启D的训练模式（启用Dropout）
                optimizer_D.zero_grad()  # 清零D的梯度

                # 1.1 训练D识别真实样本
                prob_real = D(real_imgs)
                loss_D_real = criterion(prob_real, real_labels)

                # 1.2 训练D识别生成样本
                z = torch.randn(batch_size, latent_dim).to(device)
                fake_imgs = G(z)
                prob_fake = D(fake_imgs.detach())  # 用detach()阻止梯度流向G（仅训练D）
                loss_D_fake = criterion(prob_fake, fake_labels)

                # 1.3 更新D
                loss_D = loss_D_real + loss_D_fake
                loss_D.backward()
                optimizer_D.step()

                # 1.4 计算总损失
                loss_D_total += loss_D.item() * batch_size  # 累积D的损失（注意：K次训练会累积K次损失

            # -------------------------- 阶段2：训练生成器G（目标：最小化被D识破的概率） --------------------------
            G.train()  # 开启G的训练模式（启用batchnorm）
            optimizer_G.zero_grad()

            # 2.1 生成假图像让G预测，希望D(fake_imgs) -> 1
            # 注意：此处不再detach()，梯度需流向G
            prob_fake_G = D(fake_imgs)
            loss_G = criterion(prob_fake_G, real_labels)

            # 2.2 更新G
            loss_G.backward()
            optimizer_G.step()

            # 2.3 计算总损失
            loss_G_total += loss_G.item() * batch_size

        # -------------------------- 每轮训练后：计算平均损失 + 生成样本 --------------------------
        # 计算每个epoch的平均损失（除以总样本数）
        avg_loss_D = loss_D_total / len(train_dataset)
        avg_loss_G = loss_G_total / len(train_dataset)

        # 打印训练日志
        print(f"Epoch [{epoch + 1}/{epochs}] | Loss_D: {avg_loss_D:.4f} | Loss_G: {avg_loss_G:.4f}")

        # 生成并保存样本，用固定的z，观察G的生成效果变化
        G.eval()  # 切换到评估模式，关闭batchnorm更新
        with torch.no_grad():  # 禁用梯度计算，加速推理
            fixed_fake_imgs = G(fixed_z)  # 生成32个假图像
            # 调整图像形状，(32, 784) -> (32, 28, 28)，适合torchvision保存
            fixed_fake_imgs = fixed_fake_imgs.view(-1, 1, img_size, img_size)
            # 保存图像，nrow=8，每行8个样本，共4行
            save_image(fixed_fake_imgs,
                       f"gan_mnist_results/epoch_{epoch + 1}.png",
                       nrow=8,
                       normalize=True)  # 确保像素值正确映射到[0, 1]


# -------------------------- 6. 启动训练 + 可视化结果 --------------------------
if __name__ == "__main__":
    # -------------------------- 1. 配置基础参数与设备 --------------------------
    # 随机种子（保证结果可复现）
    torch.manual_seed(666)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(666)

    # 设备配置（优先使用GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 超参数
    latent_dim = 100  # 隐变量z的维度（随机噪声的长度）
    img_size = 28  # MNIST图像尺寸（28x28）
    channels = 1  # 图像通道数（MNIST为灰度图，通道数=1）
    batch_size = 128  # 批次大小
    epochs = 50  # 训练轮次
    lr = 2e-4  # 学习率（GAN对学习率敏感，通常用1e-4~2e-4）
    beta1 = 0.5  # Adam优化器的beta1参数（GAN常用0.5，提升稳定性）
    K = 1  # 每个epoch，D的训练轮次

    # 创建结果保存目录（存储生成的图像）
    os.makedirs("gan_mnist_results", exist_ok=True)


    transform = transforms.Compose([
        transforms.ToTensor(),  # (H, W, C) -> (C, H, W)，且像素值归一化到[0, 1]
        flatten_img  # 展平为向量：28×28 -> 784维
    ])

    # 加载MNIST训练集（仅用训练集训练GAN，无需测试集）
    train_dataset = datasets.MNIST(
        root="./data",  # 数据保存路径
        train=True,  # 加载训练集
        transform=transform,  # 应用预处理
        download=True  # 若本地无数据，自动下载
    )

    # 数据加载器（批量加载+打乱）
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    # -------------------------- 4. 初始化网络、损失函数与优化器 --------------------------
    # 初始化生成器和判别器，并移动到设备
    G = Generator().to(device)
    D = Discriminator().to(device)

    # 损失函数：二分类交叉熵
    criterion = nn.BCELoss()

    # 优化器：Adam（GAN的主流优化器，beta1=0.5是经验值）
    # 注意：G和D的优化器需分开定义，确保参数独立更新
    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))

    # 启动GAN训练
    train_gan()

    # 训练结束后，可视化最后一轮生成的样本
    # 读取最后一轮的生成图像
    final_img_path = f"gan_mnist_results/epoch_{epochs}.png"
    final_img = plt.imread(final_img_path)

    # 绘制图像
    plt.figure(figsize=(10, 5))
    plt.imshow(final_img, cmap="gray")
    plt.title(f"GAN Generated MNIST Images (After {epochs} Epochs)")
    plt.axis("off")  # 隐藏坐标轴
    plt.show()

    # 保存训练好的生成器，可用于后续推理
    torch.save(G.state_dict(), "gan_mnist_generator.pth")
    print("生成器模型已保存为: gan_mnist_generator.pth")
```



# 三、参考

[GAN论文逐段精读【论文精读】_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1rb4y187vD/?spm_id_from=333.1387.search.video_card.click&vd_source=6d7dc2c06bc4259ac0e431a2824dbf9d)

