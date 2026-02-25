优化算法就是调整模型参数使损失函数尽可能地小。总公式：
$$
\theta_{t+1} = \theta_{t} - \alpha \cdot \frac{\partial L(\theta_t)}{\partial \theta}
$$

* $\theta$ ：模型参数，如权重W和偏置b。
* $\alpha$ ：学习率，通常取1e-3，步长。
* $\bigtriangledown L(\theta)$ ：损失函数对模型参数的梯度，降低损失的方向。

所有优化算法都是对这个基础公式的改进，解决 “走得慢、走不稳、走偏了” 的问题。



1. **普通GD（gradient descend）**
   $$
   \theta_{t+1} = \theta_t - \alpha \cdot \sum_{i\in全体样本} \frac{\partial L_i({\theta_t})}{\partial \theta}
   $$

   * 缺点：每次更新参数都要计算全体样本，效率低。内存占用太大，要一次导入全体样本。

2. **SGD（Stochastic Gradient Descend）**
   $$
   \theta_{t+1} = \theta_t - \alpha \cdot \sum_{i\in当前batch} \frac{\partial L_i({\theta_t})}{\partial \theta}
   $$

   * 梯度震荡剧烈（单个样本噪声大），收敛慢，容易卡在局部最优点。

3. **Momentum**
   $$
   m_{t+1} = \beta \cdot m_{t} + (1 - \beta) \cdot \sum_{i\in当前batch}\frac{\partial L_i(\theta_t)}{\partial \theta}
   \\
   \theta_{t+1} = \theta_t - \alpha \cdot m_{t+1}
   $$
   其中，$\beta \in [0, 1)$，必须小于1，因为要控制越久远时刻梯度对当前时刻的影响越小。

   模拟物理动量，不仅收当前梯度的影响，还收之前梯度（惯性）的影响。

   * 优点：减少震荡，加速收敛。可以冲过小的局部最优点。
   * 缺点：固定学习率，对不同参数无差别对待。

4. **AdaGred（Adaptive Gredient）**
   $$
   m_{t+1} = \frac {\partial L(\theta_t)}{\partial \theta}
   \\
   v_{t+1} = (\frac {\partial L(\theta_t)}{\partial \theta})^2
   \\
   \theta_{t+1} = \theta_t - \alpha \cdot \frac{m_{t+1}}{\sqrt{v_{t+1}} + \epsilon}
   $$
   其中， $\epsilon$ 表示一个极小常数，通常是1e-8，防止除0异常。

   GD和Momentum，下降的速度受梯度绝对值影响，梯度绝对值越大，损失函数的值倾向于移动越长的一段距离，而此时，可能只需移动一点距离就能到达最优点。

   * 优点：AdaGred使用归一化梯度，消除了梯度幅度的影响，梯度只控制移动方向，移动距离只由学习率 $\alpha$ 控制。
   * 缺点：只有损失函数值正好落在最小值时才能收敛，否则会在最小值周围不停震荡。

5. **Adam（Adaptive Momentum）**
   $$
   m_{t+1} = \beta \cdot m_t + (1-\beta) \cdot \frac{\partial L(\theta_t)}{\partial \theta}
   \\
   v_{t+1} = \gamma \cdot v_t + (1-\gamma) \cdot (\frac{\partial L(\theta_t)}{\partial \theta})^2
   \\
   \theta_{t+1} = \theta_t - \alpha \cdot \frac{m_{t+1}}{\sqrt{v_{t+1}} + \epsilon}
   $$
   其中，$\beta$ 通常取0.9， $\gamma$ 通常取0.999。$m_{t+1}$ 表示一阶矩的滑动平均，$v_{t+1}$ 表示二阶矩的滑动平均。

   由于在训练早期，没有历史积累值，所以用以下规则放大一阶矩和二阶矩：
   $$
   \hat m_{t+1} = \frac{m_{t+1}}{1-\beta^{t+1}}
   \\
   \hat v_{t+1} = \frac{v_{t+1}}{1-\gamma^{t+1}}
   $$
   带 $t+1$ 幂次项的系数会随着时间步数的增大而逐渐趋于0，修正效果逐渐减小。

   Adam结合了AdaGred和Momentum。代码参考：

   ```python
   m = torch.zeros_like(params)
   v = torch,zeros_like(params)
   t = 0
   
   for step in range(nu_steps):
       t += 1
       g = compute_gradient(params)
       
       m = beta1 * m + (1 - beta1) * g
       v = beta2 * v + (1 - beta2) * (g**2)
       
       m_hat = m / (1 - beta1**t)
       v_hat = v / (1 - beta2**t)
   
       params = params - lr * m_hat / (torch.sqrt(v_hat) + eps)
   ```

   

6. **AdamW（Adam Weight Decay），大模型的默认优化器**

   前置知识，权重衰减（weight decay），对原来的损失函数加上L2正则化：
   $$
   \hat L = L + \lambda \sum \theta ^2
   $$
   因为模型参数绝对值在优化过程中可能会变得很大，容易造成过拟合。所以增加权重衰减项，当参数变化时，损失成平方项的增大，限制参数绝对值的增大。

   在Adam里，L2惩罚项会直接加到原来的损失函数里，参与计算滑动平均和修正。这会导致不同的参数有不同的惩罚效果，违背了L2惩罚项用于权重衰减的设计初衷——所有参数的惩罚效果一致。

   而AdamW则是把权重衰减和梯度更新解耦，让所有参数惩罚效果一致：
   $$
   \theta_{t+1} = \theta_t - \alpha \cdot \frac{m_{t+1}}{\sqrt{v_{t+1}} + \epsilon} - \alpha \cdot \lambda \cdot \theta_t
   $$
   

   

# 