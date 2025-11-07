import torch
import torch.nn.functional as F

class DenoiseDiffusion:
    def __init__(self, num_time_step=1000):
        self.num_time_step = num_time_step
        self.beta = torch.linspace(0.001, 0.02, num_time_step) # linear beta, less noise begin easier to learn, more noise after to approch gaussian

        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha) # [alpha_0, alpha_0*alpha_1, ...]

        self.alpha_bar_prev = torch.cat([torch.tensor([1.0]), self.alpha_bar[:-1]])

    # 正向加噪 q(x_t | x_0)
    def q_sample(self, x0, t, noise):
        alpha_bar_t = self.alpha_bar[t]
        x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1-alpha_bar_t)*noise
        return x_t
    
    # 反向一步 p(x_{t-1} | x_t)
    def p_sample(self, model, x_t, t):
        # 预测噪声 ε_θ(x_t, t)
        noise = model(x_t, t)

        coef = (1 - self.alpha[t]) / torch.sqrt(1 - self.alpha_bar[t])
        mean = 1 / torch.sqrt(self.alpha[t]) * (x_t - coef * noise)

        var = (1-self.alpha[t])*(1-self.alpha_bar_prev[t]) / (1-self.alpha_bar[t])

        eps = torch.randn(x_t.shape) # 额外随机噪声
        return mean + torch.sqrt(var) * eps

    def loss(self, model, x0):
        t = torch.randint(0, self.num_time_step)
        noise = torch.randn_like(x0)

        x_t = self.q_sample(x0, t, noise)
        pred = model(x_t, t)

        return F.mse_loss(pred, noise)

    # 完整去噪：x_T ~ N(0, I) -> ... -> x_0
    def sample(self, model, x_T):
        x = x_T
        # 逐步去噪：T-1, T-2, ..., 1
        for t in range(self.num_time_step, 0, -1):
            x = self.p_sample(model, x, t)

        return x

if __name__ == "__main__":
    diffusion = DenoiseDiffusion()
