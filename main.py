import numpy as np
 
class OnlineGradientDescent:
    def __init__(self, dim, lr=0.1, proj_radius=1.0):
        self.w=np.zeros(dim); self.lr=lr; self.R=proj_radius
        self.regret_hist=[]; self.loss_hist=[]
    def project(self, w):
        n=np.linalg.norm(w); return w if n<=self.R else w/n*self.R
    def predict(self, x): return self.w@x
    def update(self, x, y):
        loss=(self.predict(x)-y)**2
         grad=2*(self.predict(x)-y)*x
        self.w=self.project(self.w-self.lr*grad)
        self.loss_hist.append(loss); return loss
 
class ExponentiatedGradient:
    def __init__(self, n_experts, lr=0.1):
        self.w=np.ones(n_experts)/n_experts; self.lr=lr
    def predict(self, expert_preds): return self.w@expert_preds
    def update(self, expert_preds, true_val):
        losses=(expert_preds-true_val)**2
        self.w*=np.exp(-self.lr*losses); self.w/=self.w.sum()
 
T=1000; dim=5; np.random.seed(42)
ogd=OnlineGradientDescent(dim, lr=0.01)
best_w=np.random.randn(dim); best_w/=np.linalg.norm(best_w)
total_loss=0; best_loss=0
for t in range(T):
    x=np.random.randn(dim); y=best_w@x+np.random.randn()*0.1
    total_loss+=ogd.update(x,y); best_loss+=(best_w@x-y)**2
print(f"OGD cumulative loss: {total_loss:.2f}")
print(f"Best in hindsight:   {best_loss:.2f}")
print(f"Regret: {total_loss-best_loss:.2f} (theoretical O(sqrt(T))={np.sqrt(T):.1f})")
