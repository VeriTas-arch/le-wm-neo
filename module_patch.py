# 替换你原有的 Monkey-patch 脚本
import torch

_orig_forward = None

def apply_patch():
    from module import ARPredictor
    global _orig_forward
    _orig_forward = ARPredictor.forward
    
    def fast_forward(self, x, c):
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        
        # 1. 经过 GRU (或者如果你在 init 里改成了 LSTM，这里也通用)
        gru_out, _ = self.gru(x)
        # 残差连接
        x_residual = self.gru_norm(x + gru_out)
        
        # 2. 扩大 Transformer 窗口，从 5 扩大到 15！
        WINDOW = 15 
        if T > WINDOW:
            x_window = x_residual[:, -WINDOW:]  # (B, W, D)
            c_window = c[:, -WINDOW:] if c.size(1) > WINDOW else c[:, -T:]
            x_out = self.transformer(x_window, c_window)
            
            # 把 Transformer 的结果拼回去
            padded = torch.zeros_like(x_residual)
            padded[:, :-WINDOW] = x_residual[:, :-WINDOW] # 前面保留 GRU 结果
            padded[:, -WINDOW:] = x_out # 最后 15 帧用 Transformer 结果
            
            # ✨ 核心修复点：把纯粹的 GRU 输出强行绑定到返回值上，让 train_fix.py 能拿到！
            padded.return_gru = gru_out 
            return padded
        else:
            out = self.transformer(x_residual, c)
            out.return_gru = gru_out # 同样绑定
            return out
    
    ARPredictor.forward = fast_forward