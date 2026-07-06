import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    def __init__(self, out_dim=4096, n_crops=6,
                 t_student=0.1, t_teacher=0.04, center_momentum=0.9): #teacher with bvery low temperature is confident
        super().__init__()
        self.t_s, self.t_t = t_student, t_teacher
        self.n_crops = n_crops
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_out, teacher_out):
        """student_out: (n_crops*B, K)  teacher_out: (2*B, K)"""
        s = (student_out / self.t_s).chunk(self.n_crops)
        t = F.softmax((teacher_out - self.center) / self.t_t, dim=-1)
        t = t.detach().chunk(2)

        loss, n_terms = 0.0, 0
        for ti, t_probs in enumerate(t):
            for si, s_logits in enumerate(s):
                if si == ti:
                    continue   # never compare a view with itself
                loss += (-t_probs * F.log_softmax(s_logits, -1)).sum(-1).mean()
                n_terms += 1
        self.update_center(teacher_out)
        return loss / n_terms

    @torch.no_grad()
    def update_center(self, teacher_out):
        batch_mean = teacher_out.mean(dim=0, keepdim=True)
        self.center = (self.center * self.center_momentum
                       + batch_mean * (1 - self.center_momentum))