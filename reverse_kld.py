# TODO: TRY Kld WITHH SAVED SPARSE LOGITS

import torch
import torch.nn.functional as F



def reverse_kld(student_logits, teacher_logits):
    log_ps = F.log_softmax(student_logits, dim=-1)
    ps     = log_ps.exp()                       # p_s

    with torch.no_grad():
        log_pt = F.log_softmax(teacher_logits, dim=-1)   # log p_t

    loss_kd = (ps * (log_ps - log_pt)).sum(-1).mean()    
    return loss_kd


if __name__ == "__main__":
    

    student_logits = torch.log(torch.tensor([[[10, 10, 10]], [[20, 10, 20]]])) 
    teacher_logits = torch.log(torch.tensor([[[50, 5, 1]], [[5, 35, 5]]]))
    print(F.kl_div(F.log_softmax(student_logits, dim=-1), F.softmax(teacher_logits, dim=-1), reduction="batchmean"))
    print(reverse_kld(student_logits, teacher_logits)) 

