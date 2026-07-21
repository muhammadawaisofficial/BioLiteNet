import torch
import torch.nn as nn
import torch.nn.functional as F


class HazardAwareFocalLoss(nn.Module):
    # HAFL formula: L = -h_c * alpha_t * (1 - p_t)^gamma * log(p_t)
    # h_c is the hazard weight for each class (based on WHO guidelines)
    def __init__(self, hazard_weights, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma     = gamma
        self.reduction = reduction

        h = torch.tensor(hazard_weights, dtype=torch.float32)
        self.register_buffer("hazard_weights", h)

        if alpha is not None:
            a = torch.tensor(alpha, dtype=torch.float32)
        else:
            a = torch.ones(len(hazard_weights), dtype=torch.float32)
        self.register_buffer("alpha", a)

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs     = torch.exp(log_probs)

        p_t     = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        log_p_t = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        alpha_t = self.alpha[targets]
        h_c     = self.hazard_weights[targets]

        focal_weight = (1.0 - p_t) ** self.gamma
        loss = -h_c * alpha_t * focal_weight * log_p_t

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_hafl(class_names, config_hazard_weights, gamma=2.0):
    # builds HAFL with h_c values matched to dataloader class order
    h_values = []
    print("\n[HAFL] Hazard weight assignment:")
    print(f"  {'Idx':<4} {'Class':<40} {'h_c':>5}")
    print("  " + "-" * 52)
    for idx, cls in enumerate(class_names):
        hc = config_hazard_weights.get(cls, 1.0)
        h_values.append(hc)
        print(f"  {idx:<4} {cls:<40} {hc:>5.1f}")
    return HazardAwareFocalLoss(hazard_weights=h_values, gamma=gamma)


class FocalLoss(nn.Module):
    # standard focal loss used in ablation variants without HAFL
    def __init__(self, num_classes, gamma=2.0):
        super().__init__()
        weights   = [1.0] * num_classes
        self.hafl = HazardAwareFocalLoss(weights, gamma=gamma)

    def forward(self, logits, targets):
        return self.hafl(logits, targets)


if __name__ == "__main__":
    from config import HAZARD_WEIGHTS, FOCAL_GAMMA
    class_names = ["used_syringes", "human_organs", "expired_tablets",
                   "used_medical_gloves", "waterbottles"]
    criterion = build_hafl(class_names, HAZARD_WEIGHTS, gamma=FOCAL_GAMMA)
    logits  = torch.randn(4, 5)
    targets = torch.tensor([0, 2, 4, 1])
    loss    = criterion(logits, targets)
    print(f"HAFL loss: {loss.item():.4f}")
