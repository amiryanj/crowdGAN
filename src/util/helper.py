def bce_loss(input, target):
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()

def l1_loss(input, target):
    loss = (input - target).abs()
    return loss.mean()
