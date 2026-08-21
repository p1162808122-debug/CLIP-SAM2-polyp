"""Small training helper using the local BiomedCLIP model."""

from .contrastive_loss import clip_loss


def train_step(model, images, tokens, optimizer, scaler=None):
    """Run one train step and return the detached scalar loss tensor.

    Mixed precision can be supplied by wrapping the model call in an autocast
    context and passing a ``torch.cuda.amp.GradScaler`` as ``scaler``.
    """

    model.train()
    optimizer.zero_grad(set_to_none=True)
    image_features, text_features, logit_scale = model(images, tokens)
    loss = clip_loss(image_features, text_features, logit_scale)

    if scaler is None:
        loss.backward()
        optimizer.step()
    else:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    return loss.detach()
