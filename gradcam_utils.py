import torch
import torch.nn.functional as F

def compute_gradcam(model, input_tensor, target_layer):
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    handle_b = target_layer.register_backward_hook(backward_hook)
    handle_f = target_layer.register_forward_hook(forward_hook)

    output = model(input_tensor)
    class_score = output.mean()
    class_score.backward()

    grad = gradients[0]
    act = activations[0]

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1).squeeze()

    cam = F.relu(cam)
    cam -= cam.min()
    cam /= cam.max() + 1e-8
    cam = cam.detach().cpu().numpy()

    handle_b.remove()
    handle_f.remove()

    return cam
