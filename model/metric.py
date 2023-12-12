import torch

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def mlm_accuracy(output, target):
    with torch.no_grad():
        mlm_output = output['language_model_logits']
        mlm_preds = torch.argmax(mlm_output, dim=1)

        correct = torch.sum((mlm_preds == target) & (target != -100)).item()
    return correct / (target != -100).sum().item()

def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
