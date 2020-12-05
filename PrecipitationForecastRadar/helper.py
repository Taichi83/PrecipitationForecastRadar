import torch


def prepare_pt_context(num_gpus, batch_size):
    """
    Correct batch size.
    Parameters
    ----------
    num_gpus : int
        Number of GPU.
    batch_size : int
        Batch size for each GPU.
    Returns
    -------
    bool
        Whether to use CUDA.
    int
        Batch size for all GPUs.
    """
    use_cuda = (num_gpus > 0)
    batch_size *= max(1, num_gpus)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return use_cuda, batch_size, device

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']