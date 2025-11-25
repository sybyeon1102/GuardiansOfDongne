import torch


def choose_torch_device() -> torch.device:
    """시스템의 CUDA 지원 여부에 따라 torch.device를 자동으로 선택한다.

    - torch.cuda.is_available() == True → device('cuda')
    - 아니면 → device('cpu')
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def is_cuda_device(device: torch.device) -> bool:
    """주어진 device가 CUDA 디바이스인지 여부를 반환한다."""
    return device.type == "cuda"
