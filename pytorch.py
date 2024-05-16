import torch

def check_gpu_usage():
    # Verifica se CUDA está disponível
    if torch.cuda.is_available():
        # Move um tensor para a GPU
        tensor = torch.randn(1).cuda()
        # Verifica se o tensor está na GPU
        if tensor.is_cuda:
            return True
    return False

if __name__ == "__main__":
    if check_gpu_usage():
        print("A GPU está sendo usada.")
    else:
        print("A GPU não está sendo usada.")

