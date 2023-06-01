import sys, os, shlex
import contextlib
import torch


# has_mps is only available in nightly pytorch (for now), `getattr` for compatibility
has_mps = getattr(torch, 'has_mps', False)

cpu = torch.device("cpu")

def extract_device_id(args, name):
    for x in range(len(args)):
        if name in args[x]: return args[x+1]
    return None


device = device_interrogate = device_gfpgan = device_swinir = device_esrgan = device_scunet = device_codeformer = None
dtype = torch.float16
dtype_vae = torch.float16

def randn(seed, shape):
    # Pytorch currently doesn't handle setting randomness correctly when the metal backend is used.
    if device.type == 'mps':
        generator = torch.Generator(device=cpu)
        generator.manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=cpu).to(device)
        return noise

    torch.manual_seed(seed)
    return torch.randn(shape, device=device)


def randn_without_seed(shape):
    # Pytorch currently doesn't handle setting randomness correctly when the metal backend is used.
    if device.type == 'mps':
        generator = torch.Generator(device=cpu)
        noise = torch.randn(shape, generator=generator, device=cpu).to(device)
        return noise

    return torch.randn(shape, device=device)


# MPS workaround for https://github.com/pytorch/pytorch/issues/79383
def mps_contiguous(input_tensor, device): return input_tensor.contiguous() if device.type == 'mps' else input_tensor
def mps_contiguous_to(input_tensor, device): return mps_contiguous(input_tensor, device).to(device)
