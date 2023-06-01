import os
import multiprocessing
import signal
import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline

'''
模型读取部分，从pipeline读取模型
'''

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
CUDA_BATCH_SIZE = 1
# clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", torch_dtype=torch.float16)
pipe1 = StableDiffusionPipeline.from_pretrained("IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Anime-Chinese-v0.1")
model_path = "miluELK/Taiyi-sd-pokemon-LoRA-zh-512-v2"
pipe2 = DiffusionPipeline.from_pretrained("miluELK/ddpm-pokemon-64-v2")


'''
函数定义部分，定义一些功能函数
'''


def condition_generation(prompt, steps, seed):
    global mp
    mp = multiprocessing.current_process().pid
    pipe1.unet.load_attn_procs(model_path)
    # pipe.to("cuda")
    generator = torch.Generator("cpu").manual_seed(seed)
    pipe1.safety_checker = lambda images, clip_input: (images, False)
    image = pipe1(prompt, generator=generator, num_inference_steps=steps, guidance_scale=7.5).images[0]
    image.save("./outputs/condition/" + prompt + ".png")
    return image

def uncondition_generation(steps, seed):
    global mp
    mp = multiprocessing.current_process().pid
    generator = torch.Generator("cpu").manual_seed(seed)
    pipe2.safety_checker = lambda images, clip_input: (images, False)
    image = pipe2(generator=generator, num_inference_steps=steps).images[0]
    image.save("./outputs/unconditon/" + str(steps)+"_"+str(seed) + ".png")
    return image

def stop():
    os.kill(mp, signal.SIGINT)
    return 0

