import streamlit as st
from share import *
import config
import cv2
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

apply_canny = CannyDetector()
model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict('./checkpoints/checkpoint_19000.ckpt', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

st.title("Control Stable Diffusion with Canny Edge Maps")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    input_image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    prompt = st.text_input("Prompt")
    a_prompt = st.text_input("Added Prompt", "best quality, extremely detailed")
    n_prompt = st.text_input("Negative Prompt",
                             "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality")
    num_samples = st.slider("Images", 1, 12, 1)
    image_resolution = st.slider("Image Resolution", 256, 768, 512, 64)
    strength = st.slider("Control Strength", 0.0, 2.0, 1.0, 0.01)
    guess_mode = st.checkbox("Guess Mode")
    low_threshold = st.slider("Canny low threshold", 1, 255, 100)
    high_threshold = st.slider("Canny high threshold", 1, 255, 200)
    ddim_steps = st.slider("Steps", 1, 100, 20)
    scale = st.slider("Guidance Scale", 0.1, 30.0, 9.0, 0.1)
    seed = st.slider("Seed", -1, 2147483647, -1)
    eta = st.number_input("eta (DDIM)", 0.0)

    if st.button("Run"):
        results = process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold)
        for i, result in enumerate(results):
            st.image(result, caption=f"Output {i+1}")

def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)
        control = torch.from_numpy(detected_map).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = control.permute(0, 3, 1, 2).clone()
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)
        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
        cond = {
            "c_concat": [control],
            "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]
        }
        un_cond = {
            "c_concat": None if guess_mode else [control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]
        }
        shape = (4, img.shape[0] // 8, img.shape[1] // 8)
        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples, shape, cond, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)
        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
        x_samples = model.decode_first_stage(samples)
        x_samples = ((x_samples.permute(0, 2, 3, 1) * 127.5 + 127.5).clip(0, 255).cpu().numpy()).astype(np.uint8)
        return [255 - detected_map] + [x_samples[i] for i in range(num_samples)]
