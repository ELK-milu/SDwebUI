import gradio as gr
from routes.function import condition_generation, stop, uncondition_generation
from routes.ui import create_seed_inputs, create_toprow, setup_progressbar

'''
样式设计部分，设计整体布局并调用功能函数
'''


with gr.Blocks(title="PikachuFace",
               css="./css/main.css",
               ) as demo:
    with gr.Row():
        gr.Markdown(
            "# 欢迎使用宝可梦中文生成图像系统!"
            "\n操作指南:\n1.输入中文词组或句子\n2.点击生成按钮\n3.享受图像\n",
            elem_id="title"
        )
    with gr.Tabs(elem_id="Tabs"):
        with gr.TabItem("无条件生成"):
            with gr.Row():
                with gr.Column(scale=1.3, variant="panel"):
                    uncondition_steps = gr.Slider(label="迭代步数", minimum=1, maximum=100, step=1, value=50, interactive=True)
                    uncondition_seed = gr.Slider(label="Seed", minimum=0, maximum=2147483647, step=1, randomize=True,
                                     interactive=True)
                    with gr.Row():
                        uncondition_generation_btn = gr.Button(value="生成", variant="primary")
                        uncondition_stop_btn = gr.Button(value="停止")
                with gr.Column(scale=1, variant="panel"):
                    uncondition_image_gener = gr.Image(label="生成图像", interactive=False).style(height=256, width=512)
        with gr.TabItem("条件生成(简单模式)"):
            with gr.Row():
                with gr.Column(scale=1.3, variant="panel"):
                    input_text = gr.Textbox(label="输入框")
                    steps = gr.Slider(label="迭代步数", minimum=1, maximum=100, step=1, value=50, interactive=True)
                    seed = gr.Slider(label="Seed", minimum=0, maximum=2147483647, step=1, randomize=True,
                                     interactive=True)
                    with gr.Row():
                        generation_btn = gr.Button(value="生成", variant="primary")
                        stop_btn = gr.Button(value="停止")
                with gr.Column(scale=1, variant="panel"):
                    image_gener = gr.Image(label="生成图像", interactive=False).style(height=256, width=512)
            with gr.Row():
                gr.Examples(
                    [["粉色的蝴蝶,小精灵,卡通", 50, 746045056], ["可爱的狗,小精灵,卡通", 50, 199901011],
                     ["可爱的,毛绒的,黄色的老鼠,小精灵,卡通", 50, 684298636]],
                    [input_text, steps, seed],
                    image_gener,
                    condition_generation,
                    cache_examples=True,
                )
        with gr.TabItem("条件生成(复杂模式)"):
            txt2img_prompt, roll, txt2img_prompt_style, txt2img_negative_prompt, txt2img_prompt_style2, submit, _, _, txt2img_prompt_style_apply, txt2img_save_style, txt2img_paste, token_counter, token_button = create_toprow(
                is_img2img=False)
            dummy_component = gr.Label(visible=False)
            txt_prompt_img = gr.File(label="", elem_id="txt2img_prompt_image", file_count="single", type="bytes",
                                     visible=False)

            with gr.Row(elem_id='txt2img_progress_row'):
                with gr.Column(variant='panel'):
                    steps_f = gr.Slider(minimum=1, maximum=150, step=1, label="Sampling Steps", value=20, interactive=True)
                    sampler_index = gr.Radio(label='Sampling method', choices=["DDPM", "DDIM"], elem_id="txt2img_sampling",
                                             type="index")

                    with gr.Group():
                        width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=512, interactive=True)
                        height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512, interactive=True)

                    with gr.Row(visible=False):
                        restore_faces = gr.Checkbox(label='Restore faces', value=False)
                        tiling = gr.Checkbox(label='Tiling', value=False)
                        enable_hr = gr.Checkbox(label='Highres. fix', value=False)

                    with gr.Row(visible=False) as hr_options:
                        firstphase_width = gr.Slider(minimum=0, maximum=1024, step=64, label="Firstpass width",
                                                     value=0, interactive=True)
                        firstphase_height = gr.Slider(minimum=0, maximum=1024, step=64, label="Firstpass height",
                                                      value=0, interactive=True)
                        denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01,
                                                       label='Denoising strength', value=0.7, interactive=True)

                    with gr.Row(equal_height=True):
                        batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, interactive=True)
                        batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, interactive=True)

                    with gr.Row(equal_height=True):
                        cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=7.0, interactive=True)

                    seed_f, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox = create_seed_inputs()

                with gr.Column():
                    progressbar = gr.HTML(elem_id="txt2img_progressbar")
                    txt2img_preview = gr.Image(elem_id='image_window', visible=True)
                    setup_progressbar(progressbar, txt2img_preview, 'txt2img')

    generation_btn.click(fn=condition_generation, inputs=[input_text, steps, seed], outputs=image_gener, api_name="condition_generation")
    stop_btn.click(fn=stop, api_name="condition_stop")
    uncondition_generation_btn.click(fn=uncondition_generation, inputs=[steps, seed], outputs=uncondition_image_gener, api_name="uncondition_generation")
    uncondition_stop_btn.click(fn=stop, api_name="uncondition_stop")

