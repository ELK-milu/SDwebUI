import html
import json
import math
import mimetypes
import os
import platform
import random
import subprocess as sp
import sys
import tempfile
import time
import traceback
from functools import partial, reduce

import gradio as gr
import gradio.routes
import gradio.utils
import numpy as np
from PIL import Image, PngImagePlugin

from routes.shared import opts


import routes.scripts
import routes.shared as shared

from routes import prompt_parser

# this is a fix for Windows users. Without it, javascript files will be served with text/html content-type and the browser will not show any UI
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')


gradio.utils.version_check = lambda: None
gradio.utils.get_local_ip_address = lambda: '127.0.0.1'


def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


sample_img2img = "assets/stable-samples/img2img/sketch-mountains-input.jpg"
sample_img2img = sample_img2img if os.path.exists(sample_img2img) else None

css_hide_progressbar = """
.wrap .m-12 svg { display:none!important; }
.wrap .m-12::before { content:"Loading..." }
.progress-bar { display:none!important; }
.meta-text { display:none!important; }
"""

# Using constants for these since the variation selector isn't visible.
# Important that they exactly match script.js for tooltip to work.
random_symbol = '\U0001f3b2\ufe0f'  # 🎲️
reuse_symbol = '\u267b\ufe0f'  # ♻️
art_symbol = '\U0001f3a8'  # 🎨
paste_symbol = '\u2199\ufe0f'  # ↙
folder_symbol = '\U0001f4c2'  # 📂
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
apply_style_symbol = '\U0001f4cb'  # 📋


def plaintext_to_html(text):
    text = "<p>" + "<br>\n".join([f"{html.escape(x)}" for x in text.split('\n')]) + "</p>"
    return text


def save_pil_to_file(pil_image, dir=None):
    use_metadata = False
    metadata = PngImagePlugin.PngInfo()
    for key, value in pil_image.info.items():
        if isinstance(key, str) and isinstance(value, str):
            metadata.add_text(key, value)
            use_metadata = True

    file_obj = tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=dir)
    pil_image.save(file_obj, pnginfo=(metadata if use_metadata else None))
    return file_obj


# override save to file function so that it also writes PNG info
gr.processing_utils.save_pil_to_file = save_pil_to_file


def wrap_gradio_call(func, extra_outputs=None):
    def f(*args, extra_outputs_array=extra_outputs, **kwargs):
        run_memmon = opts.memmon_poll_rate > 0 and not shared.mem_mon.disabled
        if run_memmon:
            shared.mem_mon.monitor()
        t = time.perf_counter()

        try:
            res = list(func(*args, **kwargs))
        except Exception as e:
            # When printing out our debug argument list, do not print out more than a MB of text
            max_debug_str_len = 131072 # (1024*1024)/8

            print("Error completing request", file=sys.stderr)
            argStr = f"Arguments: {str(args)} {str(kwargs)}"
            print(argStr[:max_debug_str_len], file=sys.stderr)
            if len(argStr) > max_debug_str_len:
                print(f"(Argument list truncated at {max_debug_str_len}/{len(argStr)} characters)", file=sys.stderr)

            print(traceback.format_exc(), file=sys.stderr)

            shared.state.job = ""
            shared.state.job_count = 0

            if extra_outputs_array is None:
                extra_outputs_array = [None, '']

            res = extra_outputs_array + [f"<div class='error'>{plaintext_to_html(type(e).__name__+': '+str(e))}</div>"]

        elapsed = time.perf_counter() - t
        elapsed_m = int(elapsed // 60)
        elapsed_s = elapsed % 60
        elapsed_text = f"{elapsed_s:.2f}s"
        if (elapsed_m > 0):
            elapsed_text = f"{elapsed_m}m "+elapsed_text

        if run_memmon:
            mem_stats = {k: -(v//-(1024*1024)) for k, v in shared.mem_mon.stop().items()}
            active_peak = mem_stats['active_peak']
            reserved_peak = mem_stats['reserved_peak']
            sys_peak = mem_stats['system_peak']
            sys_total = mem_stats['total']
            sys_pct = round(sys_peak/max(sys_total, 1) * 100, 2)

            vram_html = f"<p class='vram'>Torch active/reserved: {active_peak}/{reserved_peak} MiB, <wbr>Sys VRAM: {sys_peak}/{sys_total} MiB ({sys_pct}%)</p>"
        else:
            vram_html = ''

        # last item is always HTML
        res[-1] += f"<div class='performance'><p class='time'>Time taken: <wbr>{elapsed_text}</p>{vram_html}</div>"

        shared.state.skipped = False
        shared.state.interrupted = False
        shared.state.job_count = 0

        return tuple(res)

    return f


def calc_time_left(progress, threshold, label, force_display):
    if progress == 0:
        return ""
    else:
        time_since_start = time.time() - shared.state.time_start
        eta = (time_since_start/progress)
        eta_relative = eta-time_since_start
        if (eta_relative > threshold and progress > 0.02) or force_display:
            if eta_relative > 3600:
                return label + time.strftime('%H:%M:%S', time.gmtime(eta_relative))
            elif eta_relative > 60:
                return label + time.strftime('%M:%S',  time.gmtime(eta_relative))
            else:
                return label + time.strftime('%Ss',  time.gmtime(eta_relative))
        else:
            return ""


def check_progress_call(id_part):
    if shared.state.job_count == 0:
        return "", gr_show(False), gr_show(False), gr_show(False)

    progress = 0

    if shared.state.job_count > 0:
        progress += shared.state.job_no / shared.state.job_count
    if shared.state.sampling_steps > 0:
        progress += 1 / shared.state.job_count * shared.state.sampling_step / shared.state.sampling_steps

    time_left = calc_time_left( progress, 1, " ETA: ", shared.state.time_left_force_display )
    if time_left != "":
        shared.state.time_left_force_display = True

    progress = min(progress, 1)

    progressbar = ""
    if opts.show_progressbar:
        progressbar = f"""<div class='progressDiv'><div class='progress' style="overflow:visible;width:{progress * 100}%;white-space:nowrap;">{"&nbsp;" * 2 + str(int(progress*100))+"%" + time_left if progress > 0.01 else ""}</div></div>"""

    image = gr_show(False)
    preview_visibility = gr_show(False)

    if opts.show_progress_every_n_steps > 0:
        shared.state.set_current_image()
        image = shared.state.current_image

        if image is None:
            image = gr.update(value=None)
        else:
            preview_visibility = gr_show(True)

    if shared.state.textinfo is not None:
        textinfo_result = gr.HTML.update(value=shared.state.textinfo, visible=True)
    else:
        textinfo_result = gr_show(False)

    return f"<span id='{id_part}_progress_span' style='display: none'>{time.time()}</span><p>{progressbar}</p>", preview_visibility, image, textinfo_result


def check_progress_call_initial(id_part):
    shared.state.job_count = -1
    shared.state.current_latent = None
    shared.state.current_image = None
    shared.state.textinfo = None
    shared.state.time_start = time.time()
    shared.state.time_left_force_display = False

    return check_progress_call(id_part)


def roll_artist(prompt):
    allowed_cats = set([x for x in shared.artist_db.categories() if len(opts.random_artist_categories)==0 or x in opts.random_artist_categories])
    artist = random.choice([x for x in shared.artist_db.artists if x.category in allowed_cats])

    return prompt + ", " + artist.name if prompt != '' else artist.name


def visit(x, func, path=""):
    if hasattr(x, 'children'):
        for c in x.children:
            visit(c, func, path)
    elif x.label is not None:
        func(path + "/" + str(x.label), x)



def apply_styles(prompt, prompt_neg, style1_name, style2_name):
    prompt = shared.prompt_styles.apply_styles_to_prompt(prompt, [style1_name, style2_name])
    prompt_neg = shared.prompt_styles.apply_negative_styles_to_prompt(prompt_neg, [style1_name, style2_name])

    return [gr.Textbox.update(value=prompt), gr.Textbox.update(value=prompt_neg), gr.Dropdown.update(value="None"), gr.Dropdown.update(value="None")]


def interrogate(image):
    prompt = shared.interrogator.interrogate(image)

    return gr_show(True) if prompt is None else prompt



def create_seed_inputs():
    with gr.Row():
        with gr.Box(elem_id='seed_box'):
            with gr.Row(elem_id='seed_row'):
                seed = (gr.Number)(label='Seed', value=-1)
                seed.style(container=False)
                random_seed = gr.Button(random_symbol, elem_id='random_seed')
                reuse_seed = gr.Button(reuse_symbol, elem_id='reuse_seed')

        with gr.Box(elem_id='subseed_show_box'):
            seed_checkbox = gr.Checkbox(label='Extra', elem_id='subseed_show', value=False)

    # Components to show/hide based on the 'Extra' checkbox
    seed_extras = []

    with gr.Row(visible=False) as seed_extra_row_1:
        seed_extras.append(seed_extra_row_1)
        with gr.Box():
            with gr.Row(elem_id='subseed_row'):
                subseed = gr.Number(label='Variation seed', value=-1)
                subseed.style(container=False)
                random_subseed = gr.Button(random_symbol, elem_id='random_subseed')
                reuse_subseed = gr.Button(reuse_symbol, elem_id='reuse_subseed')
        subseed_strength = gr.Slider(label='Variation strength', value=0.0, minimum=0, maximum=1, step=0.01)

    with gr.Row(visible=False) as seed_extra_row_2:
        seed_extras.append(seed_extra_row_2)
        seed_resize_from_w = gr.Slider(minimum=0, maximum=2048, step=64, label="Resize seed from width", value=0)
        seed_resize_from_h = gr.Slider(minimum=0, maximum=2048, step=64, label="Resize seed from height", value=0)

    random_seed.click(fn=lambda: -1, show_progress=False, inputs=[], outputs=[seed])
    random_subseed.click(fn=lambda: -1, show_progress=False, inputs=[], outputs=[subseed])

    def change_visibility(show):
        return {comp: gr_show(show) for comp in seed_extras}

    seed_checkbox.change(change_visibility, show_progress=False, inputs=[seed_checkbox], outputs=seed_extras)

    return seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox


def connect_reuse_seed(seed: gr.Number, reuse_seed: gr.Button, generation_info: gr.Textbox, dummy_component, is_subseed):
    """ Connects a 'reuse (sub)seed' button's click event so that it copies last used
        (sub)seed value from generation info the to the seed field. If copying subseed and subseed strength
        was 0, i.e. no variation seed was used, it copies the normal seed value instead."""
    def copy_seed(gen_info_string: str, index):
        res = -1

        try:
            gen_info = json.loads(gen_info_string)
            index -= gen_info.get('index_of_first_image', 0)

            if is_subseed and gen_info.get('subseed_strength', 0) > 0:
                all_subseeds = gen_info.get('all_subseeds', [-1])
                res = all_subseeds[index if 0 <= index < len(all_subseeds) else 0]
            else:
                all_seeds = gen_info.get('all_seeds', [-1])
                res = all_seeds[index if 0 <= index < len(all_seeds) else 0]

        except json.decoder.JSONDecodeError as e:
            if gen_info_string != '':
                print("Error parsing JSON generation info:", file=sys.stderr)
                print(gen_info_string, file=sys.stderr)

        return [res, gr_show(False)]

    reuse_seed.click(
        fn=copy_seed,
        _js="(x, y) => [x, selected_gallery_index()]",
        show_progress=False,
        inputs=[generation_info, dummy_component],
        outputs=[seed, dummy_component]
    )




def create_toprow(is_img2img):
    id_part = "txt2img" if is_img2img else "txt2img"

    with gr.Row(elem_id="toprow"):
        with gr.Column(scale=5, min_width=500):
            with gr.Row():
                prompt = gr.Textbox(label="Prompt", elem_id=f"{id_part}_prompt", show_label=False, lines=2,
                                    placeholder="Prompt (press Ctrl+Enter or Alt+Enter to generate)"
                                    )
            with gr.Row():
                negative_prompt = gr.Textbox(label="Negative prompt", elem_id=f"{id_part}_neg_prompt", show_label=False,
                                             lines=2,
                                             placeholder="Negative prompt (press Ctrl+Enter or Alt+Enter to generate)"
                                             )

        with gr.Column(scale=0.1, min_width=40):
            with gr.Row(elem_id="choosen"):
                roll = gr.Button(value=art_symbol, elem_id="roll")
                paste = gr.Button(value=paste_symbol, elem_id="paste")
                save_style = gr.Button(value=save_style_symbol, elem_id="style_create")
                prompt_style_apply = gr.Button(value=apply_style_symbol, elem_id="style_apply")

                token_counter = gr.HTML(value="<span></span>", elem_id=f"{id_part}_token_counter")
                token_button = gr.Button(visible=False, elem_id=f"{id_part}_token_button")

        button_interrogate = None
        button_deepbooru = None
        if is_img2img:
            with gr.Column(elem_id="interrogate_col"):
                button_interrogate = gr.Button('Interrogate\nCLIP', elem_id="interrogate")
                button_deepbooru = gr.Button('Interrogate\nDeepBooru', elem_id="deepbooru")

        with gr.Column(scale=1):
            submit = gr.Button('Generate', elem_id=f"{id_part}_generate", variant='primary')
            skip = gr.Button('Skip', elem_id=f"{id_part}_skip")
            interrupt = gr.Button('Interrupt', elem_id=f"{id_part}_interrupt")

            skip.click(
                fn=lambda: shared.state.skip(),
                inputs=[],
                outputs=[],
            )

            interrupt.click(
                fn=lambda: shared.state.interrupt(),
                inputs=[],
                outputs=[],
            )

            interrupt.click(
                fn=lambda: shared.state.interrupt(),
                inputs=[],
                outputs=[],
            )

            with gr.Row(visible=False, ):
                with gr.Column(visible=False, scale=1, elem_id="style_pos_col"):
                    prompt_style = gr.Dropdown(label="Style 1")
                    prompt_style.save_to_config = True

                with gr.Column(visible=False, scale=1, elem_id="style_neg_col"):
                    prompt_style2 = gr.Dropdown(label="Style 2")
                    prompt_style2.save_to_config = True

    return prompt, roll, prompt_style, negative_prompt, prompt_style2, submit, button_interrogate, button_deepbooru, prompt_style_apply, save_style, paste, token_counter, token_button


def setup_progressbar(progressbar, preview, id_part, textinfo=None):
    if textinfo is None:
        textinfo = gr.HTML(visible=False)

    check_progress = gr.Button('Check progress', elem_id=f"{id_part}_check_progress", visible=False)
    check_progress.click(
        fn=lambda: check_progress_call(id_part),
        show_progress=False,
        inputs=[],
        outputs=[progressbar, preview, preview, textinfo],
    )

    check_progress_initial = gr.Button('Check progress (first)', elem_id=f"{id_part}_check_progress_initial", visible=False)
    check_progress_initial.click(
        fn=lambda: check_progress_call_initial(id_part),
        show_progress=False,
        inputs=[],
        outputs=[progressbar, preview, preview, textinfo],
    )





def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id):
    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        for k, v in args.items():
            setattr(refresh_component, k, v)

        return gr.update(**(args or {}))

    refresh_button = gr.Button(value=refresh_symbol, elem_id=elem_id)
    refresh_button.click(
        fn=refresh,
        inputs=[],
        outputs=[refresh_component]
    )
    return refresh_button




    def create_setting_component(key, is_quicksettings=False):
        def fun():
            return opts.data[key] if key in opts.data else opts.data_labels[key].default

        info = opts.data_labels[key]
        t = type(info.default)

        args = info.component_args() if callable(info.component_args) else info.component_args

        if info.component is not None:
            comp = info.component
        elif t == str:
            comp = gr.Textbox
        elif t == int:
            comp = gr.Number
        elif t == bool:
            comp = gr.Checkbox
        else:
            raise Exception(f'bad options item type: {str(t)} for key {key}')

        elem_id = "setting_"+key

        if info.refresh is not None:
            if is_quicksettings:
                res = comp(label=info.label, value=fun, elem_id=elem_id, **(args or {}))
                create_refresh_button(res, info.refresh, info.component_args, "refresh_" + key)
            else:
                with gr.Row(variant="compact"):
                    res = comp(label=info.label, value=fun, elem_id=elem_id, **(args or {}))
                    create_refresh_button(res, info.refresh, info.component_args, "refresh_" + key)
        else:
            res = comp(label=info.label, value=fun, elem_id=elem_id, **(args or {}))


        return res

    components = []
    component_dict = {}

    script_callbacks.ui_settings_callback()
    opts.reorder()

    with gr.Blocks(analytics_enabled=False) as settings_interface:
        settings_submit = gr.Button(value="Apply settings", variant='primary')
        result = gr.HTML()

        settings_cols = 3
        items_per_col = int(len(opts.data_labels) * 0.9 / settings_cols)

        quicksettings_names = [x.strip() for x in opts.quicksettings.split(",")]
        quicksettings_names = set(x for x in quicksettings_names if x != 'quicksettings')

        quicksettings_list = []

        cols_displayed = 0
        items_displayed = 0
        previous_section = None
        column = None
        with gr.Row(elem_id="settings").style(equal_height=False):
            for i, (k, item) in enumerate(opts.data_labels.items()):
                section_must_be_skipped = item.section[0] is None

                if previous_section != item.section and not section_must_be_skipped:
                    if cols_displayed < settings_cols and (items_displayed >= items_per_col or previous_section is None):
                        if column is not None:
                            column.__exit__()

                        column = gr.Column(variant='panel')
                        column.__enter__()

                        items_displayed = 0
                        cols_displayed += 1

                    previous_section = item.section

                    elem_id, text = item.section
                    gr.HTML(elem_id="settings_header_text_{}".format(elem_id), value='<h1 class="gr-button-lg">{}</h1>'.format(text))

                if k in quicksettings_names and not shared.cmd_opts.freeze_settings:
                    quicksettings_list.append((i, k, item))
                    components.append(dummy_component)
                elif section_must_be_skipped:
                    components.append(dummy_component)
                else:
                    component = create_setting_component(k)
                    component_dict[k] = component
                    components.append(component)
                    items_displayed += 1

        with gr.Row():
            request_notifications = gr.Button(value='Request browser notifications', elem_id="request_notifications")
            download_localization = gr.Button(value='Download localization template', elem_id="download_localization")

        with gr.Row():
            reload_script_bodies = gr.Button(value='Reload custom script bodies (No ui updates, No restart)', variant='secondary')
            restart_gradio = gr.Button(value='Restart Gradio and Refresh components (Custom Scripts, ui.py, js and css only)', variant='primary')

        request_notifications.click(
            fn=lambda: None,
            inputs=[],
            outputs=[],
            _js='function(){}'
        )

        download_localization.click(
            fn=lambda: None,
            inputs=[],
            outputs=[],
            _js='download_localization'
        )


        reload_script_bodies.click(
            fn=reload_scripts,
            inputs=[],
            outputs=[]
        )

        def request_restart():
            shared.state.interrupt()
            shared.state.need_restart = True

        restart_gradio.click(

            fn=request_restart,
            inputs=[],
            outputs=[],
            _js='restart_reload'
        )

        if column is not None:
            column.__exit__()

    interfaces = [
        (txt2img_interface, "txt2img", "txt2img"),
        (img2img_interface, "img2img", "img2img"),
        (extras_interface, "Extras", "extras"),
        (pnginfo_interface, "PNG Info", "pnginfo"),
        (modelmerger_interface, "Checkpoint Merger", "modelmerger"),
        (train_interface, "Train", "ti"),
    ]

    css = ""

    for cssfile in modules.scripts.list_files_with_name("style.css"):
        if not os.path.isfile(cssfile):
            continue

        with open(cssfile, "r", encoding="utf8") as file:
            css += file.read() + "\n"

    if os.path.exists(os.path.join(script_path, "user.css")):
        with open(os.path.join(script_path, "user.css"), "r", encoding="utf8") as file:
            css += file.read() + "\n"

    if not cmd_opts.no_progressbar_hiding:
        css += css_hide_progressbar

    interfaces += script_callbacks.ui_tabs_callback()
    interfaces += [(settings_interface, "Settings", "settings")]

    extensions_interface = ui_extensions.create_ui()
    interfaces += [(extensions_interface, "Extensions", "extensions")]

    with gr.Blocks(css=css, analytics_enabled=False, title="Stable Diffusion") as demo:
        with gr.Row(elem_id="quicksettings"):
            for i, k, item in quicksettings_list:
                component = create_setting_component(k, is_quicksettings=True)
                component_dict[k] = component

        parameters_copypaste.integrate_settings_paste_fields(component_dict)
        parameters_copypaste.run_bind()

        with gr.Tabs(elem_id="tabs") as tabs:
            for interface, label, ifid in interfaces:
                with gr.TabItem(label, id=ifid, elem_id='tab_' + ifid):
                    interface.render()

        if os.path.exists(os.path.join(script_path, "notification.mp3")):
            audio_notification = gr.Audio(interactive=False, value=os.path.join(script_path, "notification.mp3"), elem_id="audio_notification", visible=False)

        text_settings = gr.Textbox(elem_id="settings_json", value=lambda: opts.dumpjson(), visible=False)
        settings_submit.click(
            fn=run_settings,
            inputs=components,
            outputs=[result, text_settings],
        )

        for i, k, item in quicksettings_list:
            component = component_dict[k]

            component.change(
                #fn=lambda value, k=k: run_settings_single(value, key=k),
                inputs=[component],
                outputs=[component, text_settings],
            )

        def modelmerger(*args):
            try:
                results = routes.extras.run_modelmerger(*args)
            except Exception as e:
                print("Error loading/saving model file:", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                routes.sd_models.list_models()  # to remove the potentially missing models from the list
                return ["Error loading/saving model file. It doesn't exist or the name contains illegal characters"] + [gr.Dropdown.update(choices=routes.sd_models.checkpoint_tiles()) for _ in range(3)]
            return results

        modelmerger_merge.click(
            fn=modelmerger,
            inputs=[
                primary_model_name,
                secondary_model_name,
                tertiary_model_name,
                interp_method,
                interp_amount,
                save_as_half,
                custom_name,
            ],
            outputs=[
                submit_result,
                primary_model_name,
                secondary_model_name,
                tertiary_model_name,
                component_dict['sd_model_checkpoint'],
            ]
        )

    ui_config_file = cmd_opts.ui_config_file
    ui_settings = {}
    settings_count = len(ui_settings)
    error_loading = False

    try:
        if os.path.exists(ui_config_file):
            with open(ui_config_file, "r", encoding="utf8") as file:
                ui_settings = json.load(file)
    except Exception:
        error_loading = True
        print("Error loading settings:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

    def loadsave(path, x):
        def apply_field(obj, field, condition=None, init_field=None):
            key = path + "/" + field

            if getattr(obj, 'custom_script_source', None) is not None:
              key = 'customscript/' + obj.custom_script_source + '/' + key

            if getattr(obj, 'do_not_save_to_config', False):
                return

            saved_value = ui_settings.get(key, None)
            if saved_value is None:
                ui_settings[key] = getattr(obj, field)
            elif condition and not condition(saved_value):
                print(f'Warning: Bad ui setting value: {key}: {saved_value}; Default value "{getattr(obj, field)}" will be used instead.')
            else:
                setattr(obj, field, saved_value)
                if init_field is not None:
                    init_field(saved_value)

        if type(x) in [gr.Slider, gr.Radio, gr.Checkbox, gr.Textbox, gr.Number] and x.visible:
            apply_field(x, 'visible')

        if type(x) == gr.Slider:
            apply_field(x, 'value')
            apply_field(x, 'minimum')
            apply_field(x, 'maximum')
            apply_field(x, 'step')

        if type(x) == gr.Radio:
            apply_field(x, 'value', lambda val: val in x.choices)

        if type(x) == gr.Checkbox:
            apply_field(x, 'value')

        if type(x) == gr.Textbox:
            apply_field(x, 'value')

        if type(x) == gr.Number:
            apply_field(x, 'value')

        # Since there are many dropdowns that shouldn't be saved,
        # we only mark dropdowns that should be saved.
        if type(x) == gr.Dropdown and getattr(x, 'save_to_config', False):
            apply_field(x, 'value', lambda val: val in x.choices, getattr(x, 'init_field', None))
            apply_field(x, 'visible')

    visit(txt2img_interface, loadsave, "txt2img")
    visit(img2img_interface, loadsave, "img2img")
    visit(extras_interface, loadsave, "extras")
    visit(modelmerger_interface, loadsave, "modelmerger")

    if not error_loading and (not os.path.exists(ui_config_file) or settings_count != len(ui_settings)):
        with open(ui_config_file, "w", encoding="utf8") as file:
            json.dump(ui_settings, file, indent=4)

    return demo



