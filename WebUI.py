import gradio as gr
from fastapi import FastAPI
from fastapi.responses import FileResponse
from starlette.responses import StreamingResponse

from routes.main import demo

'''
接口定义部分，自定义一些API
'''
import requests

API_URL = "https://api-inference.huggingface.co/models/miluELK/Taiyi-sd-pokemon-LoRA-zh-512-v2"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content

# You can access the image with PIL.Image for example
import io
from PIL import Image



CUSTOM_PATH = "/main"
app = FastAPI()



@app.get("/")
def read_main():
    return {"message": "在网址后输入/gener/+中文提示词进行条件生成,在网址后输入/main进入UI界面"}

@app.get("/gener/{inputs}")
def generation(inputs: str):
    print(inputs)
    image_bytes = query({
        "inputs": inputs,
    })
    image = Image.open(io.BytesIO(image_bytes))
    while True:
        return StreamingResponse(image_bytes, media_type="image/jpg")

@app.get("/get")
def read_main():
    return {"message": "This is your main app"}


favicon_path = './public/favicon.ico'
api_logo = './assets/static/img/api-logo.svg'


@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)


@app.get('/static/img/api-logo.svg')
async def logo():
    return FileResponse(api_logo)

app = gr.mount_gradio_app(app, demo, path=CUSTOM_PATH)

# 这行代码会控制子进程数量,如果concurrency_count过少会导致无法加载stop()进程
# demo.queue(concurrency_count=1)

if __name__ == "__main__":
    demo.launch()
