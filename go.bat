CALL venv\Scripts\activate
gradio UI.py
uvicorn UI:app --host='127.0.0.1' --port=7860 --reload
start http://127.0.0.1:7860/main
cmd /k