# Copyright (c) 2023-2024 DeepSeek.
# ... (保留原始的版权声明和导入语句) ...

import base64
from io import BytesIO

import gradio as gr
import torch
from app_modules.gradio_utils import (
    cancel_outputing,
    delete_last_conversation,
    reset_state,
    reset_textbox,
    transfer_input,
    wrap_gen_fn,
)
from app_modules.overwrites import reload_javascript
from app_modules.presets import CONCURRENT_COUNT, description, description_top, title
from app_modules.utils import configure_logger, is_variable_assigned, strip_stop_words

from deepseek_vl.serve.inference import (
    convert_conversation_to_prompts,
    deepseek_generate,
    load_model,
)
from deepseek_vl.utils.conversation import SeparatorStyle

# 新增：导入 FastAPI 相关库
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Union
from PIL import Image
import signal
import sys
import json
from datetime import datetime

def load_models():
    models = {
        "DeepSeek-VL 1.3b": "C:\\Users\\Administrator\\Code\\deepseek-vl-1.3b-chat",
    }

    for model_name in models:
        models[model_name] = load_model(models[model_name])

    return models


logger = configure_logger()
models = load_models()
MODELS = sorted(list(models.keys()))


def generate_prompt_with_history(
    text, image, history, vl_chat_processor, tokenizer, max_length=2048
):
    """
    Generate a prompt with history for the deepseek application.

    Args:
        text (str): The text prompt.
        image (str): The image prompt.
        history (list): List of previous conversation messages.
        tokenizer: The tokenizer used for encoding the prompt.
        max_length (int): The maximum length of the prompt.

    Returns:
        tuple: A tuple containing the generated prompt, image list, conversation, and conversation copy. If the prompt could not be generated within the max_length limit, returns None.
    """

    sft_format = "deepseek"
    user_role_ind = 0
    bot_role_ind = 1

    # Initialize conversation
    conversation = vl_chat_processor.new_chat_template()

    if history:
        conversation.messages = history

    if image is not None:
        if "<image_placeholder>" not in text:
            text = (
                "<image_placeholder>" + "\n" + text
            )  # append the <image_placeholder> in a new line after the text prompt
        text = (text, image)

    conversation.append_message(conversation.roles[user_role_ind], text)
    conversation.append_message(conversation.roles[bot_role_ind], "")

    # Create a copy of the conversation to avoid history truncation in the UI
    conversation_copy = conversation.copy()
    logger.info("=" * 80)
    logger.info(get_prompt(conversation))

    rounds = len(conversation.messages) // 2

    for _ in range(rounds):
        current_prompt = get_prompt(conversation)
        current_prompt = (
            current_prompt.replace("</s>", "")
            if sft_format == "deepseek"
            else current_prompt
        )

        if torch.tensor(tokenizer.encode(current_prompt)).size(-1) <= max_length:
            return conversation_copy

        if len(conversation.messages) % 2 != 0:
            gr.Error("The messages between user and assistant are not paired.")
            return

        try:
            for _ in range(2):  # pop out two messages in a row
                conversation.messages.pop(0)
        except IndexError:
            gr.Error("Input text processing failed, unable to respond in this round.")
            return None

    gr.Error("Prompt could not be generated within max_length limit.")
    return None


def to_gradio_chatbot(conv):
    """Convert the conversation to gradio chatbot format."""
    ret = []
    for i, (role, msg) in enumerate(conv.messages[conv.offset :]):
        if i % 2 == 0:
            if type(msg) is tuple:
                msg, image = msg
                if isinstance(image, str):
                    with open(image, "rb") as f:
                        data = f.read()
                    img_b64_str = base64.b64encode(data).decode()
                    image_str = f'<video src="data:video/mp4;base64,{img_b64_str}" controls width="426" height="240"></video>'
                    msg = msg.replace("\n".join(["<image_placeholder>"] * 4), image_str)
                else:
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if H > W:
                        H, W = longest_edge, shortest_edge
                    else:
                        H, W = shortest_edge, longest_edge
                    image = image.resize((W, H))
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                    img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                    msg = msg.replace("<image_placeholder>", img_str)
            ret.append([msg, None])
        else:
            ret[-1][-1] = msg
    return ret


def to_gradio_history(conv):
    """Convert the conversation to gradio history state."""
    return conv.messages[conv.offset :]


def get_prompt(conv) -> str:
    """Get the prompt for generation."""
    system_prompt = conv.system_template.format(system_message=conv.system_message)
    if conv.sep_style == SeparatorStyle.DeepSeek:
        seps = [conv.sep, conv.sep2]
        if system_prompt == "" or system_prompt is None:
            ret = ""
        else:
            ret = system_prompt + seps[0]
        for i, (role, message) in enumerate(conv.messages):
            if message:
                if type(message) is tuple:  # multimodal message
                    message, _ = message
                ret += role + ": " + message + seps[i % 2]
            else:
                ret += role + ":"
        return ret
    else:
        return conv.get_prompt


@wrap_gen_fn
def predict(
    text,
    image,
    chatbot,
    history,
    top_p,
    temperature,
    repetition_penalty,
    max_length_tokens,
    max_context_length_tokens,
    model_select_dropdown,
):
    """
    Function to predict the response based on the user's input and selected model.

    Parameters:
    user_text (str): The input text from the user.
    user_image (str): The input image from the user.
    chatbot (str): The chatbot's name.
    history (str): The history of the chat.
    top_p (float): The top-p parameter for the model.
    temperature (float): The temperature parameter for the model.
    max_length_tokens (int): The maximum length of tokens for the model.
    max_context_length_tokens (int): The maximum length of context tokens for the model.
    model_select_dropdown (str): The selected model from the dropdown.

    Returns:
    generator: A generator that yields the chatbot outputs, history, and status.
    """
    print("running the prediction function")
    print("image \n", image)
    try:
        tokenizer, vl_gpt, vl_chat_processor = models[model_select_dropdown]

        if text == "":
            yield chatbot, history, "Empty context."
            return
    except KeyError:
        yield [[text, "No Model Found"]], [], "No Model Found"
        return

    conversation = generate_prompt_with_history(
        text,
        image,
        history,
        vl_chat_processor,
        tokenizer,
        max_length=max_context_length_tokens,
    )
    prompts = convert_conversation_to_prompts(conversation)

    stop_words = conversation.stop_str
    gradio_chatbot_output = to_gradio_chatbot(conversation)

    full_response = ""
    with torch.no_grad():
        for x in deepseek_generate(
            prompts=prompts,
            vl_gpt=vl_gpt,
            vl_chat_processor=vl_chat_processor,
            tokenizer=tokenizer,
            stop_words=stop_words,
            max_length=max_length_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
        ):
            full_response += x
            response = strip_stop_words(full_response, stop_words)
            conversation.update_last_message(response)
            gradio_chatbot_output[-1][1] = response
            yield gradio_chatbot_output, to_gradio_history(
                conversation
            ), "Generating..."

    print("flushed result to gradio")
    torch.cuda.empty_cache()

    if is_variable_assigned("x"):
        print(f"{model_select_dropdown}:\n{text}\n{'-' * 80}\n{x}\n{'=' * 80}")
        print(
            f"temperature: {temperature}, top_p: {top_p}, repetition_penalty: {repetition_penalty}, max_length_tokens: {max_length_tokens}"
        )

    yield gradio_chatbot_output, to_gradio_history(conversation), "Generate: Success"


def retry(
    text,
    image,
    chatbot,
    history,
    top_p,
    temperature,
    repetition_penalty,
    max_length_tokens,
    max_context_length_tokens,
    model_select_dropdown,
):
    if len(history) == 0:
        yield (chatbot, history, "Empty context")
        return

    chatbot.pop()
    history.pop()
    text = history.pop()[-1]
    if type(text) is tuple:
        text, image = text

    yield from predict(
        text,
        image,
        chatbot,
        history,
        top_p,
        temperature,
        repetition_penalty,
        max_length_tokens,
        max_context_length_tokens,
        model_select_dropdown,
    )


def build_demo(MODELS):
    with open("deepseek_vl/serve/assets/custom.css", "r", encoding="utf-8") as f:
        customCSS = f.read()

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        history = gr.State([])
        input_text = gr.State()
        input_image = gr.State()

        with gr.Row():
            gr.HTML(title)
            status_display = gr.Markdown("Success", elem_id="status_display")
        gr.Markdown(description_top)

        with gr.Row(equal_height=True):
            with gr.Column(scale=4):
                with gr.Row():
                    chatbot = gr.Chatbot(
                        elem_id="deepseek_chatbot",
                        show_share_button=True,
                        likeable=True,
                        bubble_full_width=False,
                        height=600,
                    )
                with gr.Row():
                    with gr.Column(scale=4):
                        text_box = gr.Textbox(
                            show_label=False, placeholder="Enter text", container=False
                        )
                    with gr.Column(
                        min_width=70,
                    ):
                        submitBtn = gr.Button("Send")
                    with gr.Column(
                        min_width=70,
                    ):
                        cancelBtn = gr.Button("Stop")
                with gr.Row():
                    emptyBtn = gr.Button(
                        "🧹 New Conversation",
                    )
                    retryBtn = gr.Button("🔄 Regenerate")
                    delLastBtn = gr.Button("🗑️ Remove Last Turn")

            with gr.Column():
                image_box = gr.Image(type="pil")

                with gr.Tab(label="Parameter Setting") as parameter_row:
                    top_p = gr.Slider(
                        minimum=-0,
                        maximum=1.0,
                        value=0.95,
                        step=0.05,
                        interactive=True,
                        label="Top-p",
                    )
                    temperature = gr.Slider(
                        minimum=0,
                        maximum=1.0,
                        value=0.1,
                        step=0.1,
                        interactive=True,
                        label="Temperature",
                    )
                    repetition_penalty = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=1.1,
                        step=0.1,
                        interactive=True,
                        label="Repetition penalty",
                    )
                    max_length_tokens = gr.Slider(
                        minimum=0,
                        maximum=4096,
                        value=2048,
                        step=8,
                        interactive=True,
                        label="Max Generation Tokens",
                    )
                    max_context_length_tokens = gr.Slider(
                        minimum=0,
                        maximum=4096,
                        value=4096,
                        step=128,
                        interactive=True,
                        label="Max History Tokens",
                    )
                    model_select_dropdown = gr.Dropdown(
                        label="Select Models",
                        choices=MODELS,
                        multiselect=False,
                        value=MODELS[0],
                        interactive=True,
                    )

        examples_list = [
            [
                "deepseek_vl/serve/examples/rap.jpeg",
                "Can you write me a master rap song that rhymes very well based on this image?",
            ],
            [
                "deepseek_vl/serve/examples/app.png",
                "What is this app about?",
            ],
            [
                "deepseek_vl/serve/examples/pipeline.png",
                "Help me write a python code based on the image.",
            ],
            [
                "deepseek_vl/serve/examples/chart.png",
                "Could you help me to re-draw this picture with python codes?",
            ],
            [
                "deepseek_vl/serve/examples/mirror.png",
                "How many people are there in the image. Why?",
            ],
            [
                "deepseek_vl/serve/examples/puzzle.png",
                "Can this 2 pieces combine together?",
            ],
        ]
        gr.Examples(examples=examples_list, inputs=[image_box, text_box])
        gr.Markdown(description)

        input_widgets = [
            input_text,
            input_image,
            chatbot,
            history,
            top_p,
            temperature,
            repetition_penalty,
            max_length_tokens,
            max_context_length_tokens,
            model_select_dropdown,
        ]
        output_widgets = [chatbot, history, status_display]

        transfer_input_args = dict(
            fn=transfer_input,
            inputs=[text_box, image_box],
            outputs=[input_text, input_image, text_box, image_box, submitBtn],
            show_progress=True,
        )

        predict_args = dict(
            fn=predict,
            inputs=input_widgets,
            outputs=output_widgets,
            show_progress=True,
        )

        retry_args = dict(
            fn=retry,
            inputs=input_widgets,
            outputs=output_widgets,
            show_progress=True,
        )

        reset_args = dict(
            fn=reset_textbox, inputs=[], outputs=[text_box, status_display]
        )

        predict_events = [
            text_box.submit(**transfer_input_args).then(**predict_args),
            submitBtn.click(**transfer_input_args).then(**predict_args),
        ]

        emptyBtn.click(reset_state, outputs=output_widgets, show_progress=True)
        emptyBtn.click(**reset_args)
        retryBtn.click(**retry_args)

        delLastBtn.click(
            delete_last_conversation,
            [chatbot, history],
            output_widgets,
            show_progress=True,
        )

        cancelBtn.click(cancel_outputing, [], [status_display], cancels=predict_events)

    return demo

# ------------------------ FastAPI 部分 ------------------------

app = FastAPI()

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)

# 定义请求体模型 (Pydantic 模型)
class ChatRequest(BaseModel):
    text: str
    image_path: Optional[str] = None
    history: List[List[Union[str, tuple]]] = []  # 允许字符串或包含图片路径的元组
    top_p: float = 0.95
    temperature: float = 0.1
    repetition_penalty: float = 1.1
    max_length_tokens: int = 2048
    max_context_length_tokens: int = 4096
    model_name: str = MODELS[0]

# 定义响应体模型
class ChatResponse(BaseModel):
    response: str
    history: List[List[Union[str, tuple]]]

def convert_image_to_base64(image: Image.Image) -> str:
    """将 PIL Image 对象转换为 Base64 字符串"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")  # 或者 "PNG"，取决于你想要的格式
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


# 创建 API 路由
@app.post("/chat")
async def chat_api(request: ChatRequest):
    """
    Chat API endpoint.

    Args:
        request: The request body containing text, image_path, and other parameters.

    Returns:
        ChatResponse: The response containing the generated text and updated history.
    """
    filePath = "C:\\Users\\Administrator\\Code\\video-finder\\public\\images\\"
    print(filePath + request.image_path)
    try:
        tokenizer, vl_gpt, vl_chat_processor = models[request.model_name]

        if request.text == "":
            raise HTTPException(status_code=400, detail="Empty context.")
    except KeyError:
        raise HTTPException(status_code=404, detail="No Model Found")

    # 处理图像（如果提供了 image_path）
    if request.image_path:
         try:
            with open(filePath + request.image_path, "rb") as image_file:
                image =  Image.open(image_file).convert("RGB") # 加载并转换为 PIL.Image
         except FileNotFoundError:
            raise HTTPException(status_code=400, detail="Image file not found.")
         except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image: {e}")
    else:
        image = None

    # 转换 Gradio 历史格式为 DeepSeek VL 格式
    history = request.history
    conversation = generate_prompt_with_history(
        request.text,
        image,
        history,
        vl_chat_processor,
        tokenizer,
        max_length=request.max_context_length_tokens,
    )
    if conversation is None:  # 历史记录过长
         raise HTTPException(status_code=400, detail="Context exceeds maximum length.")
    prompts = convert_conversation_to_prompts(conversation)

    stop_words = conversation.stop_str

    full_response = ""
    with torch.no_grad():
        for x in deepseek_generate(
            prompts=prompts,
            vl_gpt=vl_gpt,
            vl_chat_processor=vl_chat_processor,
            tokenizer=tokenizer,
            stop_words=stop_words,
            max_length=request.max_length_tokens,
            temperature=request.temperature,
            repetition_penalty=request.repetition_penalty,
            top_p=request.top_p,
        ):
            full_response += x

    response = strip_stop_words(full_response, stop_words)
    conversation.update_last_message(response)
    history = to_gradio_history(conversation) # 更新后的历史记录

    # 在返回之前，将 history 中的 PIL Image 对象转换为 Base64 字符串
    processed_history = []
    for role, message in history:
        if isinstance(message, tuple):  # 如果是 (text, image) 元组
            text, img = message
            if isinstance(img, Image.Image):
                img_b64 = convert_image_to_base64(img)
                processed_history.append([role, (text, img_b64)])  # 存储 Base64 字符串
            else:
                processed_history.append([role, message]) # 如果img已经是字符串(例如文件路径),则直接添加
        else:
            processed_history.append([role, message])

    torch.cuda.empty_cache()
    return ChatResponse(response=response, history=[])

@app.post("/chat-stream")
async def chat_api_stream(request: ChatRequest):
    """
    Chat API endpoint with streaming response.
    """
    file_path = "C:\\Users\\Administrator\\Code\\video-finder\\public\\images\\"

    try:
        tokenizer, vl_gpt, vl_chat_processor = models[request.model_name]
        if request.text == "":
            raise HTTPException(status_code=400, detail="Empty context.")
    except KeyError:
        raise HTTPException(status_code=404, detail="No Model Found")

    # 图像处理：如果提供了 image_path，则加载图像；否则，image 为 None
    if request.image_path:
        try:
            with open(file_path + request.image_path, "rb") as image_file:
                image = Image.open(image_file).convert("RGB")
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail="Image file not found.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image: {e}")
    else:
        image = None

    async def generate():
        nonlocal image, tokenizer, vl_gpt, vl_chat_processor, request  # 引用外部变量
        
        # 历史消息的处理
        history = request.history
        conversation = generate_prompt_with_history(
            request.text,
            image,
            history,
            vl_chat_processor,
            tokenizer,
            max_length=request.max_context_length_tokens,
        )
        if conversation is None:
            yield "data: [ERROR]\n\n"  # 使用 SSE 格式发送错误
            return

        prompts = convert_conversation_to_prompts(conversation)
        stop_words = conversation.stop_str
        full_response = ""

        with torch.no_grad():
            for x in deepseek_generate(
                prompts=prompts,
                vl_gpt=vl_gpt,
                vl_chat_processor=vl_chat_processor,
                tokenizer=tokenizer,
                stop_words=stop_words,
                max_length=request.max_length_tokens,
                temperature=request.temperature,
                repetition_penalty=request.repetition_penalty,
                top_p=request.top_p,
            ):
                full_response += x
                print(x)
                response_chunk = strip_stop_words(x, stop_words)  # 移除可能的停止词
                
                # 构建 JSON 响应
                json_response = {
                    "model": request.model_name,
                    "created_at": datetime.utcnow().isoformat() + "Z",  # 使用 UTC 时间
                    "message": {"role": "assistant", "content": response_chunk},
                    "done": False,
                }
                yield json.dumps(json_response) + "\n" #添加换行符
        
        response = strip_stop_words(full_response, stop_words)
        conversation.update_last_message(response)
        history = to_gradio_history(conversation)
        
        # 清理 CUDA 缓存
        torch.cuda.empty_cache()

        # 发送结束标记（可选，但建议）
        final_json_response = {
            "model": request.model_name,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "message": {"role": "assistant", "content": response},
            "done": True,
        }
        yield json.dumps(final_json_response)


    return StreamingResponse(generate(), media_type="text/event-stream")

# -------------------- 启动 Gradio 和 FastAPI --------------------

if __name__ == "__main__":
    import uvicorn

    demo = build_demo(MODELS)
    demo.title = "DeepSeek-VL Chatbot"

    reload_javascript()

    # 使用 uvicorn 同时启动 FastAPI 和 Gradio
    # 注意：这里需要在一个单独的线程中启动 Gradio
    import threading

    gradio_thread = None  # 全局变量，用于引用 Gradio 线程
    server_running = True #标志服务器是否在运行

    def run_gradio():
        global server_running # 声明使用全局变量
        demo.queue(concurrency_count=CONCURRENT_COUNT).launch(
            share=False,
            favicon_path="deepseek_vl/serve/assets/favicon.ico",
            inbrowser=False,  # 在生产环境中，通常不需要在浏览器中打开
            server_name="0.0.0.0",
            server_port=8122,
        )
        server_running = False #Gradio 停止后设置标志

    def signal_handler(sig, frame):
        """处理 SIGINT 信号 (Ctrl+C)"""
        global gradio_thread
        global server_running
        print("\n正在停止服务器...")
        server_running = False  # 设置标志，以便 Gradio 线程可以退出

        if gradio_thread and gradio_thread.is_alive():
            try:
                # 尝试关闭 Gradio 服务器。  这可能需要一些时间。
                demo.close()  # 显式关闭 Gradio 应用
            except Exception as e:
                print(f"关闭 Gradio 时出错: {e}")
            gradio_thread.join(timeout=5) # 等待线程结束,设置超时
            if gradio_thread.is_alive():
                print("警告：Gradio 线程未能正常终止。")

        print("服务器已停止。")
        sys.exit(0) # 确保进程退出

    # 注册信号处理程序
    signal.signal(signal.SIGINT, signal_handler)

    gradio_thread = threading.Thread(target=run_gradio, daemon=True)  # 设置为守护线程
    gradio_thread.start()

     # 启动 FastAPI
    uvicorn.run(app, host="0.0.0.0", port=8000)