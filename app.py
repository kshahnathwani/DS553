import gradio as gr
from huggingface_hub import InferenceClient

"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")


def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
    personality,
):
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content

        response += token
        yield response

def clear_chat():
    return "",[]

"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="Chat")
    system_message = gr.Textbox(value="You are a friendly Chatbot.", label="System message")
    max_tokens = gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens")
    temperature = gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature")
    top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)")
    personality = gr.Dropdown(choices=["friendly", "professional", "humorous", "serious"], value="friendly", label="Personality")
    message = gr.Textbox(placeholder="Enter your message here...", label="Message")
    submit_button = gr.Button("Send Message")
    clear_button = gr.Button("Clear Chat")

    submit_button.click(
        respond,
        inputs=[message, chatbot, system_message, max_tokens, temperature, top_p, personality],  # Pass personality as input
        outputs=[chatbot]
    )

    clear_button.click(
        clear_chat,
        outputs=[message,chatbot]
    )
   
    
)


if __name__ == "__main__":
    demo.launch()