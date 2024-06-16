import gradio as gr
import openai
import langchain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
import os

# set the title  
title = "TAKE5 Activity ChatBot"

#Add a description
description = """
This is a chat-bot that recommends multiple self-care activites that only take 5 minutes, convenient, and ideally at home for users based on the self-care activites they enjoyed. 
If your user did not enjoy their activity, it will recommend multiple other self-care activities that are not similar to the one they disliked.
The user must input a detailed reason as to why they liked or disliked their activity."""

#Initialize default key parameters
history_list = []
llm = None
memory = None
conversation = None

#clear input textboxt after submission
def clear_and_save_textbox(message: str) -> tuple[str, str]:
    return '', message

#Display user input prior to model's response.
def display_input(message: str,
                  history: list[tuple[str, str]]) -> list[tuple[str, str]]:
    history.append((message, ''))
    return history

#process examples in app
def process_example(message: str) -> tuple[str, list[tuple[str, str]]]:
    x = predict(message)
    return '', x
    
    
#set api key and initialize openai models
def set_openai_api_key(api_key):
    if api_key and api_key.startswith("sk-") and len(api_key) > 50:
        os.environ["OPENAI_API_KEY"] = api_key
        openai.api_key = api_key
        global llm 
        global memory
        global conversation
        llm = OpenAI(temperature = 0.6)
    
        # Initialize Chatbot memory
        memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
        
        # Initialize the conversation pipeline (chain)
        conversation = ConversationChain(
        llm=llm,
        memory=memory)
    else:
        raise ValueError("Wrong API key")
        
#Utilize model for conversations
def predict(input):    
    response = conversation.predict(input = input)
    # messages = memory.buffer
    # history_list = create_tuples(messages,response_list=[])
    history_list.append((input,response))
    return history_list

#delete conversation history
def clear_message_history():
    global history_list
    history_list = []
    return ([], '')

#Modifying existing Gradio Theme
theme='shivi/calm_seafoam'

#description block
with gr.Blocks(theme = theme) as demo:
    gr.Markdown(description)

    # chatbot block
    with gr.Group():
        chatbot = gr.Chatbot(label='Chatbot')
        with gr.Row():
            openai_api_key_textbox = gr.Textbox(label="OpenAI Key", value="", type="password", placeholder="sk..", info = "You have to provide your own GPT3 keys for this app to function properly",)
           
        with gr.Row():
            textbox = gr.Textbox(
                container=False,
                show_label=False,
                placeholder='Type a message...',
            )

    with gr.Row():
        submit_button = gr.Button('Submit')
        clear_button = gr.Button('🗑️ Clear')

    saved_input = gr.State()

 
    
    #Example Block
    gr.Examples(
        examples=[
            ["Hi! Today I completed 5 minutes of yoga and I enjoyed it very much!"],
            ["Today I did journaling for 5 minutes and I enjoyed it. Can you recommend to me other mindfulness self care activities?"],
            ["Hello! Today I did some squats and did not enjoy it. Can you recommend less physically straining physical activities?"],
            ["Today I went for a run and enjoyed it very much! Can you recommend more physical activities that I can do outside?"],
            ["Hi! Today I did 5 minutes of meditation like you recommended to me and I did not enjoy it. Can you recommend to me physical exercise that I can do in 5 minutes for tomorrows activity?"],
         ],
        inputs=textbox,
        outputs=[textbox,chatbot],
        fn=process_example,
        cache_examples=False,
    )

    # "pass submission and get response" block
    textbox.submit(
        fn=clear_and_save_textbox,
        inputs=textbox,
        outputs=[textbox, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=display_input,
        inputs=[saved_input, chatbot],
        outputs=chatbot,
        api_name=False,
        queue=False,
    ).success(
        fn=predict,
        inputs=saved_input,
        outputs=chatbot,
        api_name=False,
    )

    # Activate Submit button
    button_event_preprocess = submit_button.click(
        fn=clear_and_save_textbox,
        inputs=textbox,
        outputs=[textbox, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=display_input,
        inputs=[saved_input, chatbot],
        outputs=chatbot,
        api_name=False,
        queue=False,
    ).success(
        fn=predict,
        inputs=saved_input,
        outputs=chatbot,
        api_name=False,
    )    
        

    # "Activate clear button when clicked"
    clear_button.click(
        fn=clear_message_history,
        outputs=[chatbot, saved_input],
        queue=False,
        api_name=False,
    )
    openai_api_key_textbox.change(set_openai_api_key,
                                          inputs=[openai_api_key_textbox],
                                          outputs=[])
    openai_api_key_textbox.submit(set_openai_api_key,
                                          inputs=[openai_api_key_textbox],
                                          outputs=[])
                                          outputs=[])
demo.queue(max_size=20).launch()
