import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import streamlit as st
import torch


def load_tokenizer_and_model(model="microsoft/DialoGPT-large"):
    """
      Load tokenizer and model instance for some specific DialoGPT model.
    """
    # Initialize tokenizer and model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model)

    # Return tokenizer and model
    return tokenizer, model

@st.cache(hash_funcs={transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast: hash}, suppress_st_warning=True)
def load_data():
    #DialoGPT-medium
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
    return tokenizer, model

tokenizer, model = load_data()

#Set up the streamlit
st.write("Welcome to the Chatbot. I am still learning, please be patient")
input = st.text_input('User:')
if 'count' not in st.session_state or st.session_state.count == 6:
    st.session_state.count = 0
    st.session_state.chat_history_ids = None
    st.session_state.old_response = ''
else:
    st.session_state.count += 1

#Tokenizing the user input and returning the tensor output
new_user_input_ids = tokenizer.encode(input + tokenizer.eos_token, return_tensors='pt')

#Appending the user input ids to the chat history ids
bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_user_input_ids], dim=-1) if st.session_state.count > 1 else new_user_input_ids

#Generating a response while limiting the total chat history to 5000 tokens
st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=5000, pad_token_id=tokenizer.eos_token_id)

#Decoding the response
response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

#Regenerating the response if the old response from the model is the same as the current response.
if st.session_state.old_response == response:
    bot_input_ids = new_user_input_ids

    st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=5000,
                                                       pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
                                skip_special_tokens=True)

#Displaying the response on the UI
st.write(f"Chatbot: {response}")

#Updating the old_response variable
st.session_state.old_response = response


