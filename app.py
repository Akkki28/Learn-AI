import streamlit as st
import torch
from transformers import BertForQuestionAnswering, BertTokenizer
from textwrap3 import wrap
import nltk
import spacy
from spacy.util import is_package
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import string
import pke
import traceback
from flashtext import KeywordProcessor
from transformers import T5ForConditionalGeneration, T5Tokenizer
from functions import load_model_summary, load_model_question, nltk_download, postprocesstext, summarizer, get_nouns_multipartite, get_keywords, get_question, summarize_text, get_answer

st.markdown(
    """
    <style>
    .entity-person {
        background-color: red;
        color: black;
        border-radius: 5px;
    }
    .entity-date {
        background-color: blue;
        color: black;
        border-radius: 5px;
    }
    .entity-org {
        background-color: green;
        color: black;
        border-radius: 5px;
    }
    .entity-other {
        background-color: yellow;
        color: black;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

nltk_download()
summary_model, summary_tokenizer, device = load_model_summary()
question_model, question_tokenizer, device = load_model_question()
if not is_package("en_core_web_sm"):
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
nlp = spacy.load("en_core_web_sm")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
doubt_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

st.markdown("<h1 style='text-align: center; font-size: 64px;'>LearnAI</h1>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; color: blue;'>Learn efficiently with artificial intelligence</h6>", unsafe_allow_html=True)

if 'text' not in st.session_state:
    st.session_state.text = ""
if 'summarized_text' not in st.session_state:
    st.session_state.summarized_text = ""
if 'questions' not in st.session_state:
    st.session_state.questions = []

text_input = st.text_area("Enter the material you want to study:", st.session_state.text)
if st.button("Analyze"):
    st.session_state.text = text_input
    text = text_input
    num_sentences = len(sent_tokenize(text))
    num_sentences = (int)(num_sentences / 3)
    response = summarize_text(text, num_sentences=num_sentences)
    if isinstance(response, dict) and 'summary' in response:
        st.session_state.summarized_text = response['summary']
    else:
        st.session_state.summarized_text = response

    st.session_state.questions = []
    summarized_sentences = sent_tokenize(st.session_state.summarized_text)
    imp_keywords = get_keywords(text, st.session_state.summarized_text, device)
    final_imp = []
    for sentence in summarized_sentences:
        doc = nlp(sentence)
        highlighted_sentence = sentence
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                highlighted_sentence = highlighted_sentence.replace(ent.text, f"<span class='entity-person'>{ent.text}</span>")
            elif ent.label_ == "DATE":
                highlighted_sentence = highlighted_sentence.replace(ent.text, f"<span class='entity-date'>{ent.text}</span>")
            elif ent.label_ == "ORG":
                highlighted_sentence = highlighted_sentence.replace(ent.text, f"<span class='entity-org'>{ent.text}</span>")
            else:
                highlighted_sentence = highlighted_sentence.replace(ent.text, f"<span class='entity-other'>{ent.text}</span>")
        for token in doc:
            if token.dep_ == "nsubj":
                if token.text.lower() in imp_keywords and token.text.lower() not in final_imp:
                    final_imp.append(token.text.lower())
            if token.pos_ == "PROPN":
                if token.text.lower() in imp_keywords and token.text.lower() not in final_imp:
                    final_imp.append(token.text.lower())

    for answer in final_imp:
        ques = get_question(st.session_state.summarized_text, answer, question_model, question_tokenizer, device)
        st.session_state.questions.append({"question": ques, "answer": answer})

if st.session_state.summarized_text:
    st.title("Key points:")
    summarized_sentences = sent_tokenize(st.session_state.summarized_text)
    for sentence in summarized_sentences:
        doc = nlp(sentence)
        highlighted_sentence = sentence
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                highlighted_sentence = highlighted_sentence.replace(ent.text, f"<span class='entity-person'>{ent.text}</span>")
            elif ent.label_ == "DATE":
                highlighted_sentence = highlighted_sentence.replace(ent.text, f"<span class='entity-date'>{ent.text}</span>")
            elif ent.label_ == "ORG":
                highlighted_sentence = highlighted_sentence.replace(ent.text, f"<span class='entity-org'>{ent.text}</span>")
            else:
                highlighted_sentence = highlighted_sentence.replace(ent.text, f"<span class='entity-other'>{ent.text}</span>")
        st.markdown(f"- {highlighted_sentence}", unsafe_allow_html=True)

if st.session_state.questions:
    st.title("Q/A:")
    for qa in st.session_state.questions:
        with st.expander(f"Question: {qa['question']}"):
            st.write(f"Answer: {qa['answer']}")

st.title("Doubt Solver:")
doubt = st.text_area("Enter your doubt:")
if st.button("Submit Doubt"):
    answer = get_answer(doubt, st.session_state.text, tokenizer, doubt_model)
    st.write("Answer:", answer)
