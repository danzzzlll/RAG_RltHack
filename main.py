import utils.generatives
import os
from utils.get_prompt import get_documents
from utils.config import GenerationConfig
from utils.ranking import MyExistingRetrievalPipeline
from ragatouille import RAGPretrainedModel
import streamlit as st

@st.cache_resource
def load_model():
    mistral_path = GenerationConfig.mistral_path
    mistral = utils.generatives.Mistral("Mistral", mistral_path)
    giga = utils.generatives.GigaApi("gigachat")
    
    pipeline = MyExistingRetrievalPipeline()
    pipeline.load_index(GenerationConfig.index_44, GenerationConfig.collection_44,
                        GenerationConfig.index_223, GenerationConfig.collection_223,
                        GenerationConfig.index_others, GenerationConfig.collection_others  
                    )

    RAG = RAGPretrainedModel.from_pretrained("antoinelouis/colbert-xm")
    
    return mistral, giga, pipeline, RAG


def get_answer(query, selected_law, pipeline, RAG, mistral, model_type):
    if model_type == 'giga':
        mistral.update_system_prompt(system_prompt=GenerationConfig.giga_prompt)
    else:
        mistral.update_system_prompt(system_prompt=GenerationConfig.mistral_prompt)   

    prompt, context1, context2, context3 = get_documents(query, selected_law, pipeline, RAG)
    
    answer = mistral.inference(prompt, max_new_tokens=1000)

    return answer, context1, context2, context3


if 'history' not in st.session_state:
    st.session_state['history'] = []

        
st.title("Консультант государственных закупок")
        
mistral, giga, pipeline, RAG = load_model()

law_options = ["223-ФЗ", "44-ФЗ", "Не известно/оба"]
selected_law = st.selectbox("Выберите контекст закона:", law_options)

user_question = st.text_input("Введите свой запрос:", "")


if st.button('Отправить'):
    if user_question:
        if not selected_law:
            st.error("Пожалуйста, выберите по какому закону Вы хотите задать вопрос")
        else:
            answer_giga, context1, context2, context3 = get_answer(user_question, selected_law, pipeline, RAG, giga, 'giga')
            
            st.subheader("Вопрос")
            st.write(user_question)
            st.subheader("Ответ GigaChat Pro")
            st.write(answer_giga)
            answer_mistral, context1, context2, context3 = get_answer(user_question, selected_law, pipeline, RAG, mistral, 'Mistral')
            st.subheader("Использованные отрывки документов")
            
            st.markdown(f"""
            <div style="margin-bottom: 20px;">
                <div style="background-color: #333; padding: 10px; border-radius: 5px; color: white;">{context1}</div>
            </div>
            <div style="margin-bottom: 20px;">
                <div style="background-color: #333; padding: 10px; border-radius: 5px; color: white;">{context2}</div>
            </div>
            <div style="margin-bottom: 20px;">
                <div style="background-color: #333; padding: 10px; border-radius: 5px; color: white;">{context3}</div>
            </div>
            """, unsafe_allow_html=True)
            
            context = f"{context1}\n\n{context2}\n\n{context3}"

            st.session_state['history'].append((user_question, answer, context))
    else:
        st.error("Пожалуйста, введите вопрос")

        
st.sidebar.title("История запросов")
for idx, (question, answer, context) in enumerate(reversed(st.session_state['history']), start=1):
    with st.sidebar.expander(f"Вопрос {idx}: {question}"):
        st.write(f"**Ответ:** {answer}")
        st.write(f"**Контекст:** {context}") 