from utils.ranking import  *


def get_documents(query, selected_law, existing_pipeline, RAG):
    
#     if selected_law == '223-ФЗ':
#         query = '223-ФЗ 223 ФЗ 223 федеральный закон' + query
#     elif selected_law == '43-ФЗ':
#         query = '43-ФЗ 43 ФЗ 43 федеральный закон' + query
        
    raw_results = existing_pipeline.query(selected_law, query, k=20)
    
#     documents_as_strings = [i['doc_id'] + ' ' + i['content'] for i in raw_results]    
    documents_as_strings = [i['content'] for i in raw_results] # поиск делаем только по пассажу
    contents = RAG.rerank(query=query, documents=documents_as_strings, k=5)
    context1 = contents[0]['content']
    context2 = contents[1]['content']
    context3 = contents[2]['content']
    
    prompt = f"""
        Вопрос:{query}
        Контекст:
        '{context1}',
        '{context2}',
        '{context3}'
        
        Твой ответ:
    """
    return prompt, context1, context2, context3 