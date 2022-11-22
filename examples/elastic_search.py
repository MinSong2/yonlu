from yonlu.semantic_search.elasticsearch_manager import ElasticSearchManager

mode = 'refresh' #index, search, update, delete, refresh,
manager = ElasticSearchManager()
file_name = '../data/ChatbotData.csv'
field1 = 'Q'
field2 = 'A'
index = 'chatbot'
if mode == 'index':
    manager.index(file_name=file_name, index_name=index, field1=field1,field2=field2)
elif mode == 'search':
    query = '노트북'
    results, hits = manager.search(index=index, field='Q', query=query, size=20)
    print("Got %d Hits:" % hits)
    for hit in results:
        print('score:', hit['_score'], 'source:', hit["_source"])
elif mode == 'delete':
    manager.delete(index)
elif mode == 'update':
    doc = {
            'Q': '오늘은 어땠어?',
            'A': '오전 수업이라 졸립고 힘들어..',
    }
    resp = manager.update(index=index, id=1, document=doc)
elif mode == 'refresh':
    manager.refresh(index)