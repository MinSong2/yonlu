from elasticsearch import Elasticsearch, helpers
from datetime import datetime
import pandas as pd

class ElasticSearchManager():
    def __init__(self):
        # Create the client instance
        self.client = Elasticsearch("http://localhost:9200")

        # Successful response!
        print(self.client.info())

    def _generator(self, df2, index_name, field1, field2):
        for c, line in enumerate(df2):
            yield {
                '_index': index_name,
                '_id': str(c),
                '_source': {
                    field1:line.get(field1,""),
                    field2: line.get(field2,""),
                }
            }

    def index(self, file_name='', index_name = '', field1 = '', field2=''):
        df = pd.read_csv(file_name)
        df = df.dropna()

        df2 = df.to_dict("records")
        try:
            res = helpers.bulk(self.client, self._generator(df2, index_name, field1, field2))
            print("Response", res)
        except Exception as ex:
            print(ex)
            pass

    def search(self, index='', field='Q', query='', size=20):
        resp = self.client.search(index=index, body={'from': 0, 'size': size, 'query': {'match': {field: query}}})

        return resp['hits']['hits'], resp['hits']['total']['value']

    def refresh(self, index_name):
        self.client.indices.refresh(index=index_name)

    def update(self, index='', id=1, document=None):
        self.client.update(index=index, id=id, doc=document)

    def delete(self, index_name):
        self.client.delete(index=index_name, id=1)

if __name__ == "__main__":
    manager = ElasticSearchManager()
    file_name = '../data/ChatbotData.csv'
    field1 = 'Q'
    field2 = 'A'
    index = 'chatbot'
    #manager.index(file_name=file_name, index_name=index, field1=field1,field2=field2)
    query = '노트북'
    results = manager.search(index=index, field='Q', query=query, size=20)
    # print("Got %d Hits:" % resp['hits']['total']['value'])
    for hit in results:
        print('score:', hit['_score'], 'source:', hit["_source"])

    #manager.delete(index)