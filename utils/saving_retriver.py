import pandas as pd
from ranking import MyExistingRetrievalPipeline

def name():
    data = pd.read_csv(path_to_csv_data)

    existing_pipeline = MyExistingRetrievalPipeline()
    existing_pipeline.index_documents(data, batch_size=1024)

    index_44 = './indexes/index_44.index'
    collection_44 = './indexes/collection_44.pkl'

    index_223 = './indexes/index_223.index'
    collection_223 = './collection_223.pkl'

    index_others = './indexes/index_others.index'
    collection_others = './indexes/collection_ohers.pkl'

    existing_pipeline.save_index(index_44, collection_44, index_223, collection_223, index_others, collection_others)

if __name__ == "__main__":
    main()