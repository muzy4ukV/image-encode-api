import base64
import numpy as np
from annoy import AnnoyIndex

from .base import BaseDB
from fragment import Fragment


class MongoDB(BaseDB):
    def __init__(self, url='mongodb://172.20.240.1:27017/', db_name='fragments', collection_name='fragments'):
        print('Connecting to MongoDB...')
        #self.client = MongoClient(url)
        self.db = self.client[db_name]
        # self.db.drop_collection(collection_name) # Clear collection
        self.collection = self.db[collection_name]
        self.target_size = 16
        self.tree = None


    def is_empty(self):
        return self.collection.count_documents({}) == 0
    

    def build_tree(self):
        dims = np.frombuffer(self.collection.find_one({})['feature'], dtype=np.float32).shape[0]
        self.tree = AnnoyIndex(dims, 'angular')
        for fragment in self.collection.find():
            fragment_id = int(fragment['_id'])
            feature = np.frombuffer(fragment['feature'], dtype=np.float32)
            self.tree.add_item(fragment_id, feature)
        self.tree.build(10, n_jobs=-1)


    def add_fragment(self, fragment: Fragment):
        image_base64 = base64.b64encode(fragment.img.tobytes()).decode('utf-8')
        feature_bytes = fragment.feature.tobytes()

        fragment_doc = {
            '_id': self.collection.count_documents({}),
            'image': image_base64,
            'feature': feature_bytes
        }

        result = self.collection.insert_one(fragment_doc)

        return result.inserted_id
    

    def get_fragment_by_id(self, fragment_id: int):
        fragment = self.collection.find_one({'_id': int(fragment_id)})
        fragment['image'] = np.frombuffer(base64.b64decode(fragment['image']), dtype=np.uint8).reshape(self.target_size, self.target_size, 3)
        fragment['feature'] = np.frombuffer(fragment['feature'], dtype=np.float32)
        return fragment
        

    def find_similar_fragment_id(self, fragment_feature):
        if self.is_empty() or self.tree is None:
            self.build_tree()

        similar_fragment_id = self.tree.get_nns_by_vector(fragment_feature, 1)[0]
        return similar_fragment_id

