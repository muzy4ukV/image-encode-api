import numpy as np
from annoy import AnnoyIndex
from google.cloud import bigquery
from google.oauth2 import service_account
import os
import pandas as pd
from typing import Optional
from time import time
from fragment import Fragment
from .base import BaseDB
from .bucket_utils import GCSBucketUtils


class BigQueryDB(BaseDB):
    def __init__(self):
        super().__init__()
        # Завантажуємо облікові дані
        credentials = service_account.Credentials.from_service_account_file(os.environ['GOOGLE_CREDENTIALS_PATH'])
        # Створюємо клієнта BigQuery
        self.client = bigquery.Client(credentials=credentials, project=credentials.project_id)
        self.target_size = 16
        self.tree = None
        self.project_id = credentials.project_id
        self.dataset_id = os.environ['GCP_DATASET_ID']
        self.table_id = os.environ['GCP_TABLE_ID']
        self.target_table = self.client.get_table(f"{self.project_id}.{self.dataset_id}.{self.table_id}")
        self.bucket = GCSBucketUtils()
        self.label_generator = None
        self.fragments = {}
        self.build_tree()


    def is_empty(self):
        return len(self.fragments) == 0

    def build_tree(self):
        start_time = time()
        query = f"SELECT * FROM `{self.project_id}.{self.dataset_id}.{self.table_id}`"
        features_df = self.client.query(query).to_dataframe()

        if features_df.empty:
            print("No features found in DB.")
            return

        features_dims = np.frombuffer(features_df.iloc[0]['features'], dtype=np.float32).shape[0]

        for i, row in features_df.iterrows():
            fragment_doc = {
                'image': np.frombuffer(row['image'], dtype=np.float32),
                'feature': np.frombuffer(row['features'], dtype=np.float32)
            }
            self.fragments[i] = fragment_doc

        self.tree = AnnoyIndex(features_dims, 'euclidean')
        for i, fragment in self.fragments.items():
            self.tree.add_item(i, fragment['feature'])

        self.tree.build(100, n_jobs=-1)
        print("Time building tree: ", time() - start_time)


    def add_fragments(self, fragments: list[Fragment]) -> Optional[str]:
        rows = []
        for fragment in fragments:
            rows.append({
                "image": fragment.img.tobytes(),
                "features": fragment.feature.tobytes()
            })

        fragments_df = pd.DataFrame(rows)

        try:
            job = self.client.load_table_from_dataframe(fragments_df, self.target_table)
            job.result()
            return "Data loaded successfully"

        except Exception as e:
            print(f"Error occurred while adding fragments to BigQuery: {e}")
            return None

    def add_fragment(self, fragment: Fragment):
        fragment_elem = {
            'image': fragment.img,
            'feature': fragment.feature
        }
        fragment_id = len(self.fragments)
        self.fragments[fragment_id] = fragment_elem
        return fragment_id

    def get_db_size(self):
        return len(self.fragments)

    def get_fragment_by_id(self, fragment_id: int):
        fragment = self.fragments[fragment_id]
        return fragment

    def find_similar_fragment_id(self, fragment_feature):
        similar_fragment_id = self.tree.get_nns_by_vector(fragment_feature, 1)[0]
        return similar_fragment_id
