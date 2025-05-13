import numpy as np
from annoy import AnnoyIndex
from google.cloud import bigquery
from google.oauth2 import service_account
import os
import pandas as pd
import cv2
from typing import Optional
from time import time
from fragment import Fragment
from .base import BaseDB
from .bucket_utils import GCSBucketUtils
from .label_generator import LabelGenerator

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
        self.init_label_generator()
        self.build_tree()

    def init_label_generator(self):
        query = (f"SELECT COALESCE(MAX(id), 0) as max_db_label "
                 f"FROM `{self.project_id}.{self.dataset_id}.{self.table_id}`")
        result = self.client.query(query).result()
        row = next(result)
        self.label_generator = LabelGenerator(row["max_db_label"])

    def is_empty(self):
        return self.tree is None

    def build_tree(self):
        start_time = time()
        query = f"SELECT * FROM `{self.project_id}.{self.dataset_id}.{self.table_id}`"
        features_df = self.client.query(query).to_dataframe()

        features_dims = np.frombuffer(features_df.iloc[0]['features'], dtype=np.float32).shape[0]
        self.tree = AnnoyIndex(features_dims, 'angular')
        features_df.apply(
            lambda row: self.tree.add_item(row['id'], np.frombuffer(row['features'], dtype=np.float32)), axis=1
        )
        self.tree.build(10, n_jobs=-1)
        print("Time building tree: ", time() - start_time)


    def add_fragments(self, fragments: list[Fragment]) -> Optional[str]:
        rows = []
        for fragment in fragments:
            img_label = self.label_generator.generate()
            self.bucket.add_fragment(fragment.img, img_label)
            rows.append({
                "id": img_label,
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
        return self.add_fragments([fragment])

    def get_db_size(self):
        try:
            query = f"SELECT COUNT(*) as row_count FROM `{self.project_id}.{self.dataset_id}.{self.table_id}`"
            result = self.client.query(query).result()
            row = next(result)
            return row["row_count"]
        except Exception as e:
            print(f"Error while querying row count: {e}")
            return -1

    def find_similar_fragment_id(self, fragment_feature):
        similar_fragment_id = self.tree.get_nns_by_vector(fragment_feature, 1)[0]
        return similar_fragment_id

    def get_fragment_by_id(self, fragment_id: int):
        try:
            img_bytes = self.bucket.get_fragment(fragment_id)
            img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        except:
            raise ValueError(f"Cannot decode the {fragment_id}.png fragment")
        return img
