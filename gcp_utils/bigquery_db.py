import base64
from decorators import timing
import numpy as np
from annoy import AnnoyIndex
from google.cloud import bigquery
import os
import pandas as pd
from typing import Optional
from fragment import Fragment
from .credentials import CREDENTIALS
from .bucket_utils import GCSBucketUtils
from .label_generator import LabelGenerator

class BigQueryDB:
    def __init__(self, kernel_size: int = 16):
        super().__init__()
        # Завантажуємо облікові дані
        # Створюємо клієнта BigQuery
        self.client = bigquery.Client(credentials=CREDENTIALS, project=CREDENTIALS.project_id)
        self.target_size = kernel_size
        self.tree = None
        self.project_id = CREDENTIALS.project_id
        self.dataset_id = os.environ['GCP_DATASET_ID']
        self.table_id = os.environ['GCP_TABLE_ID']
        self.target_table = self.client.get_table(f"{self.project_id}.{self.dataset_id}.{self.table_id}")
        self.fragments = {}
        self.label_generator = None
        self.prepare_fragments()
        self.buffer_fragments_ids = []


    def is_empty(self):
        return len(self.fragments) == 0

    @timing("Time preparing fragments")
    def prepare_fragments(self):
        query = f"SELECT * FROM `{self.project_id}.{self.dataset_id}.{self.table_id}`"
        features_df = self.client.query(query).to_dataframe()

        if features_df.empty:
            print("No features found in DB.")
            return

        for i, row in features_df.iterrows():
            fragment_id = int(row['id'])
            fragment_doc = {
                'image': np.frombuffer(base64.b64decode(row['image']), dtype=np.uint8).reshape(self.target_size, self.target_size, 3),
                'feature': np.frombuffer(row['features'], dtype=np.float32)
            }
            self.fragments[fragment_id] = fragment_doc

        self.build_tree()

    def build_tree(self):
        features_dims = self.fragments[0]['feature'].shape[0]
        self.tree = AnnoyIndex(features_dims, 'euclidean')
        for i, fragment in self.fragments.items():
            self.tree.add_item(i, fragment['feature'])

        n_trees = min(100, max(10, int(np.log2(len(self.fragments))) * 5))
        self.tree.build(n_trees, n_jobs=-1)
        print("Tree was built")

    def add_fragments(self, fragments: list[Fragment]) -> Optional[str]:
        rows = []
        for fragment in fragments:
            rows.append({
                "id": len(self.fragments),
                "image": base64.b64encode(fragment.img.tobytes()).decode('utf-8'),
                "features": fragment.feature.tobytes()
            })
            self.add_fragment(fragment)

        fragments_df = pd.DataFrame(rows)
        fragments_df['id'] = fragments_df['id'].astype(int)

        try:
            job = self.client.load_table_from_dataframe(fragments_df, self.target_table)
            job.result()
            return "Data loaded successfully"

        except Exception as e:
            print(f"Error occurred while adding fragments to BigQuery: {e}")
            raise e

    def add_fragment(self, fragment: Fragment):
        fragment_elem = {
            'image': fragment.img,
            'feature': fragment.feature
        }
        fragment_id = len(self.fragments)
        self.fragments[fragment_id] = fragment_elem
        self.buffer_fragments_ids.append(fragment_id)
        return fragment_id

    @timing("Time updating fragments")
    def update_fragments(self):
        if len(self.buffer_fragments_ids) == 0:
            return
        else:
            fragment_to_add = []
            for i, fragment_id in enumerate(self.buffer_fragments_ids):
                fragment = self.fragments[fragment_id]
                fragment_to_add.append({
                    "id": fragment_id,
                    "image": base64.b64encode(fragment['image'].tobytes()).decode('utf-8'),
                    "features": fragment['feature'].tobytes()
                })
            fragments_df = pd.DataFrame(fragment_to_add)
            try:
                job = self.client.load_table_from_dataframe(fragments_df, self.target_table)
                job.result()
                print("Updated fragments loaded successfully")
                self.buffer_fragments_ids = []
            except Exception as e:
                print(f"Error occurred while adding fragments to BigQuery: {e}")
            self.build_tree()

    def get_db_size(self):
        return len(self.fragments)

    def get_image_by_id(self, fragment_id: int):
        fragment = self.fragments[fragment_id]
        return fragment['image']

    def find_similar_fragment_id(self, fragment_feature):
        similar_fragment_id = self.tree.get_nns_by_vector(fragment_feature, 1)[0]
        return similar_fragment_id
