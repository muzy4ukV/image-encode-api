from decorators import timing
import numpy as np
from annoy import AnnoyIndex
from google.cloud import bigquery
import os
import cv2
import pandas as pd
from typing import Optional, List
from fragment import Fragment
from .credentials import CREDENTIALS
from .bucket_utils import GCSBucketUtils
from .fragments_storage import FragmentsStorage


class BigQueryDB:
    TABLE_CONFIG = bigquery.LoadJobConfig(
        schema=[
            bigquery.SchemaField("id", "INTEGER", "REQUIRED"),
            bigquery.SchemaField("image", "BYTES", "REQUIRED"),  # <== ключове
            bigquery.SchemaField("features", "BYTES", "REQUIRED")
        ]
    )

    def __init__(self):
        super().__init__()
        # Завантажуємо облікові дані
        # Створюємо клієнта BigQuery
        self.client = bigquery.Client(credentials=CREDENTIALS, project=CREDENTIALS.project_id)
        self.tree = None
        self.project_id = CREDENTIALS.project_id
        self.dataset_id = os.environ['GCP_DATASET_ID']
        self.table_id = os.environ['GCP_TABLE_ID']
        self.target_table = self.client.get_table(f"{self.project_id}.{self.dataset_id}.{self.table_id}")
        self.storage = FragmentsStorage()
        self.label_generator = None
        self.prepare_fragments()
        self.bucket_storage = GCSBucketUtils()
        self.buffer_fragments_ids = []

    def is_empty(self):
        return len(self.storage) == 0

    @timing("Time preparing fragments")
    def prepare_fragments(self):
        query = f"SELECT * FROM `{self.project_id}.{self.dataset_id}.{self.table_id}`"
        features_df = self.client.query(query).to_dataframe()

        if features_df.empty:
            print("No features found in DB.")
            return

        for i, row in features_df.iterrows():
            fragment_id = int(row['id'])

            # Декодуємо PNG байти (row['image'] — це bytes)
            image = cv2.imdecode(np.frombuffer(row['image'], dtype=np.uint8), cv2.IMREAD_COLOR)
            # Приводимо до RGB (якщо потрібно)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            try:
                self.storage.add_fragment_with_id(fragment_id, image,
                                                  np.frombuffer(row['features'], dtype=np.float32))
            except ValueError as e:
                print(e)

        self.build_tree()

    def build_tree(self):
        features_dims = self.storage.get_random_fragment().features.shape[0]
        self.tree = AnnoyIndex(features_dims, 'euclidean')
        for i, fragment in self.storage.items():
            self.tree.add_item(i, fragment.features)

        n_trees = min(100, max(10, int(np.log2(len(self.storage))) * 5))
        self.tree.build(n_trees, n_jobs=-1)
        print("Tree was built")

    def add_fragments(self, fragments: list[Fragment]) -> Optional[str]:
        if len(fragments) == 0:
            return "No fragments to add into base"
        rows = []
        for fragment in fragments:
            fragment_id = self.add_fragment(fragment)
            rows.append({
                "id": fragment_id,
                "image": BigQueryDB.compress_nparr_to_bytes(fragment.image),
                "features": fragment.features.tobytes()
            })

        fragments_df = pd.DataFrame(rows)
        fragments_df['id'] = fragments_df['id'].astype(int)

        try:
            job = self.client.load_table_from_dataframe(fragments_df, self.target_table,
                                                        job_config=BigQueryDB.TABLE_CONFIG)
            job.result()
            return "Successfully added fragments into base"

        except Exception as e:
            print(f"Error occurred while adding fragments to BigQuery: {e}")
            raise e

    def add_fragment(self, fragment: Fragment, flag: bool = False):
        new_fragment_id = self.storage.add_fragment(fragment.image, fragment.features)
        if flag:
            self.buffer_fragments_ids.append(new_fragment_id)
        return new_fragment_id

    @timing("Time updating fragments")
    def update_fragments(self):
        if len(self.buffer_fragments_ids) == 0:
            return
        else:
            fragment_to_add = []
            for fragment_id in self.buffer_fragments_ids:
                fragment = self.storage.get_fragment(fragment_id)
                fragment_to_add.append({
                    "id": fragment_id,
                    "image": BigQueryDB.compress_nparr_to_bytes(fragment.image),
                    "features": fragment.features.tobytes()
                })
            fragments_df = pd.DataFrame(fragment_to_add)
            try:
                job = self.client.load_table_from_dataframe(fragments_df, self.target_table,
                                                            job_config=BigQueryDB.TABLE_CONFIG)
                job.result()
                print("Updated fragments loaded successfully")
                self.buffer_fragments_ids = []
            except Exception as e:
                print(f"Error occurred while adding fragments to BigQuery: {e}")

    def get_db_size(self) -> int:
        return self.storage.get_fragments_count()

    def get_fragment_by_id(self, fragment_id: int) -> Fragment:
        return self.storage.get_fragment(fragment_id)

    def find_similar_fragment_id(self, fragment_feature) -> int:
        similar_fragment_id = self.tree.get_nns_by_vector(fragment_feature, 1)[0]
        return similar_fragment_id

    def find_k_similar_fragments(self, fragment_feature, k) -> List[Fragment]:
        similar_fragment_ids = self.tree.get_nns_by_vector(fragment_feature, k)
        return [self.storage.get_fragment(fr_id) for fr_id in similar_fragment_ids]

    @staticmethod
    def compress_nparr_to_bytes(nparr) -> bytes:
        # PNG-стиснення
        _, encoded_png = cv2.imencode('.jpg', cv2.cvtColor(nparr, cv2.COLOR_RGB2BGR))
        return encoded_png.tobytes()

    def get_fragments_signed_url(self, fragments_base_name: str) -> str:
        return self.bucket_storage.get_signed_url(fragments_base_name)

    def upload_new_fragments_base(self):
        return self.bucket_storage.add_fragments_to_gcs(self.storage)
