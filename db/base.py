from fragment import Fragment


class BaseDB:
    def __init__(self):
        self.fragments = {}

    def is_empty(self):
        pass

    def build_tree(self):
        pass

    def add_fragment(self, fragment: Fragment):
        pass

    def add_fragments(self, fragments: list[Fragment]):
        pass

    def get_image_by_id(self, fragment_id: int):
        pass

    def find_similar_fragment_id(self, fragment_feature):
        pass

    def save_results(self, path_to_save=None):
        pass

    def get_db_size(self):
        pass