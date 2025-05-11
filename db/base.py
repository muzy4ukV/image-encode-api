from fragment import Fragment


class BaseDB:
    def __init__(self):
        pass

    def is_empty(self):
        pass

    def build_tree(self):
        pass

    def add_fragment(self, fragment: Fragment):
        pass

    def get_fragment_by_id(self, fragment_id: int):
        pass

    def find_similar_fragment_id(self, fragment_feature):
        pass

    def save_results(self, path_to_save=None):
        pass