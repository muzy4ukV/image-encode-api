

class Fragment:
    def __init__(self, img = None, x = None, y = None, features = None):
        self.image = img
        self.features = features
        self.x = x
        self.y = y

    def to_dict(self):
        return {
            'image': self.image,
            'features': self.features,
            'x': self.x,
            'y': self.y
        }