from PIL import Image

class Predictor():
    @classmethod
    def get_model(cls, model_path):
        cls.model = None
        cls.debug = False
        return True


    @classmethod
    def predict(cls, v):
        if cls.debug:
            print(v)
        img = Image.open(v)
        pred = '1234567890'

        return pred

    def test():
        return "test"

if __name__ == '__main__':
    pass