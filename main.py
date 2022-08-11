from pet_surveillance.models import segformer


def run():
    obe = segformer.Segformer()
    obe.predict_labels('data/processed/semantic_segmentation/unity_residential_interiors/train_images/7.png')

if __name__ == '__main__':
    print(run())



