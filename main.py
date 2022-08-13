from pet_surveillance.models import segformer


def run():
    obe = segformer.Segformer()
    img = obe.predict_labels('data/processed/semantic_segmentation/unity_residential_interiors/train_images/7.png', 'tmp/images/7.png')

    print(obe.detect_floor(img))

if __name__ == '__main__':
    run()



