from regex import D
from vigilancia_mascotas.data.make_dataset import DataDownload

dd = DataDownload()
#dd.unzip_files()
dd.move_files()
dd.correct_label_images()
dd.move_files_to_train_and_validation_folders()



