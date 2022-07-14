from vigilancia_mascotas.data import make_dataset as make
from vigilancia_mascotas.utils.paths import is_valid

if __name__ == '__main__':
    dd = make.DataDownload()
    dd.move_files_to_train_and_validation_folders()