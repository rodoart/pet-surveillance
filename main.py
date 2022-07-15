from vigilancia_mascotas.data import make_dataset as make
from vigilancia_mascotas.utils.paths import is_valid

if __name__ == '__main__':
    dd = make.DataDownload()
    dd.unzip_files()
    dd.move_files()
    dd.correct_label_images()