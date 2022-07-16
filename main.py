from vigilancia_mascotas.utils.paths import TwoWorkspacePath

if __name__ == '__main__':

    #dd = make.DataDownload()
    REMOTE_PATH = '..\\remote_example'
    LOCAL_PATH = '.'

    twp = TwoWorkspacePath(
        'data', 'raw', 'semantic_segmentation', 'unity_residential_interiors', 'images', 'rgb_4.png' , 
        local_workspace = LOCAL_PATH,
        remote_workspace = REMOTE_PATH
    )
    twp_2 = twp.joinpath('hola')

    print(twp_2)