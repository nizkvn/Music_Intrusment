import pathlib
path = './templates'
data_root = pathlib.Path(path)
all_image_paths = list(data_root.glob('*'))
all_image_paths = [str(path).split('/')[-1] for path in all_image_paths]
print(all_image_paths)

