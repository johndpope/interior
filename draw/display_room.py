
import numpy as np
from PIL import Image
import os


cad_drawings = {
    0: Image.open('./draw/assets/2personsofa.png').convert('RGBA'),
    1: Image.open('./draw/assets/furnitureset.png').convert('RGBA'),
    2: Image.open('./draw/assets/1personsofa.png').convert('RGBA'),
    3: Image.open('./draw/assets/chair.png').convert('RGBA'),
    4: Image.open('./draw/assets/tvstand.png').convert('RGBA'),
    5: Image.open('./draw/assets/teatable.png').convert('RGBA'),
    6: Image.open('./draw/assets/sidetable.png').convert('RGBA'),
    7: Image.open('./draw/assets/dinningtableset.png').convert('RGBA'),
    8: Image.open('./draw/assets/cabinet.png').convert('RGBA'),
}

cad_length = [
    160,  # in cm
    300,
    90,
    50,
    180,
    156.3,
    74.4,
    210,
    100
]

color_map_with_furniture = np.transpose(np.array([[1., 0., 0., 0.5, 0.5, 0. , 0.25, 0.5, 0. , 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                                                  [0., 1., 0., 0.5, 0. , 0.5, 0.25, 0. , 0.5, 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                                                  [0., 0., 1., 0. , 0.5, 0.5, 0.25, 0. , 0. , 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                                                  ])
                                        )
color_map = np.transpose(np.array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                                   [1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                                   [1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                         ])
                         )
color_map = color_map * 255.0

ROOM_MAX = 1200
WIDTH = 32
CANVAS_SIZE = 256

EPS = 1e-3
def remove_eps(prob):
    return np.array([0 if p < EPS else p - EPS for p in prob])

def draw_color_layout(layout, save_path):
    room = layout
    old_shape = room.shape
    room = np.reshape(room, newshape=(-1, old_shape[-1]))
    room = np.matmul(room, color_map_with_furniture) * 255
    room = np.reshape(room, newshape=(old_shape[0], old_shape[1], 3)).astype('uint8')
    img = Image.fromarray(room, 'RGB')
    img = img.resize(size=(CANVAS_SIZE, CANVAS_SIZE))
    img.save(save_path)

def draw_cad_layout(layout, save_path, pixels_per_cm=WIDTH/ROOM_MAX, HARD=2, ROTATION_COUNT=8):
    old_shape = layout.shape
    room = np.copy(layout)
    room = np.reshape(room, newshape=(-1, old_shape[-1]))
    room = np.matmul(room, color_map)
    room = np.reshape(room, newshape=(old_shape[0], old_shape[1], 3)).astype('uint8')
    img = Image.fromarray(room, 'RGB')
    img = img.resize(size=(CANVAS_SIZE, CANVAS_SIZE))
    scale = CANVAS_SIZE / old_shape[0]
    model_code_cnt = old_shape[-1] - HARD - ROTATION_COUNT

    wall_category_code = [0] * (old_shape[-1] - ROTATION_COUNT)
    wall_category_code[-2] = 1

    space_category_code = [0] * (old_shape[-1] - ROTATION_COUNT)
    space_category_code[-1] = 1

    for h in range(old_shape[0]):
        for w in range(old_shape[0]):
            one_hot = layout[h][w]
            category_prob = remove_eps(one_hot[:old_shape[-1] - ROTATION_COUNT])
            category_one_hot = np.random.multinomial(1, category_prob)
            category_one_hot = category_one_hot.tolist()
            category_index = category_one_hot.index(1)

            if category_one_hot == wall_category_code or category_one_hot == space_category_code:
                continue

            rotation_prob = remove_eps(one_hot[old_shape[-1] - ROTATION_COUNT:])
            rotation_one_hot = np.random.multinomial(1, rotation_prob)
            rotation_one_hot = rotation_one_hot.tolist()

            rotation_index = rotation_one_hot.index(1)
            rotation = 45 * rotation_index

            target_size = cad_drawings[category_index].size
            target_size = (cad_length[category_index] * pixels_per_cm * scale, scale * target_size[1] * cad_length[category_index] * pixels_per_cm / target_size[0])

            furniture = cad_drawings[category_index].resize((int(target_size[0]), int(target_size[1]))).rotate(-rotation, expand=1)
            upper_left = (int(round(w * scale - furniture.size[0] * 0.5 + scale * 0.5)),
                          int(round(h * scale - furniture.size[1] * 0.5 + scale * 0.5)))
            img.paste(furniture, upper_left, furniture)
    img.save(save_path)


def check_file():
    design = np.load('output/06_45AM_on_June_21_2017\digits/00000454-90.npy')
    print(design.shape)


def main(path):
    data = np.load(path)
    data = np.squeeze(data, axis=0)
    draw_cad_layout(data, path + '.jpg')


def draw_np_cad(data, path):
    data = np.squeeze(data, axis=0)
    draw_cad_layout(data, path)


def draw_np_color(data, path):
    data = np.squeeze(data, axis=0)
    draw_color_layout(data, path)

