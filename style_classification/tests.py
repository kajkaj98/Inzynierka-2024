import torch
from style_classification.pre_processing import preprocess_image_p1, preprocess_image_p2, resize_and_crop, pad_image_to_square, resize_and_crop_p3, preprocess_image_p3



def test_preprocess_image_p1():
    transformed_images = preprocess_image_p1('C:\\Users\\asiak\\PycharmProjects\\web_app\\static\\images\\mona-lisa.jpg')
    assert len(transformed_images) > 0
    for transformed_image in transformed_images:
        assert torch.is_tensor(transformed_image)


def test_preprocess_image_p2():
    transformed_image = preprocess_image_p2('C:\\Users\\asiak\\PycharmProjects\\web_app\\static\\images\\mona-lisa.jpg')
    assert torch.is_tensor(transformed_image)

def test_preprocess_image_p3():
    transformed_images = preprocess_image_p3('C:\\Users\\asiak\\PycharmProjects\\web_app\\static\\images\\mona-lisa.jpg')
    assert len(transformed_images) > 0
    for transformed_image in transformed_images:
        assert torch.is_tensor(transformed_image)


def test_resize_and_crop():

    list_of_images = resize_and_crop('C:\\Users\\asiak\\PycharmProjects\\web_app\\static\\images\\mona-lisa.jpg')

    assert len(list_of_images) > 0

    for image in list_of_images:
        width, height = image.size
        assert width == 224
        assert height == 224

def test_pad_to_square():

    image = pad_image_to_square('C:\\Users\\asiak\\PycharmProjects\\web_app\\static\\images\\mona-lisa.jpg')
    width, height = image.size

    assert width == 224
    assert height == 224

def test_resize_and_crop_3():

    list_of_images = resize_and_crop_p3('C:\\Users\\asiak\\PycharmProjects\\web_app\\static\\images\\mona-lisa.jpg')

    assert len(list_of_images) > 0

    for image in list_of_images:
        width, height = image.size
        assert width == 224
        assert height == 224

