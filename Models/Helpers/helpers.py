import json, cv2, random
import numpy as np
from os import listdir
from os.path import isfile, join


def get_image_data(detect_name: str, image_dir: str, data_dir: str, test_pct: float = 0.0,
                 valid_pct: float = 0.0, use_grayscale=False, dataset='pokemonDataSet',
                 normalize=True):
    """
    Fetches all data and automatically splits it into training/test/validation.
    """

    assert test_pct + valid_pct < 1.0, 'Make sure test + validation < 1.0 \
                                           so that some data remains for training.'
    assert test_pct >= 0 and valid_pct >= 0, 'You must enter positive percent values.'

    train_pct = 1.0 - test_pct - valid_pct

    image_names = listdir(join(image_dir, '.')) # Get list of all image names
    num_images = len(image_names)
    random.shuffle(image_names) # Randomly shuffle image names

    # Load image data
    image_data_file = open(data_dir)
    image_data = json.load(image_data_file)

    # Initialize lists that will store train/test/validation data
    x_train = []
    y_train = []
    x_test  = []
    y_test  = []
    x_valid = []
    y_valid = []

    # Get training data
    for image_index in range(int(num_images * train_pct)):
        if (    use_grayscale and '_g'     in image_names[image_index]) or \
           (not use_grayscale and '_g' not in image_names[image_index]):
            # Append imported image value (converted to numpy array)
            x_train.append(np.array(cv2.imread(
                join(image_dir, image_names[image_index]),
                0 if use_grayscale else 1)) / 255.0 if normalize else 1.0)

            # Check if image data name matches desired name
            y_train.append(image_data[dataset][image_names[image_index]]['name'] == detect_name)

    test_index = len(x_train)

    # Get testing data
    for image_index in range(test_index, test_index + int(num_images * test_pct)):
        if (    use_grayscale and '_g'     in image_names[image_index]) or \
           (not use_grayscale and '_g' not in image_names[image_index]):
            # Append imported image value (converted to numpy array)
            x_test.append(np.array(cv2.imread(
                join(image_dir, image_names[image_index]),
                0 if use_grayscale else 1)))

            # Check if image data name matches desired name
            y_test.append(image_data[dataset][image_names[image_index]]['name'] == detect_name)

    valid_index = len(x_train) + len(x_test)

    # Get validation data
    for image_index in range(valid_index, valid_index + int(num_images * valid_pct)):
        if (    use_grayscale and '_g'     in image_names[image_index]) or \
           (not use_grayscale and '_g' not in image_names[image_index]):
            # Append imported image value (converted to numpy array)
            x_valid.append(np.array(cv2.imread(
                join(image_dir, image_names[image_index]),
                0 if use_grayscale else 1)))

            # Check if image data name matches desired name
            y_valid.append(image_data[dataset][image_names[image_index]]['name'] == detect_name)

    return np.array(x_train), np.array(y_train), \
           np.array(x_test), np.array(y_test),   \
           np.array(x_valid), np.array(y_valid)


def get_single_image(image_dir: str, use_grayscale: bool=False, normalize: bool=True):
    """
    Returns single image for inference. Image is placed within array due to model input requirements.
    """
    loaded_image = np.array([np.array(cv2.imread(image_dir, 0 if use_grayscale else 1) / 255.0 if normalize else 1.0)]).reshape(-1, 270, 480, 1)
    return loaded_image

def grayscale(input_dir: str, filename: str):
    """Converts specific RGB image to grayscale."""
    image = cv2.imread(input_dir + '/' + filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(input_dir + '/' + filename + '_g.png', gray)


def grayscale_all(input_dir: str):
    """
    Converts all RGB images in given directory to grayscale.
    """
    poses = listdir(input_dir + '/.')
    for pose in poses:
        images = listdir(join(input_dir, pose + '/.'))
        for img in images:
            if '_g' not in img:
                image = cv2.imread(join(input_dir, pose, img), 1)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(join(input_dir, pose, img.split('.')[0] + '_g.png'), gray)


def extract_data_from_images(images_dir: str, data_dir: str):
    """
    Creates and populates JSON data file based on all existing grayscale images.
    """
    open(data_dir, 'w').close()
    images_data = {}

    # Loop through images
    poses = listdir(images_dir + '/.')
    for pose in poses:
        images = listdir(join(images_dir, pose + '/.'))

        # Only extract from grayscale images
        for img in images:
            if '_g' in img:
                img_name_split = img.split("_")
                img_data = {}
                img_data["name"] = img_name_split[0]
                img_data["pose"] = int(img_name_split[1][4:])
                img_data["cameraPosition"] = int(img_name_split[2])
                img_data["angle"] = int(img_name_split[3])

                images_data[img] = img_data

    # Write data to JSON file
    with open(data_dir, 'w') as fp:
        json.dump({"pokemonDataSet": images_data}, fp)


# Command for turning all non-grayscale images into grayscale images:
# grayscale_all('RhydonMate/Data/UE4')

# Command for re-generating JSON data based on all new images:
# extract_data_from_images('RhydonMate/Data/UE4', 'RhydonMate/Data/image_data.json')
