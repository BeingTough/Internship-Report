from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='constant')

i = 0
for batch in datagen.flow_from_directory('/home/kshuai/Dokumente/Augmentation/', batch_size=1,
                          save_to_dir='/home/kshuai/Dokumente/Augmentation/', save_prefix='cat', save_format='jpeg'):
    i += 1
    if i >= 20:
        break  # otherwise the generator would loop indefinitely
