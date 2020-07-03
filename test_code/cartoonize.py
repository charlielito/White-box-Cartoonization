import os
import cv2
import numpy as np
import tensorflow as tf
import network
import guided_filter
from tqdm import tqdm
from pathlib import Path
import imageio


def bytes2video(videobytes):
    with imageio.get_reader(videobytes, "ffmpeg") as reader:
        for image in reader:
            yield image


def read_fn(filepath):
    with tf.io.gfile.GFile(filepath, "rb") as f:
        return f.read()


def read_video(path):
    return bytes2video(read_fn(path))


def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720 * h / w), 720
        else:
            h, w = 720, int(720 * w / h)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    h, w = (h // 8) * 8, (w // 8) * 8
    image = image[:h, :w, :]
    return image


def cartoonize(load_folder, save_folder, model_path):
    input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
    network_out = network.unet_generator(input_photo)
    final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)

    all_vars = tf.trainable_variables()
    gene_vars = [var for var in all_vars if "generator" in var.name]
    saver = tf.train.Saver(var_list=gene_vars)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(model_path))

    video_file = Path(load_folder).is_file()
    # check if it is a video
    if video_file:
        name_list = read_video(load_folder)
    else:
        name_list = os.listdir(load_folder)

    for i, name in enumerate(tqdm(name_list)):
        # try:
        if video_file:
            image = name
            save_path = os.path.join(save_folder, f"frame{i:05d}.png")
        else:
            load_path = os.path.join(load_folder, name)
            save_path = os.path.join(save_folder, name)
            image = cv2.imread(load_path)
        image = resize_crop(image)
        batch_image = image.astype(np.float32) / 127.5 - 1
        batch_image = np.expand_dims(batch_image, axis=0)
        output = sess.run(final_out, feed_dict={input_photo: batch_image})
        output = (np.squeeze(output) + 1) * 127.5
        output = np.clip(output, 0, 255).astype(np.uint8)
        cv2.imwrite(save_path, output)
        # except:
        #     print("cartoonize {} failed".format(load_path))


if __name__ == "__main__":
    model_path = "saved_models"
    load_folder = "test_images"
    load_folder = "video.mp4"
    save_folder = "cartoonized_images"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    cartoonize(load_folder, save_folder, model_path)

# ffmpeg -r 1 -i LORT.mp4 -r 1 "$filename%03d.png"
# conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge
# ffmpeg -framerate 25 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p out.mp4
