import argparse
import os

import cv2
import time

import tensorflow as tf
import tensorflow_hub as hub


def main(args):
    if args.i:
        model = hub.load('https://tfhub.dev/captain-pool/esrgan-tf2/1')
        imgraw = tf.io.read_file(args.i)

        img = tf.image.decode_image(imgraw)
        img = tf.cast(img, tf.float32)

        output = model(tf.expand_dims(img, 0))

        highres = tf.cast(tf.clip_by_value(output, 0, 255), tf.uint8)

        # Save image to jpeg form
        img = tf.image.encode_jpeg(highres[0])
        if args.o:
            tf.io.write_file(os.path.join(args.o, 'image.jpeg'), img)
        else:
            tf.io.write_file('image.jpeg', img)

    else:
        print('invalid image path')


def process(img, type='in'):
    """
    input 3 dimension
    """
    if type == 'in':
        img = tf.cast(img, tf.float32)
        img = tf.expand_dims(img, 0)
    elif type == 'out':
        img = tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8)

    return img


def videotofile(args):
    if args.i:
        model = hub.load('https://tfhub.dev/captain-pool/esrgan-tf2/1')

        video = cv2.VideoCapture(args.i)

        count = 0
        while video.grab():
            _, img = video.retrieve()

            timeinit = time.time()

            highres = model(process(img))
            img = process(highres, 'out')

            timefinal = time.time()

            print(f'FPS: {1 / (timefinal - timeinit)}')
            print(f'Write to: {args.o}/{count}.jpg')
            cv2.imwrite(f'{args.o}/{count}.jpg', img[0].numpy())
            count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i', help='image to be enhanced, absolute path', required=True)
    parser.add_argument('-o', help='output directory path', required=True)
    # main(parser.parse_args())
    videotofile(parser.parse_args())
