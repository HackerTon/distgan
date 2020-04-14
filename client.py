import argparse
import asyncio
import websockets
import json
import sys

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


with tf.device('/device:cpu:0'):
    model = hub.load('https://tfhub.dev/captain-pool/esrgan-tf2/1')

args = None


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


def encode(img):
    _, img = cv2.imencode('.jpg', img)
    return img


def decode(img):
    return cv2.imdecode(img, 3)


async def client():
    url = f'ws://{args.i}:{args.p}'

    try:
        async with websockets.connect(url, max_size=None) as websocket:
            async for msg in websocket:
                print('|', end='')
                imgarr = np.frombuffer(msg, np.uint8)
                imgarr = decode(imgarr)

                img = process(imgarr)
                with tf.device('/device:cpu:0'):
                    img = model(img)
                img = process(img, 'out')

                # Encode to JPEG
                img = encode(img.numpy()[0]).tobytes()
                await websocket.send(img)
    except websockets.ConnectionClosedError as e:
        print(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='ip address of server', required=True)
    parser.add_argument('-p', help='port of server', required=True)
    args = parser.parse_args()

    print('client start')
    asyncio.get_event_loop().run_until_complete(client())
