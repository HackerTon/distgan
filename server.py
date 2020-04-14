import argparse
import asyncio
import os
import sys
import selectors as sl
import websockets
import json
import logging
import time

import aiofiles as io
import cv2
import numpy as np

file: cv2.VideoCapture = None
args = None
count = 0


async def read():
    global count

    file.grab()
    _, result = file.retrieve()

    cur_count = count
    count += 1

    return result, cur_count


def encode(img):
    _, img = cv2.imencode('.jpg', img)
    return img


def decode(img):
    return cv2.imdecode(img, 3)


async def server(websocket, path):
    global count

    logging.info(websocket)

    while True:
        starttime = time.time()

        try:
            content, cur_count = await read()
            content = encode(content).tobytes()

            await websocket.send(content)

            msg = await websocket.recv()

            # Decode from JPEG
            imgarr = np.frombuffer(msg, np.uint8)
            imgarr = decode(imgarr)

            cv2.imwrite(os.path.join(args.o, f'{count}.jpg'), imgarr)
            logging.info(f'{count}.jpg processed and saved')

        except websockets.ConnectionClosed as e:
            logging.warning(e.code, e.reason)

        endtime = time.time()
        logging.info(f'Time per Frame: {round(endtime - starttime)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='ip address of server')
    parser.add_argument('-p', help='port of server', required=True)
    parser.add_argument('-f', help='file to upsample', required=True)
    parser.add_argument('-o', help='output of dir', required=True)

    args = parser.parse_args()
    startserver = websockets.serve(server, args.i, args.p)
    file = cv2.VideoCapture(args.f)

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO,
                        handlers=[logging.StreamHandler(), logging.FileHandler('server.log')])
    logging.info(f'Server start: ip: {args.i} port: {args.p}')

    # print(f'Server start: ip:{args.i} port:{args.p}')
    asyncio.get_event_loop().run_until_complete(startserver)
    asyncio.get_event_loop().run_forever()
