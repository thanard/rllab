import os.path as osp
import joblib
from rllab import config
import numpy as np
import random
from gym.monitoring.video_recorder import ImageEncoder
import tensorflow as tf
import cv2
from rllab.misc import console
import argparse

frame_size = (500, 500)

def to_img(obs, frame_size=(100, 100)):
    return cv2.resize(np.cast['uint8'](obs), frame_size)
    # return cv2.resize(np.cast['uint8']((obs / 2 + 0.5) * 255.0), frame_size)
    # return obs

with tf.Session() as sess:
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file', default='swimmer_normalized_com_cost')
    parser.add_argument('--horizon', type=int, default=100,
                        help='Max length of rollout')
    args = parser.parse_args()

    np.random.seed(0)
    random.seed(0)
    filepath = args.file
    pkl_file = osp.join(config.PROJECT_PATH,
                        filepath,"params.pkl"
                        )
    output_path = osp.join(config.PROJECT_PATH, filepath, "video.mp4")

    # import pdb; pdb.set_trace()
    data = joblib.load(pkl_file)

    policy = data["policy"]

    env = data["env"]
    # env = SwimmerEnv()


    encoder = ImageEncoder(output_path=output_path,
                           frame_shape=frame_size + (3,),
                           frames_per_sec=60)

    print("Generating %s"%output_path)
    obs = env.reset()
    image = env.render(mode='rgb_array')
    policy.reset()
    for t in range(args.horizon):
        compressed_image = to_img(image, frame_size=frame_size)
        # cv2.imshow('frame{}'.format(t), compressed_image)
        # cv2.waitKey(10)
        encoder.capture_frame(compressed_image)
        action, _ = policy.get_action(obs)
        next_obs, reward, done, info = env.step(action)
        obs = next_obs
        image = env.render(mode='rgb_array')
        if done:
            break
    encoder.close()
