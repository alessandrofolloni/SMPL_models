import os
import subprocess
import warnings
from datetime import datetime

import numpy as np
from path import Path

import mmu


class VideoWriter(object):

    def __init__(self, vid_path, fps, quality=0.5):
        # type: (str, int, float) -> None
        """
        :param vid_path: path of the video to write
        :param fps: frames per second
        :param quality: quality of the video
            ->> float in range [0, 1], where 0 is the lowest quality
                (i.e. maximum compression) and 1 is the highest quality
                (i.e. lossless compression)
        """

        # linux and macos only
        assert os.name == 'posix', \
            'this class is only compatible with UNIX-style operating systems'

        self.vid_path = Path(vid_path).abspath()
        assert self.vid_path.ext == '.mp4', \
            'only ".mp4" videos are supported'
        if self.vid_path.exists():
            w_msg = f'"{self.vid_path}" already exists, '
            w_msg += 'and will be overwritten'
            warnings.warn(w_msg)

        assert 0 <= quality <= 1, \
            ('quality must be in range [0, 1], '
             'where 0 is the lowest quality '
             'and 1 is the highest quality')

        self.crf = 1 + int(round(49 * (1 - quality)))
        self.fps = fps

        self.frame_number = 0
        self.img_shape = None
        self.done = False

        now = datetime.now().isoformat()
        self.tmp_dir = Path('/') / 'tmp' / 'video_writer' / now
        self.tmp_dir.makedirs_p()


    def write(self, img, is_rgb=True):
        # type: (np.ndarray, bool) -> None
        """
        :param img: image to write
        :param is_rgb: if `True`, the image is assumed to be in RGB format;
            if `False`, the image is assumed to be in BGR format
            ->> default: `True` (i.e. RGB format)
        """
        if self.frame_number >= 999999999:
            raise RuntimeError('[ERROR]: maximum number of frames reached')

        if self.img_shape is None:
            self.img_shape = img.shape
        elif self.img_shape != img.shape:
            raise ValueError('[ERROR]: all images must have the same shape')

        f = self.tmp_dir / f'{self.frame_number:09d}.png'
        if is_rgb:
            mmu.imwrite(f, img)
        else:
            mmu.imwrite(f, img[..., ::-1])

        self.frame_number += 1


    def release(self, verbose=False):
        # type: (bool) -> None
        """
        Release the video writer and save the video. This method must
        be called at the end of the video writing process.
        :param verbose: if `True`, prints the output of the ffmpeg
            command used to build the video; if `False`, does not
            print anything
        """

        verbose = '' if verbose else '-loglevel quiet'
        if not self.done:
            subprocess.run(
                f'ffmpeg -y -r {self.fps} -f image2 '
                f'-s {self.img_shape[1]}x{self.img_shape[0]} '
                f'-i "{self.tmp_dir}/%09d.png" '
                f'-vcodec libx264 -crf {self.crf} '
                f'-pix_fmt yuv420p "{self.vid_path}" '
                f'{verbose}', shell=True
            )
            subprocess.run(
                f'rm -r "{self.tmp_dir}"', shell=True
            )
            self.done = True


    def __del__(self):
        # type: () -> None
        self.release()
