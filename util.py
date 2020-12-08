import numpy as np
import os
import pickle
import nibabel as nib

from crop_black import *
from filter_func import *
from get_lr import *
from hashtable import *
from matrix_compute import *
from util import *

import filter_constant as C


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if fname.endswith('.nii.gz'):
                path = os.path.join(root, fname)
                images.append(path)
    return images


def get_train_data(file):
    raw_image = np.array(nib.load(file).get_fdata(), dtype=np.float32)
    clipped_image = clip_image(raw_image)
    im = mod_crop(clipped_image, C.R)
    slice_area = crop_slice(im, C.PATCH_HALF, C.R)

    im_blank_LR = get_lr(im) / im.max()
    im_LR = im_blank_LR[slice_area]
    im_HR = im[slice_area] / im.max()

    return im_HR, im_LR


def ask_save_qv(Q, V, finished_files):
    try:
        a = input_timer("\r Enter to save >> ", 10)
        save_qv(Q, V, finished_files)
    except TimeoutError as e:
        pass


def save_qv(Q, V, finished_files, count):
    print('\rSaving QVF...', end='', flush=True)
    np.savez('./arrays/QVF_{}'.format(C.R), Q=Q, V=V, finished_files=np.array(finished_files), count=count)


def init_buckets():
    patchS = [[] for j in range(C.Q_TOTAL)]
    xS = [[] for j in range(C.Q_TOTAL)]
    return patchS, xS


def load_files():
    if os.path.isfile('./arrays/QVF_{}.npz'.format(C.R)):
        print('Loading QVF...', end=' ', flush=True)
        QVF = np.load('./arrays/QVF_{}.npz'.format(C.R))
        Q = QVF['Q']
        V = QVF['V']
        finished_files = QVF['finished_files'].tolist()
        count = QVF['count']
        QVF.close()
        print('Done', flush=True)
    else:
        Q = np.zeros((C.Q_TOTAL, C.FILTER_VOL+1, C.FILTER_VOL+1), dtype=np.float64)
        V = np.zeros((C.Q_TOTAL, C.FILTER_VOL+1), dtype=np.float64)
        finished_files = []
        count = np.zeros(C.Q_TOTAL, dtype=int)

    return Q, V, finished_files, count


# Original Code Source : https://greenfishblog.tistory.com/257
def input_timer(prompt, timeout_sec):
    
    import subprocess
    import sys
    import threading
    import locale

    class Local:
        # check if timeout occured
        _timeout_occured = False

        def on_timeout(self, process):
            self._timeout_occured = True
            process.kill()
            # clear stdin buffer (for linux)
            # when some keys hit and timeout occured before enter key press,
            # that input text passed to next input().
            # remove stdin buffer.
            try:
                import termios
                termios.tcflush(sys.stdin, termios.TCIFLUSH)
            except ImportError:
                # windows, just exit
                pass

        def input_timer_main(self, prompt_in, timeout_sec_in):
            # print with no new line
            print(prompt_in, end="")

            # print prompt_in immediately
            sys.stdout.flush()

            # new python input process create.
            # and print it for pass stdout
            cmd = [sys.executable, '-c', 'print(input())']
            with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
                timer_proc = threading.Timer(timeout_sec_in, self.on_timeout, [proc])
                try:
                    # timer set
                    timer_proc.start()
                    stdout, stderr = proc.communicate()

                    # get stdout and trim new line character
                    result = stdout.decode(locale.getpreferredencoding()).strip("\r\n")
                finally:
                    # timeout clear
                    timer_proc.cancel()

            # timeout check
            if self._timeout_occured is True:
                # move the cursor to next line
                #print("")
                raise TimeoutError
            return result

    t = Local()
    return t.input_timer_main(prompt, timeout_sec)