"""
Code from opencv dnn module: 
    https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector
"""

from __future__ import print_function
import os
import hashlib
import time
import sys
import xml.etree.ElementTree as ET
if sys.version_info[0] < 3:
    from urllib2 import urlopen
else:
    from urllib.request import urlopen


class _HashMismatchException(Exception):
    def __init__(self, expected, actual):
        Exception.__init__(self)
        self.expected = expected
        self.actual = actual
    def __str__(self):
        return 'Hash mismatch: {} vs {}'.format(self.expected, self.actual)


class _MetalinkDownloader(object):
    BUFSIZE = 10*1024*1024
    NS = {'ml': 'urn:ietf:params:xml:ns:metalink'}
    tick = 0

    def download(self, metalink_file):
        status = True
        for file_elem in ET.parse(metalink_file).getroot().findall('ml:file', self.NS):
            url = file_elem.find('ml:url', self.NS).text
            fname = file_elem.attrib['name']
            hash_sum = file_elem.find('ml:hash', self.NS).text
            print('*** {}'.format(fname))
            try:
                self.verify(hash_sum, fname)
            except Exception as ex:
                print('  {}'.format(ex))
                try:
                    print('  {}'.format(url))
                    with open(fname, 'wb') as file_stream:
                        self.buffered_read(urlopen(url), file_stream.write)
                    self.verify(hash_sum, fname)
                except Exception as ex:
                    print('  {}'.format(ex))
                    print('  FAILURE')
                    status = False
                    continue
            print('  SUCCESS')
        return status

    def print_progress(self, msg, timeout = 0):
        if time.time() - self.tick > timeout:
            print(msg, end='')
            sys.stdout.flush()
            self.tick = time.time()

    def buffered_read(self, in_stream, processing):
        self.print_progress('  >')
        while True:
            buf = in_stream.read(self.BUFSIZE)
            if not buf:
                break
            processing(buf)
            self.print_progress('>', 5)
        print(' done')

    def verify(self, hash_sum, fname):
        sha = hashlib.sha1()
        with open(fname, 'rb') as file_stream:
            self.buffered_read(file_stream, sha.update)
        if hash_sum != sha.hexdigest():
            raise _HashMismatchException(hash_sum, sha.hexdigest())


def maybe_download_model():
    """
    Download OpenCV Face Dectector model into designated model_path.

    Arguments:
        + model_path (str) -- model_path directory to OpenCV Caffe model.
    
    Files:
        + deploy.prototxt
        + res10_300x300_ssd_iter_140000_fp16.caffemodel
    """

    # Files path
    model_path = os.path.join(os.environ['HOME'], ".keras/opencv/face_detector/")
    model_caffe_path = os.path.join(model_path, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
    proto_caffe_path = os.path.join(model_path, "deploy.prototxt")
    
    # Download model if it does not exists
    if not os.path.isfile(model_caffe_path) or not os.path.isfile(proto_caffe_path):
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # OpenCV files url
        url_files = {
            "weights.meta4":"https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/weights.meta4",
            "deploy.prototxt":"https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        }

        # Download OpenCV script do download Face Detector weights
        for file in url_files:
            with open(os.path.join(model_path, file), "wb") as f:
                f.write(urlopen(url_files[file]).read())
            
        # Execute OpenCV script
        current_model_path = os.getcwd()
        os.chdir(model_path)
        _MetalinkDownloader().download('weights.meta4')
        os.chdir(current_model_path)