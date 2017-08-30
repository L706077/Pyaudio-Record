#!/usr/bin/env python
# -*- coding: utf-8 -*-

import threading
from array import array
from queue import Queue, Full
import numpy as np
import pyaudio
from python_speech_features import mfcc
import tensorflow as tf
import datetime
import wave
from struct import pack
from collections import namedtuple  # structure
import logging
import time
import csv
import os

CHUNK_SIZE = 32000  # if 512 means
CHUNK_OVERLAP = 16000
SAMPLE_RATE = 16000
FORMAT = pyaudio.paInt16
BUF_MAX_SIZE = CHUNK_SIZE * 10  # if the recording thread can't consume fast enough, the listener will start discarding
THRESHOLD = 0  # threshold for keyword founded
tf.app.flags.DEFINE_string('export_dir', './export/', 'directory in which exported models are stored - if omitted, the model won\'t get exported')
FLAGS = tf.app.flags.FLAGS


def log_text():
    logging.basicConfig(level=logging.NOTSET,
                        format='%(asctime)s.%(msecs)03d %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S',
                        handlers=[logging.FileHandler('test_model.log', 'a', 'utf-8'),])
    console = logging.StreamHandler()
    console.setLevel(logging.CRITICAL)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)  # handler set output format
    logging.getLogger('').addHandler(console)  # add handler to root logger

    key_path = "./keyword_wav/Keyword"
    nonekey_path = './keyword_wav/None_Keyword'
    if not os.path.isdir(key_path):  # if path is not exist, make the direction
        os.mkdir(key_path)
    if not os.path.isdir(nonekey_path):  # if path is not exist, make the direction
        os.mkdir(nonekey_path)


def stopwatch(start_duration=0):
    """This function will toggle a stopwatch.
    The first call starts it, second call stops it, third call continues it etc.
    So if you want to measure the accumulated time spent in a certain area of the code,
    you can surround that code by stopwatch-calls like this:

    .. code:: python

        fun_time = 0 # initializes a stopwatch
        [...]
        for i in range(10):
          [...]
          # Starts/continues the stopwatch - fun_time is now a point in time (again)
          fun_time = stopwatch(fun_time)
          fun()
          # Pauses the stopwatch - fun_time is now a duration
          fun_time = stopwatch(fun_time)
        [...]
        # The following line only makes sense after an even call of :code:`fun_time = stopwatch(fun_time)`.
        print 'Time spent in fun():', format_duration(fun_time)
    """
    if start_duration == 0:
        return datetime.datetime.utcnow()
    else:
        return datetime.datetime.utcnow() - start_duration


def model_init(sess):
    saver = tf.train.import_meta_graph(FLAGS.export_dir + 'model.ckpt.meta')  # import .ckpt.meta direction
    saver.restore(sess, FLAGS.export_dir + 'model.ckpt')
    graph = tf.get_default_graph()
    input_tensor = graph.get_tensor_by_name('input_source_1:0')
    logits = graph.get_tensor_by_name('add_7:0')
    keep_prob = graph.get_tensor_by_name('keep_prob_1:0')
    return input_tensor, logits, keep_prob


def wav_save(r, path,type):
    lastmod_date = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H:%M:%S:%f.wav')
    r = pack('<' + ('h' * len(r)), *r)
    wf = wave.open(path + lastmod_date, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(r)
    wf.close()
    if type:
        csv_data = [(lastmod_date, 'b')]  # Keyword
        f = open(path + "keyword.csv", "a")  # 'a' means append

    else:
        csv_data = [(lastmod_date, 'a')]  # Nonkeyword
        f = open(path + "nokeyword.csv", "a")

    w = csv.writer(f)
    w.writerows(csv_data)
    del csv_data[:]
    f.close()


def final_record(ans, keyword_ans, none_keyword_ans, record, inference_time):
    logger_test = logging.getLogger('testing_model')
    ans_0 = ans[0][0]  # no keyword probability
    ans_1 = ans[0][1]  # keyword probability

    # and abs(ans_0) < abs(ans_1)
    if ans_0 < ans_1 and ans_1 > THRESHOLD:
        logger_test.critical("keyword founded")
        keyword_ans += 1
        wav_save(r=record, path='./keyword_wav/Keyword/', type=True)

    else:
        logger_test.info("None Keyword")
        none_keyword_ans += 1
        wav_save(r=record, path='./keyword_wav/None_Keyword/', type=False)

    logger_test.info("Keyword Probability: {}".format(ans_1))
    logger_test.info("None Keyword Probability: {}".format(ans_0))
    logger_test.info("Inference Time: {}".format(inference_time))
    logging.critical("Summary keyword answer: {}".format(keyword_ans))
    logging.critical("Summary None keyword answer: {}".format(none_keyword_ans))
    return keyword_ans, none_keyword_ans


def input_model(stopped, q, input_tensor, logits, keep_prob, sess):
    keyword_ans = 0
    none_keyword_ans = 0
    while 1:
        if stopped.wait(timeout=0):
            break
        source, record_save = q.get()
        # general initialization
        batch_x = [source,]
        inference_time = stopwatch()
        ans = sess.run(logits, feed_dict={input_tensor: batch_x, keep_prob: 1.0})
        inference_time = stopwatch(inference_time)
        keyword_ans, none_keyword_ans = final_record(ans=ans,
        keyword_ans=keyword_ans,
        none_keyword_ans=none_keyword_ans,
        record=record_save,
        inference_time=inference_time)


def record(stopped, q, stream):
    logger_record = logging.getLogger('recording')
    first_time = True
    stream_save = []
    stream_test = []
    r_temp = array('h')  # save wav.
    while 1:
        if stopped.wait(timeout=0):
            break
        try:
            record_time = stopwatch()
            print("Please speak to microphone...")
            stream_record = stream.read(CHUNK_SIZE)
            r_temp.extend(array('h', stream_record))  # for save wav.
            stream_test.extend(stream_record)
            for i in range(len(r_temp)):
                stream_save.append(r_temp[i])
            audio = np.array(stream_save)  # transfer array size
            orig_inputs = mfcc(audio, samplerate=SAMPLE_RATE, numcep=26)  # MFCC feature extraction
            orig_inputs = (orig_inputs - np.mean(orig_inputs)) / np.std(orig_inputs)
            source = orig_inputs
            source = source.ravel()
            output = mystruct(source=source, audio_wav=r_temp)
            q.put(output)
            time.sleep(0.1)
            del stream_save[:]
            del r_temp[:]
            record_time = stopwatch(record_time)
            logger_record.info("Recording time: {}".format(record_time))
        except Full:
            logger_record.error("Thread full")
            pass


def main():
    global mystruct
    mystruct = namedtuple("struct_pass", "source audio_wav")
    print('Welcome to NLP_KWS_1.0.0 !')
    print('TensorFlow detected: V {}'.format(tf.__version__))
    log_text()
    logging.critical("***** START AT {} *****".format(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H:%M:%S')))
    sess = tf.Session()
    input_tensor, logits, keep_prob = model_init(sess)
    stream = pyaudio.PyAudio().open(format=FORMAT, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK_SIZE)

    stopped = threading.Event()
    q = Queue(maxsize=int(round(BUF_MAX_SIZE / CHUNK_SIZE)))

    record_t = threading.Thread(target=record, args=(stopped, q, stream))
    record_t.start()
    input_model_t = threading.Thread(target=input_model, args=(stopped, q, input_tensor, logits, keep_prob, sess))
    input_model_t.start()

    try:
        while True:
            record_t.join(0.1)
            input_model_t.join(0.1)
    except KeyboardInterrupt:
        logging.warning("***** KeyboardInterrupt *****")
        stopped.set()

    record_t.join()
    input_model_t.join()


if __name__ == '__main__':
    main()
