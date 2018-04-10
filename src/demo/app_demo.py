
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np

import src.facenet as facenet
import src.align.detect_face
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import src.demo.db_util as db_util
import threading
import cv2
import datetime
import json
import time


class BoundingBox:
    def __init__(self):
        self.top_left_w = 0
        self.top_left_h = 0
        self.bottom_right_w = 0
        self.bottom_right_h = 0
        self.confidence = 0


class FaceDetector:
    """
    人脸检测器

    使用MTCNN技术(
    《2016 Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks》.
    http://arxiv.org/abs/1604.02878v1
    )
    从图中找到人脸
    使用方法：
    face_detector = FaceDetector()
    ...
    bounding_boxes = face_detector.detect_faces(images)
    """
    def __init__(self):
        self._minsize = 20  # minimum size of face
        self._threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        self._factor = 0.709  # scale factor
        # Face image resize to (height, width) in pixels.
        self._face_size = 160
        # 输出人脸框 比 神经网络检测到的框 要大margin
        self._margin = 44
        print('Creating networks and loading parameters')
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                self._pnet, self._rnet, self._onet = src.align.detect_face.create_mtcnn(sess, None)

    def detect_faces(self, images):
        """
        使用MTCNN 检测 图像中的人脸，得到人脸的框
        Args:
            images: list, 其中每个元素是一个3维的 numpy [H, W, C]
        Returns:
            list[list[BoundingBox]] result.
            result[i] 对应于images[i]，result[i][j]是第j个探测到的BoundingBox
        """
        result_list = []
        for image in images:
            img_size = np.asarray(image.shape)[0:2]
            bounding_boxes, _ = src.align.detect_face.detect_face(image, self._minsize, self._pnet, self._rnet, self._onet, self._threshold, self._factor)
            if len(bounding_boxes) < 1:
                continue
            bounding_box_list = []
            for idx, bounding_box in enumerate(bounding_boxes):
                bb = BoundingBox()
                bb.top_left_w = np.maximum(int(bounding_box[0] - self._margin / 2), 0)
                bb.top_left_h = np.maximum(int(bounding_box[1] - self._margin / 2), 0)
                bb.bottom_right_w = np.minimum(int(bounding_box[2] + self._margin / 2), img_size[1])
                bb.bottom_right_h = np.minimum(int(bounding_box[3] + self._margin / 2), img_size[0])
                bb.confidence = bounding_box[4]
                bounding_box_list.append(bb)
            result_list.append(bounding_box_list)
        return result_list


class FaceFeatureExtractor:
    """
    人脸特征提取器, 使用facenet 技术，从人脸图像中提取特征。
    每张脸形成128维特征
    """
    def __init__(self, model):
        """
            Args:
                model: 模型的路径
        """
        with tf.Graph().as_default():
            self._sess = tf.Session()
            facenet.load_model(model, self._sess)
            # Get input and output tensors
            self._images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            self._embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            self._phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    def extract_features_from_images(self, images):
        """
            Args:
                images: [N, H, W, C] 的numpy，表示一组图像
            Returns:
                embeddings: [N, 128] 的 numpy，表示每个输入图像对应的特征
        """
        feed_dict = {self._images_placeholder: images, self._phase_train_placeholder: False}
        embeddings = self._sess.run(self._embeddings, feed_dict=feed_dict)
        return embeddings


class FaceInfo:
    """
    一个结构体，用来保存脸的信息
    """
    def __init__(self, name='unknown', face_image=None, face_features=None, raw_image_id=None):
        self.name = name  # str, 脸对应的人的名字
        self.face_image = face_image  # PIL.Image, 脸的图像, 'RGB'模式
        self.face_features = face_features  # 一位数组, 脸的特征
        self.raw_image_id = raw_image_id  # 该脸图像的来自的图像的ID，只有在录入新脸时才有效
        self.similarity = 0

    def insert_into_db(self):
        """
        将数据写入数据库
        Returns:
             写入的记录ID
        """
        conn = db_util.get_connection()
        cursor = conn.cursor()
        cursor.execute('INSERT INTO face (name, raw_image_id, features, face_image, image_height, image_width ) values '
                       '(%s, %s, %s, %s, %s, %s)',
                       (self.name,
                        self.raw_image_id,
                        json.dumps(self.face_features),
                        self.face_image.tobytes(),
                        self.face_image.size[0],
                        self.face_image.size[1]
                        ))
        conn.commit()
        cursor.execute('SELECT LAST_INSERT_ID()')
        id = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return id


def insert_raw_image_into_db(image, name):
    """
    将一个捕获到的图像(完整大小)，以及相应的信息 写入数据库
    Args:
        image: PIL.Image, 待写入的图像, 'RGB'模式
        name: 对应的人的名字
    Returns:
        写入的记录的ID
    """
    conn = db_util.get_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO raw_image (image, image_height, image_width, name) '
                   'values(%s, %s, %s, %s)',
                   (image.tobytes(),
                    image.size[0],
                    image.size[1],
                    name))
    conn.commit()
    cursor.execute('SELECT LAST_INSERT_ID()')
    result = cursor.fetchone()
    id = result[0]
    cursor.close()
    conn.close()
    return id


class CapturedImage:
    """
    一个结构体，用来存储捕捉到的最近的图像。含有lock，用于多线程访问。
    """
    def __init__(self):
        self.lock = threading.Condition()  # 锁
        self.image = None   # 存储图像的成员，应该是numpy [H, W, C]
        self.frame_counter = 0  # 对帧数进行计数
        self.captured_time = None  # 该帧的捕获时间

    def set_image(self, input_image):
        """
        设置内部存储的图像数据，该操作是线程安全的。
        Args:
         input_image: numpy[N, W, C]
        """
        self.lock.acquire()
        self.image = np.copy(input_image)
        self.frame_counter += 1
        self.captured_time = datetime.datetime.now()
        self.lock.release()

    def get_image(self):
        return self.image

    def lock_acquire(self):
        self.lock.acquire()

    def lock_release(self):
        self.lock.release()


def search_most_similar_face_from_db(features):
    """
    遍历输入库中的脸，找到与目标特征最接近的一个
    Args:
        features: 待匹配的脸的特征
    Returns:
        best_match_face: 最佳匹配的脸
    """
    conn = db_util.get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT face_id, features FROM face ORDER BY face_id')
    results = cursor.fetchall()
    largest_sim = 0
    closest_face_id = None
    if len(results) == 0:
        return
    for result in results:
        recorded_id = result[0]
        recorded_features = json.loads(result[1])
        cos_theta = np.dot(features, recorded_features) / (np.linalg.norm(features) * np.linalg.norm(recorded_features))
        sim = (1 + cos_theta) / 2
        if sim > largest_sim:
            largest_sim = sim
            closest_face_id = recorded_id
    cursor.execute('SELECT name, face_image, image_height, image_width, record_create_time, features '
                   'FROM face WHERE face_id=%s', (closest_face_id, ))
    result = cursor.fetchone()
    best_match_image = Image.frombytes(mode='RGB', size=(result[2], result[3]), data=result[1])
    best_match_face = FaceInfo(name=result[0], face_image=best_match_image, face_features=json.loads(result[5]))
    best_match_face.similarity = sim
    return best_match_face


class ImageCapturingThread(threading.Thread):
    """
    图像捕获线程。
    从摄像头捕获图像，放置到self.captured_image中，找到人脸，并绘制出来，放入image_widget中
    """
    def __init__(self, captured_image, image_widget, face_resize):
        """
        Args:
            captured_image: 捕获到的图像所存入的位置，用于内部数据存储。
            image_widget: 图像以及人脸框所出入的位置，用于界面显示。
        """
        threading.Thread.__init__(self)
        self.captured_image = captured_image
        self.image_widget = image_widget
        self.face_detector = FaceDetector()
        self.face_feature_extractor = FaceFeatureExtractor('../pre-trained_models/20170512-110547')
        self.face_resize = face_resize
        self.frame_counter = 0

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            # get a frame
            self.frame_counter += 1

            _, np_image = cap.read()  # GBR格式
            np_image = np.roll(np_image, 1, axis=-1) # 重新排列通道，转变为 RGB格式

            t_start = time.time()
            self.captured_image.set_image(input_image=np_image)
            image = Image.fromarray(np_image, mode='RGB')
            draw = ImageDraw.Draw(image)
            tmp = self.face_detector.detect_faces([np_image])
            if np.shape(tmp)[0] != 0:
                face_bb_list = tmp[0]

                for face_bb in face_bb_list:
                    draw.rectangle(
                        (face_bb.top_left_w, face_bb.top_left_h, face_bb.bottom_right_w, face_bb.bottom_right_h),
                        outline='red')

                    cropped = np_image[face_bb.top_left_h:face_bb.bottom_right_h,
                              face_bb.top_left_w:face_bb.bottom_right_w, :]
                    aligned = misc.imresize(cropped, (self.face_resize, self.face_resize), interp='bicubic')
                    prewhitened = facenet.prewhiten(aligned)
                    face_image = np.reshape(prewhitened, (1,) + np.shape(prewhitened))
                    features = self.face_feature_extractor.extract_features_from_images(face_image)
                    most_similar_face = search_most_similar_face_from_db(features)
                    if most_similar_face is None:
                        continue
                    if most_similar_face.similarity > 0.9:
                        draw.text(
                            (face_bb.top_left_w, face_bb.top_left_h),
                            text=most_similar_face.name + ' similarity : ' + str(most_similar_face.similarity)
                        )
                    else:
                        draw.text(
                            (face_bb.top_left_w, face_bb.top_left_h),
                            text='No Match, smallest distance : ' + str(most_similar_face.similarity)
                        )

            tk_image = ImageTk.PhotoImage(image=image)
            self.image_widget['image'] = tk_image
            t_finish = time.time()
            print('frame %d: time cost %.3fs' % (self.frame_counter, t_finish - t_start))
        cap.release()
        cv2.desotryAllWindows()


class FaceRecognitionApp:
    def __init__(self, master):
        """
        初始化窗口
        Args:
            master: Tkinter.Tk()的root对象
        """
        window = tk.Frame(master)
        window.pack()
        self.image_label = tk.Label(window, width=800, height=800)
        self.image_label.grid(row=0, column=0, sticky=tk.N + tk.W)

        self.record_face_panel = tk.Frame(window)
        self.record_face_panel.grid(row=0, column=1, sticky=tk.N)

        self.face_name_label = tk.Label(self.record_face_panel, text='名字:')
        self.face_name_label.grid(row=0, column=0)
        self.face_name_entry = tk.Entry(self.record_face_panel)
        self.face_name_entry.grid(row=0, column=1)

        self.record_face_button = tk.Button(self.record_face_panel, text='录入', command=self.record_face)
        self.record_face_button.grid(row=1)

        self.image_id = 1
        self.tk_image = None
        self.face_detector = FaceDetector()
        self.face_feature_extractor = FaceFeatureExtractor('../pre-trained_models/20170512-110547')
        self.face_resize = 160
        self.captured_image = CapturedImage()
        self.capturing_image_thread = ImageCapturingThread(self.captured_image, self.image_label, self.face_resize)
        print('starting capturing thread')
        self.capturing_image_thread.start()

    def record_face(self):
        """
        从最近捕获到的一帧中(self.captured_image) 找脸、提取特征、连同self.face_name_entry录入的信息，写入数据库
        """
        self.captured_image.lock_acquire()
        np_image = np.copy(self.captured_image.get_image())
        self.captured_image.lock_release()
        tmp = self.face_detector.detect_faces([np_image])
        if np.shape(tmp)[0] != 0:
            raw_image_id = insert_raw_image_into_db(image=Image.fromarray(np_image), name=self.face_name_entry.get())
            face_bb = tmp[0][0]
            cropped = np_image[face_bb.top_left_h:face_bb.bottom_right_h, face_bb.top_left_w:face_bb.bottom_right_w, :]
            aligned = misc.imresize(cropped, (self.face_resize, self.face_resize), interp='bicubic')
            prewhitened = facenet.prewhiten(aligned)
            face_image = np.reshape(prewhitened, (1,) + np.shape(prewhitened))
            features = self.face_feature_extractor.extract_features_from_images(face_image)
            face_info = FaceInfo(name=self.face_name_entry.get(), face_image=Image.fromarray(cropped),
                                 face_features=features[0, :].tolist(), raw_image_id=raw_image_id)
            face_info.insert_into_db()


def main():
    root = tk.Tk(className='Face Recognition DEMO')
    app = FaceRecognitionApp(root)
    root.mainloop()
    # root.destroy()


if __name__=='__main__':
    main();
