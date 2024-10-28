#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 10/12/2024 11:14 AM
# @Author  : Loading
import cv2
import dlib
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import os
import random
import matplotlib.pyplot as plt




class FaceAlign:
    def __init__(self, predictor_path='./shape_predictor_68_face_landmarks.dat'):
        # 初始化 Dlib 的人脸检测器和面部关键点检测器
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def __call__(self, img):
        # 将 PIL 图像转换为 OpenCV 格式
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # 进行中心裁切
        img_cv = self.center_crop(img_cv, size=112)

        # 获取人脸关键点
        landmarks = self.get_face_landmarks(img_cv)

        # 如果检测到关键点，则在原图上绘制
        if landmarks:
            # # 在 img_cv 上绘制关键点
            # img_with_landmarks = img_cv.copy()  # 创建图像副本用于绘制关键点
            # for (x, y) in landmarks:
            #     # 在每个关键点上绘制一个红色圆点，半径为2
            #     cv2.circle(img_with_landmarks, (x, y), radius=2, color=(0, 0, 255), thickness=-1)
            #
            # # 展示绘制了关键点的图像
            # cv2.imshow("Landmarks Detected", img_with_landmarks)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # 如果检测到关键点，则进行对齐
            img_cv = self.align_face(img_cv, landmarks)

        # 将对齐后的图像转换回 PIL 格式，作为输出
        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    def center_crop(self, image, size=112):
        h, w = image.shape[:2]
        # 计算裁剪区域的中心
        center_h, center_w = h // 2, w // 2
        # 计算裁剪边界
        crop_h = size // 2
        crop_w = size // 2

        # 确保裁剪区域在图像范围内
        start_h = max(center_h - crop_h, 0)
        start_w = max(center_w - crop_w, 0)
        end_h = min(center_h + crop_h, h)
        end_w = min(center_w + crop_w, w)

        # 裁剪图像
        cropped_image = image[start_h:end_h, start_w:end_w]

        # 如果裁剪结果不是112x112，可以调整大小
        if cropped_image.shape[0] != size or cropped_image.shape[1] != size:
            cropped_image = cv2.resize(cropped_image, (size, size))
        return cropped_image

    def get_face_landmarks(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        landmarks = []

        for face in faces:
            shape = self.predictor(gray, face)
            landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
            # print(f"Detected landmarks: {landmarks}")  # Debug print to verify landmarks

        return landmarks

    def align_face(self, image, landmarks, output_size=(112, 112)):
        # 选择鼻子、左眼和右眼的关键点
        selected_landmarks = np.array([
            landmarks[30],  # 鼻子
            landmarks[36],  # 左眼（外角）
            landmarks[45]  # 右眼（外角）
        ], dtype=np.float32)

        # 目标位置
        h, w = image.shape[:2]
        target_landmarks = np.array([
            [output_size[0] / 2, output_size[1] / 2],  # 鼻子
            [output_size[0] / 3, output_size[1] / 2 - 10],  # 左眼
            [2 * output_size[0] / 3, output_size[1] / 2 - 10]  # 右眼
        ], dtype=np.float32)

        # 计算仿射变换矩阵
        M = cv2.getAffineTransform(selected_landmarks, target_landmarks)
        aligned_face = cv2.warpAffine(image, M, (w, h))

        # 调整大小到统一输出大小
        aligned_face_resized = cv2.resize(aligned_face, output_size)

        return aligned_face_resized


def main(image_folder):
    # 创建 FaceAlign 实例
    face_aligner = FaceAlign()

    # 获取文件夹中的所有图像文件
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 随机选择 5 张图像
    selected_images = random.sample(image_files, min(5, len(image_files)))

    # 存储对齐后的图像
    aligned_images = []

    plt.figure(figsize=(10, 6))  # 调整图像的大小

    for i, image_file in enumerate(selected_images):
        # 读取图像
        img_path = os.path.join(image_folder, image_file)
        img = Image.open(img_path)

        # 进行人脸对齐
        aligned_img = face_aligner(img)

        # 添加到结果列表
        aligned_images.append(aligned_img)

        # 显示原始和对齐后的图像
        # 原始图像
        plt.subplot(2, 5, i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title('Original')

        # 对齐后的图像
        plt.subplot(2, 5, i + 6)
        plt.imshow(aligned_img)
        plt.axis('off')
        plt.title('Aligned')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 替换为你的图像文件夹路径
    image_folder = '../data/11-785-f24-hw2p2-verification/ver_data'
    main(image_folder)