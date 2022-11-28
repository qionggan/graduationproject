import cv2
import numpy as np


def detect(image):
    # 定义分类器
    cascade_path = 'cascade.xml'
    cascade = cv2.CascadeClassifier(cascade_path)
    # 修改图片大小
    resize_h = 400
    height = image.shape[0]
    scale = image.shape[1] / float(height)
    image = cv2.resize(image, (int(scale * resize_h), resize_h))
    # 转为灰度图
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 车牌定位
    car_plates = cascade.detectMultiScale(image_gray, 1.1, 2, minSize=(36, 9), maxSize=(36 * 40, 9 * 40))
    print("检测到车牌数", len(car_plates))
    if len(car_plates) > 0:
        for car_plate in car_plates:
            x, y, w, h = car_plate
            plate = image[y - 10: y + h + 10, x - 10: x + w + 10]
            #cv2.imshow('plate', plate
            cv2.rectangle(image, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 0), 2)
    cv2.imshow("image", image)


if __name__ == '__main__':
    image = cv2.imread('3.jpg')
    print ('image',image.shape)
    detect(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
