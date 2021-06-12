#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from cnocr import CnOcr

# 后续生成票据图像时的大小，按照标准增值税发票版式240mmX140mm来设定
height_resize = 1400
width_resize = 2400

# 实例化不同用途CnOcr对象
ocr = CnOcr(name='')  # 混合字符
ocr_numbers = CnOcr(name='numbers', cand_alphabet='0123456789.')  # 纯数字
ocr_UpperSerial = CnOcr(name='UpperSerial',
                        cand_alphabet='0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ')  # 编号，只包括大写字母(没有O)与数字

# 销售方字典
purchaser_dict = ['purchaserName', 'purchaserCode', 'purchaserAddrTel', 'purchaserBankCode']
seller_dict = ['sellerName', 'sellerCode', 'sellerAddrTel', 'sellerBankCode']
invoice_dict = ['invoiceCode', 'invoiceNumber', 'invoiceDate', 'checkCode']

# 截取图片中部分区域图像-字段
crop_range_list_name = ['invoice', 'purchaser', 'seller',
                        'totalExpense', 'totalTax', 'totalTaxExpenseZh', 'totalTaxExpense',
                        'remark', 'title', 'machineCode']

# 截取图片中部分区域图像-坐标
crop_range_list_data = [[1750, 20, 500, 250], [420, 280, 935, 220], [420, 1030, 935, 230],
                        [1500, 880, 390, 75], [2000, 880, 330, 75], [750, 960, 600, 65], [1870, 960, 300, 70],
                        [1455, 1045, 400, 180], [760, 50, 900, 110], [280, 200, 250, 75]]

# 截取图片中部分区域图像-使用ocr的类型，0：混合字符，1：纯数字，2：编号
crop_range_list_type = [3, 3, 3,
                        1, 1, 0, 1,
                        0, 0, 1]


# 调整原始图片尺寸
def resizeImg(image, height=height_resize):
    h, w = image.shape[:2]
    pro = height / h
    size = (int(w * pro), int(height))
    img = cv2.resize(image, size)
    return img


# 边缘检测
def getCanny(image):
    # 高斯模糊
    binary = cv2.GaussianBlur(image, (3, 3), 2, 2)
    # 边缘检测
    binary = cv2.Canny(binary, 60, 240, apertureSize=3)
    # 膨胀操作，尽量使边缘闭合
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)

    # 二值图
    cv2.imwrite('result/binary.jpg', binary)
    return binary


# 求出面积最大的轮廓
def findMaxContour(image):
    # 寻找边缘
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 计算面积
    max_area = 0.0
    max_contour = []
    for contour in contours:
        current_area = cv2.contourArea(contour)
        if current_area > max_area:
            max_area = current_area
            max_contour = contour
    return max_contour, max_area


# 多边形拟合凸包的四个顶点
def getBoxPoint(contour):
    # 多边形拟合凸包
    hull = cv2.convexHull(contour)
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    approx = approx.reshape((len(approx), 2))
    return approx


# 适配原四边形点集
def adapPoint(box, pro):
    box_pro = box
    if pro != 1.0:
        box_pro = box / pro
    box_pro = np.trunc(box_pro)
    return box_pro


# 四边形顶点排序，[top-left, top-right, bottom-right, bottom-left]
def orderPoints(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# 计算长宽
def pointDistance(a, b):
    return int(np.sqrt(np.sum(np.square(a - b))))


# 透视变换
def warpImage(image, box):
    w, h = pointDistance(box[0], box[1]), \
           pointDistance(box[1], box[2])
    dst_rect = np.array([[0, 0],
                         [w - 1, 0],
                         [w - 1, h - 1],
                         [0, h - 1]], dtype='float32')
    M = cv2.getPerspectiveTransform(box, dst_rect)
    warped = cv2.warpPerspective(image, M, (w, h))
    return warped


# 根据四点画四边形
def drawRect(img, pt1, pt2, pt3, pt4, color, line_width):
    cv2.line(img, pt1, pt2, color, line_width)
    cv2.line(img, pt2, pt3, color, line_width)
    cv2.line(img, pt3, pt4, color, line_width)
    cv2.line(img, pt1, pt4, color, line_width)


# 统合图片预处理
def imagePreProcessing(path):
    image = cv2.imread(path)
    # 转灰度、降噪
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # image = cv2.GaussianBlur(image, (3,3), 0)

    # 边缘检测、寻找轮廓、确定顶点
    ratio = height_resize / image.shape[0]
    img = resizeImg(image)
    binary_img = getCanny(img)
    max_contour, max_area = findMaxContour(binary_img)
    box = getBoxPoint(max_contour)
    boxes = adapPoint(box, ratio)
    boxes = orderPoints(boxes)
    # 透视变化
    warped = warpImage(image, boxes)
    # 调整最终图片大小
    size = (width_resize, height_resize)
    warped = cv2.resize(warped, size, interpolation=cv2.INTER_CUBIC)

    # 画边缘框
    drawRect(image, tuple(boxes[0]), tuple(boxes[1]), tuple(boxes[2]), tuple(boxes[3]), (0, 0, 255), 2)
    cv2.imwrite("result/outline.jpg", image)

    return warped


# 截取图片中部分区域图像，测试阶段使用，包括显示与保存图片，实际使用时不使用这个函数，使用下面的正式版函数
def cropImage_test(img, crop_range, filename='Undefined'):
    xpos, ypos, width, height = crop_range
    crop = img[ypos:ypos + height, xpos:xpos + width]
    if filename == 'Undefined':  # 如果未指定文件名，采用坐标来指定文件名
        filename = 'crop-' + str(xpos) + '-' + str(ypos) + '-' + str(width) + '-' + str(height) + '.jpg'
    cv2.imshow(filename, crop)  # 展示截取区域图片---测试用
    # cv2.imwrite(filename, crop) #imwrite在文件名含有中文时会有乱码，应该采用下方imencode---测试用
    # 保存截取区域图片---测试用
    cv2.imencode('.jpg', crop)[1].tofile(filename)
    return crop


# 截取图片中部分区域图像
def cropImage(img, crop_range):
    xpos, ypos, width, height = crop_range
    crop = img[ypos:ypos + height, xpos:xpos + width]
    return crop


# 从截取图片中识别文字
def cropOCR(crop, ocrType):
    text_crop = ''
    if ocrType == 0:
        text_crop_list = ocr.ocr_for_single_line(crop)
    elif ocrType == 1:
        text_crop_list = ocr_numbers.ocr_for_single_line(crop)
    elif ocrType == 2:
        text_crop_list = ocr_UpperSerial.ocr_for_single_line(crop)
    elif ocrType == 3:
        text_crop_list = ocr.ocr(crop)
        for i in range(len(text_crop_list)):
            ocr_text = ''.join(text_crop_list[i]).split(':')[-1].split(';')[-1]
            # 如果出现- — _ ― 一律算作边框
            if '-' in ocr_text or '—' in ocr_text or '_' in ocr_text or '―' in ocr_text:
                continue
            text_crop = text_crop + ocr_text + ','
        return text_crop
    text_crop = ''.join(text_crop_list)
    return text_crop


def imageOcr(path):
    # 预处理图像
    # path = 'test.jpg'
    warped = imagePreProcessing(path)

    # 分块识别
    receipt = {}
    for i in range(len(crop_range_list_data)):
        crop = cropImage(warped, crop_range_list_data[i])
        crop_text = cropOCR(crop, crop_range_list_type[i])
        # 发票中不会有小写字母o l O，凡是出现o的都使用0替代，凡是出现l的都使用1替代，凡是出现O的都使用0替代，并去掉空格和冒号前面的字符
        crop_text = crop_text.replace('o', '0').replace(' ', '').replace('l', '1').replace('O', '0').split(':')[-1]

        # 销售方信息
        if crop_range_list_name[i] == 'seller':
            crop_text = crop_text.split(',')
            for i in range(4):
                if i < len(crop_text):
                    receipt.update({seller_dict[i]: crop_text[i]})

                else:
                    receipt.update({seller_dict[i]: ''})

        elif crop_range_list_name[i] == 'invoice':
            crop_text = crop_text.split(',')
            for i in range(4):
                if i < len(crop_text):
                    receipt.update({invoice_dict[i]: crop_text[i]})

                else:
                    receipt.update({invoice_dict[i]: ''})

        elif crop_range_list_name[i] == 'purchaser':
            crop_text = crop_text.split(',')
            for i in range(4):
                if i < len(crop_text):
                    receipt.update({purchaser_dict[i]: crop_text[i]})

                else:
                    receipt.update({purchaser_dict[i]: ''})

        else:
            if crop_range_list_name[i] == 'title':
                crop_text = crop_text[0:2] + '增值税普通发票'
            receipt.update({crop_range_list_name[i]: crop_text})

    receipt['sellerCode'] = receipt['sellerCode'].replace('工', '1').replace('.', '')
    receipt['purchaserCode'] = receipt['purchaserCode'].replace('工', '1').replace('.', '')
    for key in receipt:
        print(key + ':' + receipt[key])
    receipt.update({"serviceDetails": []})

    cv2.imwrite('result/block.jpg', warped)

    # 展示识别区域
    for i in range(len(crop_range_list_data)):
        warped = cv2.rectangle(warped, (crop_range_list_data[i][0], crop_range_list_data[i][1]),
                               (crop_range_list_data[i][0] + crop_range_list_data[i][2],
                                crop_range_list_data[i][1] + crop_range_list_data[i][3]),
                               (0, 0, 255), 2)

    # 展示与保存预处理的图片---测试用,生产环境会报错
    # cv2.namedWindow("warpImage", 0)
    # cv2.resizeWindow("warpImage", 1200, 700)
    # cv2.imshow('warpImage', warped)

    # 保存图片到本地
    cv2.imwrite('result/result.jpg', warped)
    return receipt


if __name__ == '__main__':
    print(imageOcr("test0.jpg"))
    # cv2.waitKey(0)




