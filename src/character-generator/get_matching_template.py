# -*- coding: utf-8 -*-
import random
import os
from PIL import Image, ImageDraw, ImageFont
import pandas as pd


class RandomChar():
    """用于随机生成汉字"""

    def __init__(self, set_name):
        self.gb2312_list = []
        self.gb2312_to_unicode = {}
        self.unicode_to_gb2312 = {}
        self.gb2312_to_Chinese = {}
        self.unicode_to_Chinese = {}
        self.load_character_set(set_name)

    def load_character_set(self, set_name):
        if set_name=='level1':
            table = pd.read_csv("character_set/gb2312_level1.csv")
        elif set_name=='level2':
            table = pd.read_csv("character_set/gb2312_level2.csv")
        character_list = [item for item in table.values]
        for character in character_list:
            self.gb2312_list.append(character[1])
            self.gb2312_to_unicode[character[1]] = character[0]
            self.unicode_to_gb2312[character[0]] = character[1]
            self.gb2312_to_Chinese[character[1]] = character[2]
            self.unicode_to_Chinese[character[0]] = character[2]

    def GB2312(self):
        index = random.randint(0, len(self.gb2312_list)-1)
        gb2312 = self.gb2312_list[index]
        unicode = self.gb2312_to_unicode[gb2312]
        return chr(int(unicode, 16))


class ImageChar():
    def __init__(self, randomChar=None, fontColor=(0, 0, 0), size=(192, 48), fontPath='msyhbd.ttc',
                 bgColor=(255, 255, 255), fontSize=28):
        self.size = size
        self.fontPath = fontPath
        self.bgColor = bgColor
        self.fontSize = fontSize
        self.fontColor = fontColor
        self.font = ImageFont.truetype(self.fontPath, self.fontSize)
        self.image = Image.new('RGB', size, bgColor)
        self.chars = []
        self.single_images = []
        self.char2image = {}
        self.randomChar = randomChar

    def rotate(self):
        self.image.rotate(random.randint(0, 30), expand=0)

    def drawText(self, pos, txt, fill):
        draw = ImageDraw.Draw(self.image)
        draw.text(pos, txt, font=self.font, fill=fill)
        del draw

    def randRGB(self):
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    # 随机颜色
    def colorRandom1(self):
        return (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))

    # 随机长生颜色2
    def colorRandom2(self):
        return (random.randint(32, 127), random.randint(32, 127), random.randint(32, 127))

    def randPoint(self):
        (width, height) = self.size
        return (random.randint(0, width), random.randint(0, height))

    def randLine(self, num):
        draw = ImageDraw.Draw(self.image)
        for i in range(0, num):
            draw.line([self.randPoint(), self.randPoint()], self.randRGB())
        del draw

    def drawPoints(self, num):
        draw = ImageDraw.Draw(self.image)
        for i in range(0, num):
            draw.point(self.randPoint(), self.randRGB())
        del draw

    def create_single_images(self):
        for char_gb2312 in self.randomChar.gb2312_list:
            char = chr(int(self.randomChar.gb2312_to_unicode[char_gb2312],16))
            self.chars.append(char)
            single_image = Image.new('RGB', (48, 48), self.bgColor)
            text_image = Image.new('RGB', (56, 56), self.bgColor)
            draw = ImageDraw.Draw(text_image)
            draw.text((10, 10), char, font=self.font, fill=self.colorRandom2())
            del draw
            text_image = text_image.rotate(random.randint(-30, 30), expand=0)
            text_image = text_image.crop((6, 8, 46, 48))
            box = (4, 4, 44, 44)
            single_image.paste(text_image, box)
            draw = ImageDraw.Draw(single_image)
            # 划线
            for i in range(0, 3):
                draw.line(
                    [(random.randint(0, 48), random.randint(0, 48)), (random.randint(0, 48), random.randint(0, 48))],
                    self.randRGB())
            # 画点
            for i in range(0, 80):
                draw.point((random.randint(0, 48), random.randint(0, 48)), self.randRGB())
            del draw
            self.char2image[char] = single_image

    def save_single(self, dir):
        for index, char in enumerate(self.chars):
            image = self.char2image[char]
            image.save(dir+'/'+char+'.jpg')
            print(index)


def generate_matching_template(part=None, level=None):
    dir = 'output/template/'+part
    if not os.path.exists(dir):
        os.makedirs(dir)
    randomChar = RandomChar(level)
    char_code = ImageChar(randomChar=randomChar)
    char_code.create_single_images()
    char_code.save_single(dir+'/')


if __name__=='__main__':
    generate_matching_template(part='seen', level='level1')
    generate_matching_template(part='unseen', level='level2')
