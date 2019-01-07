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

    def GB2312(self, index):
        # index = random.randint(0, len(self.gb2312_list)-1)
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
        self.mapping = ''
        self.chars = []
        self.single_chars = []
        self.single_images = []
        self.single_char_image = {}
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

    def randChinese(self, num):
        for i in range(0, num):
            char = self.randomChar.GB2312()
            while char in self.chars:
                char = self.randomChar.GB2312()
            self.single_chars.append(char)
            self.chars.append(char)
            x = 48*i
            y = 0
            text_image = Image.new('RGB', (56, 56), self.bgColor)
            draw = ImageDraw.Draw(text_image)
            draw.text((10, 10), char, font=self.font, fill=self.colorRandom2())
            del draw
            text_image = text_image.rotate(random.randint(-30, 30), expand=0)
            text_image = text_image.crop((6, 8, 46, 48))
            box = (x+4, y+4, x+44, y+44)
            self.image.paste(text_image, box)
        self.drawPoints(200)
        self.randLine(5)

    def create_single_images(self):
        for char in self.single_chars:
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
            self.single_char_image[char] = single_image

    def save(self, path):
        self.image.save(path)

    def save_single(self, dir, index):
        tmp_mapping = {}
        for char in self.single_chars:
            image = self.single_char_image[char]
            image.save(dir+'/'+str(index)+'.jpg')



def get_name(num):
    result = ''
    for i in range(5):
        result = str(num%10)+result
        num = num//10
    return result


if __name__=='__main__':
    dir = 'output/train'
    if not os.path.exists(dir):
        os.makedirs(dir)
    labels_file = open(dir+'/labels.txt', 'w')
    randomChar = RandomChar('level1')
    i = 0
    for n in range(3755):
        for m in range(4):
            char_code = ImageChar(randomChar=randomChar)
            char = randomChar.GB2312(n)
            char_code.single_chars.append(char)
            char_code.create_single_images()
            index = get_name(i)
            char_code.save_single(dir, index)
            labels_file.write(index+','+"".join(char_code.single_chars)+'\n')
            i += 1
            print(n)
    labels_file.close()
