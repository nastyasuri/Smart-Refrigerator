import numpy as np
import matplotlib.pyplot as plt

import os
import copy
import time
from PIL import Image
from tqdm import tqdm_notebook

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import datasets, models, transforms
from flask import Flask

#from flask_ngrok import run_with_ngrok

from torchvision import models
import torch

import json
import cv2
from google.colab import drive
drive.mount('/content/drive')
model_new = torch.load('/content/drive/MyDrive/model.pt')
model_new.eval()
resnet_transforms = transforms.Compose([
        transforms.Resize(256), # размер каждой картинки будет приведен к 256*256
        transforms.CenterCrop(224), # у картинки будет вырезан центральный кусок размера 224*224
        transforms.ToTensor(), # картинка из питоновского массива переводится в формат torch.Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # значения пикселей картинки нормализуются
    ])
# Telebot
import telebot
import requests
import shutil
import cv2


class User:
    def __init__(self, id):
        self.id = id
        self.fridge = ""
        self.stadium = 1
        self.optimus_products = []
        self.real_products = []


# Сюда подставляете свой токен
bot = telebot.TeleBot('2052212770:AAGQuRAZLJi14WraAtDDWE6UKv3xb2OKhI0')
milk = ["Молоко", "Йогурт", "Творог", "Сметана", "Масло", "Обычный сыр", "Плавленный сыр"]
sausages = ["Варённая колбаса", "Копчённая колбаса"]
fruits = ["Красные яблоки", "Зелёные яблоки", "Бананы", "Белый виноград", "Чёрный виноград", "Грейпфрут", "Лимоны",
          "Мандарины", "Апельсины"]
vegetables = ["Огурцы", "Помидоры", "Красный перец", "Жёлтый перец"]
users = dict()


@bot.message_handler(commands=['start'])
def start_message(message):
    print(message.chat.id)
    users.update({message.chat.id: User(message.chat.id)})
    bot.send_message(message.chat.id,
                     'Приветствуем вас в боте-настройщике Умного холодильника! Если захотите перезаполнить анкету, напишите "Перезаполнить"')
    bot.send_message(message.chat.id, 'Пожалуйста, введите id вашего холодильника')


@bot.message_handler(content_types=['text'])
def send_text(message):
    if message.chat.id in users.keys():
        user = users[message.chat.id]
        if message.text.lower() == "перезаполнить":
            del users[user.id]
            start_message(message)
        else:
            if user.stadium == 1:
                if len(message.text) == 8 and message.text[:2].isalpha() and message.text[2:].isdigit():
                    user.fridge = message.text
                    bot.send_message(message.chat.id, 'Принято')
                    key = make_buttons(milk)
                    bot.send_message(message.chat.id,
                                     "Выберите молочные продукты, которые должны быть в вашем холодильнике",
                                     reply_markup=key)
                else:
                    bot.send_message(message.chat.id, 'Извините, неправильный id, введите ещё раз пожалуйста')
            elif user.stadium == 5:
                if message.text == "Сверить":
                    bot.send_message(message.chat.id, 'Пришлите фотографии всех продуктов, затем напишите "Всё"')
                elif bot.send_message == "Всё":
                    res = set(user.optimus_products).difference(set(user.real_products))
                    bot.send_message(message.chat.id, "Продукты, которые вам нужно купить:\n" + "\n".join(list(res)))
            else:
                if message.text.lower() == "посмотреть":
                    print(user.optimus_products)
                    sas = "\n".join(user.optimus_products)
                    bot.send_message(message.chat.id, 'Продукты, которые должны быть в вашем холодильнике:\n' + sas)


@bot.message_handler(content_types=["photo"])
def echo_message(message):
    file_info = bot.get_file(message.photo[-1].file_id)
    url = 'https://api.telegram.org/file/bot{0}/{1}'.format('2052212770:AAGQuRAZLJi14WraAtDDWE6UKv3xb2OKhI0',
                                                            file_info.file_path)
    response = requests.get(url, stream=True)
    my_folder = '/content/user_pictures'
    path = f'image.jpg'
    if response.status_code == 200:
        with open(path, 'wb') as f:
            response.raw.decode_content = True
            shutil.copyfileobj(response.raw, f)
    image = cv2.imread(path)
    print(image.shape)
    image = preprocess_image(image)
    tipe = predict(image)
    users[message.chat.id].real_products.append(tipe)
    bot.send_message(message.chat.id, "Принято")
    # Вернуть тип строкой


def after1(user):
    if user.stadium == 2:
        bot.send_message(user.id, 'Принято')
        key = make_buttons(sausages)
        bot.send_message(user.id, "Выберите колбасные продукты, которые должны быть в вашем холодильнике",
                         reply_markup=key)
    elif user.stadium == 3:
        bot.send_message(user.id, 'Принято')
        key = make_buttons(fruits)
        bot.send_message(user.id, "Выберите фрукты, которые должны быть в вашем холодильнике",
                         reply_markup=key)
    elif user.stadium == 4:
        bot.send_message(user.id, 'Принято')
        key = make_buttons(vegetables)
        bot.send_message(user.id, "Выберите овощи, которые должны быть в вашем холодильнике",
                         reply_markup=key)
    elif user.stadium == 5:
        bot.send_message(user.id, 'Принято')
        bot.send_message(user.id,
                         'Ваш выбор сохранён. Чтобы посмотреть его, напишите "Посмотреть".\nЧтобы заполнить анкету заново, напишите "Перезаполнить"\nЧтобы сверить перечень имеющихся продуктов со списком, напишите "Сверить"')


@bot.callback_query_handler(func=lambda c: True)
def inlin(c):
    user = users[c.from_user.id]
    if c.data == "Закончить выбор":
        users[c.from_user.id].stadium += 1
        after1(users[c.from_user.id])
    else:
        if not (c.data in user.optimus_products):
            users[c.from_user.id].optimus_products.append(c.data)
            bot.send_message(user.id, "Продукт {} добавлен".format(c.data.lower()))


def make_buttons(arr):
    arar = arr + ["Закончить выбор"]
    key = telebot.types.InlineKeyboardMarkup()
    key.add(*[telebot.types.InlineKeyboardButton(text=arar[i], callback_data=arar[i]) for i in range(0, len(arar))])
    return key


bot.polling()


def preprocess_image(img):
    # ImageNet images size
    img = cv2.resize(img, (244, 224))[:, :, ::-1]

    # numpy to torch; uint -> float32; [0, 255] -> [0, 1]
    img = torch.from_numpy(img.copy()).float() / 255.0  # H x W x C
    img = img.permute((2, 0, 1))  # C x H x W

    # ImageNet normalization
    mean = torch.FloatTensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.FloatTensor([0.229, 0.224, 0.225])[:, None, None]
    img = (img - mean) / std

    # batch dimension
    img = img[None, ...] # B x C x H x W = (1, 3, 224, 224)

    return img


def predict(image):
    result = model_new(image)
    max_score_id = result.view(-1).argmax().item()
    slovar = dict({0: "Зелёные яблоки", 1: "Бананы", 2: "Красные яблоки", 3: "Сыр", 4: "Огурцы", 5: "Чёрный виноград", 6: "Белый виноград", 7: "Грейпфрут",
                  8: "Лимоны", 9: "Мандарины", 10: "Масло", 11: "Молоко", 12: "Апельсин", 13: "Красный перец", 14: "Жёлтый перец", 15: "Сметана", 16: "Колбасы",
                  17: "Помидоры", 18: "Творог", 19: "Йогурты"})
    return slovar[max_score_id]