from PIL import Image
import os
pathes = ["./test/unseen/", "./test/seen/",
          "./train/", "./template/unseen/", "./template/seen/"]
for path in pathes:
    for index, file in enumerate(os.listdir(path)):
        if file.endswith("jpg"):
            image = Image.open(path+file)
            image = image.resize((36, 36))
            image.save(path+file)
        print(index)