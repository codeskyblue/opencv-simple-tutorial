from matplotlib import pyplot as plt
import cv2
import os
import typing
import numpy as np


def turn_gray(im):
    if len(im.shape) == 2:
        return im
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


class TImage:
    def __init__(self, img: typing.Union[np.ndarray, str, "TImage"], flags=None):
        self._fpath = None # for lazy load
        self._imread_flags = flags or cv2.IMREAD_COLOR

        if isinstance(img, TImage):
            self._im = img.data.copy()
        elif isinstance(img, np.ndarray):
            self._im = img
        elif isinstance(img, str):
            assert os.path.isfile(img), f"{img!r} not exists"
            self._fpath = img 
            self._im = None
        else:
            raise TypeError("Invalid type image", img)

    @property
    def data(self) -> np.ndarray:
        if self._im is None:
            self._im = cv2.imread(self._fpath, self._imread_flags)
        return self._im
    
    def is_gray(self) -> bool:
        return len(self.shape) == 2

    @property
    def rgbdata(self) -> np.ndarray:
        if len(self.shape) == 3 and self.shape[2] == 3:
            return cv2.cvtColor(self.data, cv2.COLOR_BGR2RGB)
        elif self.is_gray():
            gray = self.data
            return cv2.merge((gray, gray, gray))
        elif self.shape[2] == 4:
            return self.data[:, :, :3][:, :, ::-1] #cv2.cvtColor(self.data, cv2.COLOR_)
        else:
            raise NotImplementedError("Unsupported image shape", self.shape)

    @property
    def graydata(self):
        if self.is_gray():
            return self.data
        return turn_gray(self.data)

    @property
    def size(self):
        """
        Returns tuple of (width, height)
        """
        h, w = self.data.shape[:2]
        return (w, h)
    
    @property
    def shape(self):
        return self.data.shape

    def crop(self, box: tuple) -> "TImage":
        lx, ly, rx, ry = box
        return TImage(self.data[ly:ry, lx:rx])
    
    def rectangle(self, box, color=(0, 0, 255)) -> "TImage":
        lx, ly, rx, ry = box
        data2 = self.data.copy()
        cv2.rectangle(data2, (lx, ly), (rx, ry), color, 5)
        return TImage(data2)

    def save(self, filename):
        cv2.imwrite(filename, self.data)

    def show(self, title="my picture"):
        img = cv2.cvtColor(self.data, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(title)
        plt.show()


def imread(filename, flags=None) -> TImage:
    return TImage(filename, flags)


def show_image(img: typing.Union[list, np.ndarray, TImage], title=None):
    """ show picture in jupyter notebook """
    if isinstance(img, TImage):
        img.show(title=title)
    elif isinstance(img, np.ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(title)
        plt.show()
    elif isinstance(img, (list, tuple)):
        images = img
        n: int = len(images)

        titles = title
        if titles is None:
            titles = list(range(n))
        
        f = plt.figure()
        for i in range(n):
            tim = TImage(images[i]) #img = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            f.add_subplot(1, n, i + 1)
            plt.imshow(tim.rgbdata)
            plt.title(str(titles[i]))
        plt.show(block=True)


if __name__ == "__main__":
    # test code
    ti = TImage("testdata/icon.png")
    print(ti.size)
    ti.show()