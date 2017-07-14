import numpy as np
import math


class Loader:

    def __init__(self, batch_size=1, split_fractions = [0.9, 0.1]):
        self.batch_size = batch_size

        self.rooms = np.load('./data/room.npy')
        self.layouts = np.load('./data/layout.npy')

        # only consider category without rotation
        # self.rooms = self.rooms[:, :, :, :9]
        # self.layouts = self.layouts[:, :, :, :9]

        self.room_width = self.rooms.shape[1]
        self.room_nc = self.rooms.shape[-1]
        self.layouts_nc = self.layouts.shape[-1]

        assert(len(self.rooms) == len(self.layouts))
        self.num_batches = math.floor(self.layouts.shape[0] / batch_size)

        self.ntrain = int(self.num_batches * split_fractions[0])
        self.nval = self.num_batches - self.ntrain

        self.split_sizes = [self.ntrain, self.nval]
        self.batch_ix = [0, 0]

        # raw training data
        rooms_train = self.rooms[:batch_size * self.ntrain]
        layouts_train = self.layouts[:batch_size * self.ntrain]

        rooms_test = self.rooms[-batch_size * self.nval:]
        layouts_test = self.layouts[-batch_size * self.nval:]

        # batch training data
        self.rooms_train_batch = np.array(np.split(rooms_train, self.ntrain))
        self.layouts_train_batch = np.array(np.split(layouts_train, self.ntrain))

        self.rooms_test_batch = np.array(np.split(rooms_test, self.nval))
        self.layouts_test_batch = np.array(np.split(layouts_test, self.nval))

    def width(self):
        return self.room_width

    def next_batch(self, split_index):
        pointer = self.batch_ix[split_index]

        if split_index == 0:
            room, layout = self.rooms_train_batch[pointer], self.layouts_train_batch[pointer]
        elif split_index == 1:
            room, layout = self.rooms_test_batch[pointer], self.layouts_test_batch[pointer]

        self.batch_ix[split_index] = (self.batch_ix[split_index] + 1) % self.split_sizes[split_index]

        if split_index == 0 and self.batch_ix[0] == 0:
            indices = np.arange(self.ntrain)
            np.random.shuffle(indices)
            self.rooms_train_batch = self.rooms_train_batch[indices]
            self.layouts_train_batch = self.layouts_train_batch[indices]

        return room, layout


if __name__ == '__main__':
    loader = Loader(1)
    print(loader.rooms.shape[0])
    print(len(loader.rooms))
    print(loader.num_batches)
    print(loader.room_width)
    print(loader.room_nc)
    print(loader.layouts_nc)
    room, layout = loader.next_batch(0)

