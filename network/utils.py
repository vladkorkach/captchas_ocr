import os
import numpy as np
import tensorflow as tf
import cv2
import csv

from config import FLAGS

# letters + digits + ?space + ?blank
num_classes = 26 + 10 + 1 + 1

maxPrintLen = 100


charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
encode_maps = {}
decode_maps = {}
for i, char in enumerate(charset, 1):
    encode_maps[char] = i
    decode_maps[i] = char

SPACE_INDEX = 0
SPACE_TOKEN = ''
encode_maps[SPACE_TOKEN] = SPACE_INDEX
decode_maps[SPACE_INDEX] = SPACE_TOKEN


class DataIterator:
    def __init__(self, data_dir):
        self.image = []
        self.labels = []
        for root, sub_folder, file_list in os.walk(data_dir):
            for file_path in file_list:
                image_name = os.path.join(root, file_path)
                if not image_name.endswith('.jpg'):
                    continue

                if FLAGS.image_channel == 1:
                    im = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
                else:
                    im = cv2.imread(image_name, 1).astype(np.float32) / 255.

                im = cv2.resize(im, (FLAGS.image_width, FLAGS.image_height))
                im = np.reshape(im, [FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
                self.image.append(im)

                code = image_name.split('/')[-1].split('.')[0]

                code = code.upper()
                # print(code)
                code = [SPACE_INDEX if code == SPACE_TOKEN else encode_maps[c] for c in list(code)]
                self.labels.append(code)

    @property
    def size(self):
        return len(self.labels)

    def the_label(self, indexs):
        labels = []
        for i in indexs:
            labels.append(self.labels[i])

        return labels

    def input_index_generate_batch(self, index=None):
        if index:
            image_batch = [self.image[i] for i in index]
            label_batch = [self.labels[i] for i in index]
        else:
            image_batch = self.image
            label_batch = self.labels

        def get_input_lens(sequences):
            lengths = np.asarray([FLAGS.out_channels for _ in sequences], dtype=np.int64)

            return sequences, lengths

        batch_inputs, batch_seq_len = get_input_lens(np.array(image_batch))
        batch_labels = sparse_tuple_from_label(label_batch)

        return batch_inputs, batch_seq_len, batch_labels


def accuracy_calculation(original_seq, decoded_seq, ignore_value=-1, isPrint=False, epoch=1):
    if len(original_seq) != len(decoded_seq):
        print('original lengths is different from the decoded_seq, please check again')
        return 0
    count = 0
    dc = 0
    tmp_results = []
    for i, origin_label in enumerate(original_seq):
        decoded_label = [j for j in decoded_seq[i] if j != ignore_value]
        if isPrint and i < maxPrintLen:
            if len(decoded_label) == len(origin_label):
                corr = 0
                for i, d in enumerate(decoded_label):
                    if decoded_label[i] == origin_label[i]:
                        corr += 1
                percents = [corr/len(decoded_label)]
                tmp_results.append(str(origin_label) + '\t' + str(decoded_label) + '\t' + str(percents))
            else:
                tmp_results.append(str(origin_label) + '\t' + str(decoded_label))

        if origin_label == decoded_label:
            count += 1
        for i, d in enumerate(decoded_label):
            if d in(origin_label):
                dc += 1/len(origin_label)

    # print(count)
    with open('{}/{}_test.csv'.format(FLAGS.train_labels, str(epoch)), 'a') as f:
        for ddd in tmp_results:
            f.write(ddd)
            f.write('\n')
    return count * 1.0 / len(original_seq), dc / len(original_seq)


def sparse_tuple_from_label(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def eval_expression(encoded_list):
    """
    :param encoded_list:
    :return:
    """

    eval_rs = []
    for item in encoded_list:
        try:
            rs = str(eval(item))
            eval_rs.append(rs)
        except:
            eval_rs.append(item)
            continue

    with open('./result.txt') as f:
        for ith in range(len(encoded_list)):
            f.write(encoded_list[ith] + ' ' + eval_rs[ith] + '\n')

    return eval_rs


def train_labels_metrics_reader(epoch, show_incorrect=False):
    if not os.path.isfile('{}/{}_test.csv'.format(FLAGS.train_labels, str(epoch))):
        return "No metrics for this epoch"
    with open('{}/{}_test.csv'.format(FLAGS.train_labels, str(epoch)), 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter="\t")
        inc = 0
        t = 0
        for row in reader:
            t+= 1
            if str(row[0]) != str(row[1]):
                inc += 1
                print("====>")
                row[0] = row[0].replace("[","").replace("]","").replace(" ","").split(',')
                row[1] = row[1].replace("[","").replace("]","").replace(" ","").split(',')
                str_o = ""
                str_p = ""
                for i, s in enumerate(row[0]):
                    if s.isdigit():
                        str_o += decode_maps[int(s)]
                        if i < len(row[1]):
                            str_p += decode_maps[int(row[1][i])]
                if len(row) > 2:
                    print(str_o, " ", str_p, "captcha_acc => ", row[2])
                # show incorrect recognized data
                if show_incorrect:
                    image_name = "{}{}.jpg".format(FLAGS.val_dir, str_o)
                    print(image_name)
                    cv2.imshow(str_p, cv2.imread(image_name, 1))
                    cv2.waitKey()
                    cv2.destroyAllWindows()
        print("incorrect recognized number:", inc, ", total validation images:", t, ", Percent of incorrect", (inc/t)*100)
