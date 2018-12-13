import os


if __name__ == '__main__':
    train_path = '../data/NSMC/ratings_train.txt'
    train_out_path = '../data/NSMC/train.txt'
    test_path = '../data/NSMC/ratings_test.txt'
    test_out_path = '../data/NSMC/test.txt'

    with open(train_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

        if os.path.exists(train_out_path):
            os.remove(train_out_path)
        with open(train_out_path, 'a', encoding='utf-8') as o:
            for line in lines:
                words = line.split()
                idx = words.pop(0)
                flag_ = False
                sentence = ''
                for word in words:
                    if not flag_:
                        flag_ = True
                    else:
                        sentence += ' '
                    sentence += word
                sentence += '\n'
                o.write(sentence)

        with open(test_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            if os.path.exists(test_out_path):
                os.remove(test_out_path)
            with open(test_out_path, 'a', encoding='utf-8') as o:
                for line in lines:
                    words = line.split()
                    idx = words.pop(0)
                    flag_ = False
                    sentence = ''
                    for word in words:
                        if not flag_:
                            flag_ = True
                        else:
                            sentence += ' '
                        sentence += word
                    sentence += '\n'
                    o.write(sentence)