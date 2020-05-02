import time
from pandarallel import pandarallel
from text_preprocessing import preprocess_sentence
from train_classifier import load_data


def test_preprocess_sentence():
    s = '''This is a group of examples tò tést, among other things,
           the correctness of 1,334.32 words released on 4-11'''
    result = preprocess_sentence(s)
    actual = 'group example test among thing correctness one three three ' \
             'four three two word release four one one'
    print(result)
    assert actual == result


def test_speed_single_proc():
    X, Y, _ = load_data('data/DisasterResponse.db')
    print('Start single proc test...') # 208.13 seconds
    start = time.time()
    X['message'] = X.message.apply(preprocess_sentence)
    end = time.time()
    print(f'Total time single proc = {end - start:.2f} seconds')
    assert 0 == 0
        

def test_speed_multi_proc():
    X, Y, _ = load_data('data/DisasterResponse.db')
    print('Start multi proc test...') # 108.74 seconds
    start = time.time()
    X['message'] = X.message.parallel_apply(preprocess_sentence)
    end = time.time()
    print(f'Total time multi proc = {end - start:.2f} seconds')
    assert 0 == 0