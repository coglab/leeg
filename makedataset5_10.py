import glob
import plyvel
import time
import mne
import numpy as np
import pandas as pd
import pickle
import struct
from keras.utils import np_utils

def recoderesult2(data):
#Первая цифра: тип задания, 1 - вычитание
#                           2 - умножение
#Вторая цифра: тип события, 1 - крестик для фиксации внимания
#                           2 - предъявление задачи 
#                           3 - ответ 
#                           4 - конец текущего примера 
#                           5 - участник не успел ответить (лимит 10 или 12 секунд, точно не помню).
#Третья цифра: тип тренировки, 
#                           1 - текущий пример тренируется несколько дней подряд ("зубрежка")
#                           2 - текущий пример предъявляется один раз за все дни ("трансфер")
#Четвертая цифра: тип ответа, 
#                           1 - правильный ответ
#                           2 - неправильный ответ
#                           3 - нет ответа
    reject = 0
    ret = 0
    h = 0
    l = 0
    dt = str(data)
    if (dt[0] == '1'): 
        h = 0b0 # вычитание
    elif (dt[0] == '2'): 
        h = 0b1 # умножение
    else:
        reject = 1
    if (dt[1] != '2'): # если это не предъявление задачи - отбрасываем
        reject = 1
    ret = h
    if (reject == 1):
        ret = 99
    return(ret)

def readraweeg(fname, diap):
    raw = mne.io.read_raw_brainvision(fname, montage=None, eog=('HEOGL', 'HEOGR', 'VEOG'), misc='auto', scale=1.0, preload=True, response_trig_shift=0, event_id=None, verbose=None)
    raw.set_eeg_reference('average', projection=False)  # set EEG average reference
    events = mne.find_events(raw, stim_channel='STI 014')
    event_id, tmin, tmax = None, diap[0], diap[1]
    #raw_no_ref, _ = mne.set_eeg_reference(raw, [])
    reject = dict(eeg=.05, eog=.05)
    epochs_params = dict(events=events, event_id=event_id, preload=True, 
                            tmin=tmin, tmax=tmax, reject=reject)
    epochs = mne.Epochs(raw, **epochs_params)
    alldata = epochs.get_data()

    allid = np.zeros((epochs.__len__(),), dtype = int)
    for i in range(epochs.__len__()):
        allid[i] = list(epochs[i].event_id.values())[0]
    del(raw)
    del(epochs)
    return(alldata, allid)


def putdb(datax, datay, db, index, log, num = 0):
    for i in range(datax.shape[0]):
        print(log, ": ", i)
        dtx = datax[i].astype(np.float32)
        dty = recoderesult2(datay[i])
        if (dty < 3):
            num += 1
            dtyc = np_utils.to_categorical(dty, 2)
            dtyc.shape = (1, 2)
            df = (dtx, dtyc, log)
            df1 = pickle.dumps(df, protocol = 2)
            db.put(struct.pack('>l', num), df1)
            del(dtx)
            del(df)
            del(df1)
    return(num)


if __name__ == '__main__':

    db = plyvel.DB('../eegdb5_10/', create_if_missing=True)

    a = glob.glob('../Raw/*.vhdr')
    j = 0
    k = str(len(a))
    num = 0
    for i in a:
        j += 1
        log = str(j) + "/" + k + " " + str(i)
        alldata, allid = readraweeg(i, (-.5, .5))
        num = putdb(alldata, allid, db, j, log, num = num)
    indexdb = np.random.randint(num, size = num)
    np.savetxt("indexdb5_10.csv", indexdb)
    print(num)
    db.close()
