from flask import Flask, render_template, redirect, request
import pathlib
import os
import tensorflow as tf 
import numpy as np
import glob
import librosa.display, librosa
from librosa.util import normalize as normalize
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import random

app = Flask(__name__)

UPLOAD_FOLDER = 'static/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
IMG_FOLDER = 'static/img'
app.config['IMG_FOLDER'] = IMG_FOLDER
STYLES_FOLDER = 'static/style'
app.config['STYLES_FOLDER'] = STYLES_FOLDER
model = tf.keras.models.load_model('./static/model/model441x2.h5')
# tf.config.gpu.set_per_process_memory_fraction(0.4)
# def pre_proces(path):
#     data_list=[]
#     y, sr = librosa.load(path)
#     num = int(len(y)/66150)
#     for i in range(num):
#         start = 66150 * i
#         finish = start + 66150
#         mfcc = librosa.feature.mfcc(y[start:finish], sr, n_mfcc=13, n_fft=2048, hop_length=512)
#         mfcc = mfcc.T.tolist()
#         iput = np.array(mfcc)
#         data = np.expand_dims(iput, axis=0)
#         data_list.append(data)
#     return data_list

def get_lsix(flist):
  li_ix = []
  a = list(map(int,flist))
  print (a)
  for i in a:
    if i > 0.5:
      li_ix.append(a.index(i))
  print (li_ix)
  return li_ix

def get_ls_lb(li_ix):
  label = ['cello', 'clarinet', 'flute', 'acoustic guitar', 'electronic guitar', 'organ', 'piano', 'saxophone', 'trumpet','violin','human voice']
  ls_lb = []
  for i in li_ix:
    ls_lb.append(label[i])
  return ls_lb
def get_lb(ls):
  lb = []
  for i in ls:
    if len(i) !=0:
      for j in i:
        lb.append(j)
  return lb


def predict(path,model):

    sr=44100
    pre_list=[]
    s, r = librosa.load(path,sr=sr)
    num3s = sr*3
    num = int(len(s)/num3s)
    
    for i in range(num):
        start = num3s * i
        finish = start + num3s
        mfcc = librosa.feature.mfcc(s[start:finish], r, n_mfcc=13, n_fft=2048, hop_length=512)
        mfcc = mfcc.T.tolist()
        iput = np.array(mfcc)
        data = np.expand_dims(iput, axis=0)
        pre = np.round(model.predict(data),2).tolist()[0]
        li_ix = get_lsix(pre)
        ls_lb = get_ls_lb(li_ix)
        # pre_list.append([pre,label[inx]])
        pre_list.append(ls_lb)
        print(start)    
    pre_list = get_lb(pre_list)
        
    return set(pre_list)
def dele():
  files = glob.glob('./static/upload/*.wav')
  if len(files) !=0:
    for f in files:
      try:
          os.remove(f)
      except OSError as e:
          print("Error: %s : %s" % (f, e.strerror))

def get_file():
    path = UPLOAD_FOLDER
    data_root = pathlib.Path(path)
    allpaths = list(data_root.glob('*'))
    allpaths = [str(path).split('/')[-1] for path in allpaths]    
    fpath = allpaths[-1]
    return fpath
@app.route('/')
def index():
    dele()
    return render_template('index.html')

@app.route('/upload', methods = ['GET','POST'])
def upload():
    if request.method == "POST":
        files = request.files["file"]
        lissss = [i for i in range(100)]

        files.save(os.path.join(app.config['UPLOAD_FOLDER'], files.filename))
        f = get_file()
    return render_template('upload.html',f = f)
@app.route('/predicted', methods = ['GET','POST'])
def predicted():
    lb = ['cello', 'clarinet', 'flute', 'acoustic guitar', 'electronic guitar', 'organ', 'piano', 'saxophone', 'trumpet','violin','human voice']
    if request.method =="POST":
        f = get_file()
        p_l = []
        print(f)
        pre = predict(f,model)
        for i in pre:
            nb = lb.index(i)
            img = '../static/img/imgi/'+str(nb)+'.jpg'
            p_l.append([img,i])
        if len(p_l) == 0:
            p_l = [['ssss','No idea what kind of instrument']]
    return render_template('predict.html', pre = p_l)

if __name__ == '__main__':
  app.run(host='127.0.0.1', port=8000, debug=True)
 