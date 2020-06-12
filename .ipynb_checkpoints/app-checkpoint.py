from flask import Flask, render_template, redirect, request
import pathlib
import os
import tensorflow as tf 

import librosa.display, librosa
from librosa.util import normalize as normalize
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from keras import backend as K

app = Flask(__name__)

UPLOAD_FOLDER = 'static/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
MODEL_FOLDER = 'static/model'
app.config['UPLOAD_FOLDER'] = MODEL_FOLDER
model = tf.keras.models.load_model('./static/model/music.h5')
# tf.config.gpu.set_per_process_memory_fraction(0.4)
def pre_proces(path):
  y, sr = librosa.load(path, sr=44100)
  y, index = librosa.effects.trim(y,top_db=60) #Trim
  y = normalize(y)
  duration_in_samples=librosa.time_to_samples(1, sr=sr)
  y_pad = librosa.util.fix_length(y, duration_in_samples) #Pad/Trim to same duration
  y_stft=librosa.core.stft(y_pad, n_fft=n_fft, hop_length=hop_length)
  y_spec=librosa.amplitude_to_db(abs(y_stft), np.min)
  scaler = StandardScaler()
  dtype = K.floatx()
  data = scaler.fit_transform(y_spec).astype(dtype)
  data = np.expand_dims(data, axis=0)
  data = np.expand_dims(data, axis=3)
  return data

def predict(path,model):
  label = ['bassoon', 'cello', 'clarinet', 'flute', 'guitar', 'saxophone', 'trombone', 'trumpet', 'tuba', 'violin']
  a = model.predict(pre_proces(path))[0].tolist()
  num = a.index(max(a))
  return label[num]

@app.route('/')
def index():
    path = UPLOAD_FOLDER
    data_root = pathlib.Path(path)
    all_image_paths = list(data_root.glob('*'))
    all_image_paths = [str(path).split('/')[-1] for path in all_image_paths]
    if len(all_image_paths) > 0:
        path = all_image_paths[-1]
        pre = predict(path,model)
    else: pre = 'nofile'
    return render_template('index.html',lists=pre)
@app.route('/upload', methods = ['GET','POST'])
def upload():
    if request.method == "POST":
        files = request.files["file"]
        files.save(os.path.join(app.config['UPLOAD_FOLDER'], files.filename))
        return redirect('/')
    
    return render_template('upload.html')

if __name__ == '__main__':
  app.run(host='127.0.0.1', port=8000, debug=True)
 