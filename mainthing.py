import librosa
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from flask import Flask,render_template,request,jsonify
from flask_wtf import FlaskForm
from wtforms import FileField,SubmitField
from wtforms.validators import DataRequired

def transform(waveform):
    if(len(waveform)<66150):
        waveform = np.pad(waveform, (0, max(0, 66150 - len(waveform))), mode='constant')
    else: waveform=waveform[:66150]
    mfcc = librosa.feature.mfcc(y=waveform, sr=22050, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    tempo, beats = librosa.beat.beat_track(y=waveform, sr=22050)
    tempo_ = librosa.util.sync(mfcc, beats)
    tempo_tensor = torch.tensor(tempo_, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    tempo_resized = F.interpolate(tempo_tensor, size=(40, 130), mode='bilinear', align_corners=False)
    tempo_resized = tempo_resized.squeeze(0).squeeze(0)
    tempo_resized = np.array(tempo_resized)
    chroma = librosa.feature.chroma_stft(y=waveform, sr=22050)
    chroma_tensor = torch.tensor(chroma, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    chroma_resized = F.interpolate(chroma_tensor, size=(40, 130), mode='bilinear', align_corners=False)
    chroma_resized = chroma_resized.squeeze(0).squeeze(0)
    chroma_resized = np.array(chroma_resized)
    im2d = np.stack([mfcc, delta, delta2, tempo_resized, chroma_resized], axis=-1)
    return im2d


class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.dropout_fc = nn.Dropout(0.3)
        self.out = nn.Linear(128, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn_fc(x)
        x = self.dropout_fc(x)
        x = self.out(x)
        return F.log_softmax(x, dim=1)

model = CNNClassifier()
model.to(torch.device("cuda"))
model.load_state_dict(torch.load("ckpt/withweightdecay5channels.ckpt")["model_state_dict"])
model.eval()

labels = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]

app = Flask(__name__)
app.secret_key = 'nummadethamarasserychoramnu'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

class Myform(FlaskForm):
    audio = FileField('Audio', validators=[DataRequired()])
    submit = SubmitField('submit')

@app.route('/')
def index():
    form = Myform()
    return render_template('index.html',form=form)
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        y, _ = librosa.load(file, sr=22050,mono=True)
        l = len(y)
        i=1
        x = []
        while(i*66150<l and i<10):
            start = i*66150
            end = start+66150
            chunk = y[start:end]
            if len(chunk)<66150:
                break
            chunk = transform(chunk)
            if len(chunk.shape)!=3 or 0 in chunk.shape:
                i+=1
                continue
            x.append(chunk)
            i+=1
        x = torch.from_numpy(np.stack(x)).permute(0, 3, 1, 2).float().to("cuda")
        pred = model(x)
        pred = pred.argmax(dim=1)
        unique_vals, counts = torch.unique(pred, return_counts=True)
        max_count_index = torch.argmax(counts)
        most_frequent = unique_vals[max_count_index].item()
        return jsonify({"genre": labels[most_frequent]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)