from flask import Flask, render_template
from flask import request, jsonify
from flask_cors import CORS
import pandas as pd

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
# import torchvision.transforms as transforms

import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Model.list() 

app = Flask(__name__)
CORS(app)
HOST = "localhost"
PORT = 80


@app.route("/")
def home():
    return render_template("index.html")

message = None


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']

    if file.filename == '':
        return "No selected file"
    
    if file:
        # read data from CSV file frontend
        inf_df = pd.read_csv(file, header=None)

        X_inf = inf_df.loc[:, :186]
        y_inf = np.array(inf_df.loc[:, 187:])
        X_inf = torch.tensor(X_inf.values)
        X_inf = torch.reshape(X_inf, (11, 17))
        X_inf = X_inf.unsqueeze(0).unsqueeze(0)
        X_inf = F.interpolate(X_inf, size=(112, 112), mode='nearest')
        X_inf = X_inf.type(torch.float32)

        y_inf = np.array(y_inf).astype(int).squeeze()
        y_inf = torch.tensor(y_inf)

        # img_size = 112
        # inf_transform = transforms.Compose([
        #     transforms.Resize((img_size, img_size)),
        #     transforms.ToTensor(),
        # ])

        model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model.features[-1].fc = nn.AdaptiveAvgPool2d(output_size=1)
        model.avgpool = nn.Identity()
        model.classifier.fc = nn.Linear(1000, 512)
        model.classifier.fc1 = nn.Linear(512, 128)
        model.classifier.fc2 = nn.Linear(128, 5)
        model.load_state_dict(torch.load('effnn.pth', map_location=torch.device('cpu')))
        model.eval()

        with torch.no_grad():
            inps = X_inf
            labels = y_inf
            labels = labels.type(torch.LongTensor)
            outputs = model(inps)
            print(f'prediction:{outputs}')
            print(f'prediction:{torch.argmax(outputs)}')

            # return f'predictions: {torch.argmax(outputs)}'
    prediction_list = ['Normal beat', 'Supraventricular premature beat',
        'Premature ventricular contraction',
        'Fusion of ventricular and normal beat',
        'Unclassifiable beat']
    prediction_result = prediction_list[torch.argmax(outputs)]

    
    message_list = ['정상적인 심장 박동을 보이는 사람에게 심혈관 질환을 예방할 수 있도록 조언해줘.',
                    'Supraventricular premature beat이 나타나는 사람에게 심혈관 질환과 관련하여 조언해줘.',
                    'Premature ventricular contraction이 나타나는 사람에게 심혈관 질환과 관련하여 조언해줘.',
                    'Fusion of ventricular and normal beat이 나타나는 사람에게 심혈관 질환과 관련하여 조언해줘.',
                    '현재 심장 박동 패턴을 정확히 파악할 수는 없지만, 일반적인 사람에게 권장되는 심혈관 질환과 관련된 조언을 해줘.']

    # Call the chat GPT API
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
         {"role": "user", "content": f"{message_list[torch.argmax(outputs)]}"},
        ],
        temperature=0,
        max_tokens=1024
    )
    # print(completion["choices"][0]["message"]["content"].encode("utf-8").decode())

    return jsonify({"prediction": prediction_result, "recommendation": completion["choices"][0]["message"]["content"].encode("utf-8").decode()})



if __name__ == "__main__":
    app.run(debug=True, host=HOST, port=PORT)