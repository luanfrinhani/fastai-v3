
import flask, request, url_for, jsonify
from fastai.basic_train import load_learner
from fastai.vision import open_image
from flask_cors import CORS, cross_origin
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

 

@app.route('/')
 
learn = load_learner('https://drive.google.com/uc?export=download&id=1O19qAjzBJi4EnIFslRJxm1ttKy3CpbcC', 'Segmantation-1.pkl')
classes = learn.data.classes
logging.debug('Learnt classes')


def predict_single(img_file):
    '''function to take image and return prediction'''
    logging.debug('This function to take image and return prediction')

    prediction = learn.predict(open_image(img_file))
    probs_list = prediction  # [2].numpy()
    print("Inside predict_single")
    logging.debug('Before probs_list')
    return probs_list

 

@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':  
        logging.debug('Before my_prediction')
        my_prediction = predict_single(request.files['image'])        
        logging.debug('Before final_pred')
        final_pred = my_prediction[0]
        logging.debug('After final_pred')
    return render_template('results.html', prediction=final_pred,
                           comment='asd')
 
if __name__ == '__main__':
    app.run(debug=True)
