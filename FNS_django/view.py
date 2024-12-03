from django.shortcuts import render
import pickle 
from sklearn.feature_extraction.text import TfidfVectorizer 
import json
from django.http import JsonResponse
# if u Want to use change the link to the model cuz i ended up adding it later alone 
model_path = 'C:/Users/hp/Desktop/FNS/model_and_vectorizer.pkl'
def home(request):
    return render(request,'page.html')

with open(model_path, 'rb') as model_file:
    best_model, vectorizer = pickle.load(model_file)


def model(request):
    if request.method == 'POST' : 
        data = json.loads(request.body.decode('utf-8'))
        text_input = data.get('text') 
        new_text_tfidf = vectorizer.transform([text_input])        
        prediction = best_model.predict(new_text_tfidf)
        probability = best_model.predict_proba(new_text_tfidf)
        print(prediction , probability)
        return JsonResponse({
                    'prediction': int(prediction[0]),
                    'probability': probability[0].tolist()  # Convert to list for JSON serialization
                })
        
