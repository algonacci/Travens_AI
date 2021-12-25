from flask import Flask, render_template, request
import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from tensorflow.keras.models import load_model
model = load_model('model.h5')
import json
import random
intents = json.loads(open('data.json',encoding='utf-8').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))
import pandas as pd
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
stop.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',''])



app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
  return render_template('index.html')


# Logic Chatbot
def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res



@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)


# Logic Restaurant Recommendation
food_data = pd.read_csv('Book1.csv', encoding='latin-1')
res_data = pd.read_csv('Book2.csv', encoding='latin-1')
res_data = res_data.loc[(res_data['Country Code'] == 94) & (res_data['City'] == 'Jakarta'), :]
res_data = res_data.loc[res_data['Longitude'] != 0, :]
res_data = res_data.loc[res_data['Latitude'] != 0, :]
res_data = res_data.loc[res_data['Latitude'] < 29] # clearing out invalid outlier
res_data = res_data.loc[res_data['Rating text'] != 'Not rated']
res_data['Cuisines'] = res_data['Cuisines'].astype(str)

def search_comfort(mood):
  lemmatizer = WordNetLemmatizer()
  foodcount = {}
  for i in range(124):
    temp = [temps.strip().replace('.','').replace(',','').lower() for temps in str(food_data["comfort_food_reasons"][i]).split(' ') if temps.strip() not in stop ]
    if mood in temp:
      foodtemp = [lemmatizer.lemmatize(temps.strip().replace('.','').replace(',','').lower()) for temps in str(food_data["comfort_food"][i]).split(',') if temps.strip() not in stop ]
      for a in foodtemp:
        if a not in foodcount.keys():
          foodcount[a] = 1 
        else:
          foodcount[a] += 1
  sorted_food = []
  sorted_food = sorted(foodcount, key=foodcount.get, reverse=True)
  return sorted_food

def find_my_comfort_food(mood):
  topn = []
  topn = search_comfort(mood) #function create dictionary only for particular mood
  return topn[:3]


@app.route('/restaurant-recommendation', methods=['GET'])
def restaurant_recommendation():
  return render_template('restaurant-recommendation.html')

@app.route('/find', methods=['GET'])
def find_restaurant():
  mood = request.args.get('mood')
  result = find_my_comfort_food(mood)
  result_str = 'You should eat {}, {}, or {}.'.format(result[0], result[1], result[2])
  food_to_cuisine_map = {
    "japanese": "japanese",
    "korean": "korean",
    "sunda": "sunda",
    "indonesian": "indonesian",
    "peranakan": "peranakan",
    "burger": "burger",
    "italian": "italian",
    "café": "café",
    "seafood": "seafood",
    "western food": "western food",
    "desserts": "desserts",
    "bakery": "bakery",
    "coffee and tea": "coffee and tea",
    "cafe": "italian",
    "ramen" : "japanese",
    "pizza": "pizza",
  }
  restaurants_list = []
  for item in result:
    restaurants = res_data[res_data.Cuisines.str.contains(food_to_cuisine_map[item], case=False)].sort_values(by='Aggregate rating', ascending=False).head(3)
    restaurants_list.append(restaurants.iloc[0])
    restaurants_list.append(restaurants.iloc[1])
    restaurants_list.append(restaurants.iloc[2])
  return render_template('restaurant-result.html', result = result_str, mood = mood, restaurants1 = restaurants_list[:3], restaurants2 = restaurants_list[3:6], restaurants3 = restaurants_list[6:])


# Other Services
@app.route("/other-services")
def other_services():
    return render_template("other_services.html")