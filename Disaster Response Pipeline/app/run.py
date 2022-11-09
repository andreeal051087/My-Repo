import json
import plotly
import pandas as pd
import plotly.express as px


from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DIS_RESP', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    categ_msg_counts = df.iloc[:,4:].sum(axis = 0, skipna = True).sort_values(ascending = False)
    categ_names = categ_msg_counts.index.tolist()
    perc_categ_msg=df.iloc[:,4:].sum(axis = 0, skipna = True).sort_values(ascending = False)/df.shape[0]
    
    matrix_msg_count=df.groupby(by='genre')[df.iloc[:,4:].columns].sum()
    matrix_msg_count_s=matrix_msg_count[matrix_msg_count.sum().sort_values(ascending = False).index]

    matrix_genre=matrix_msg_count_s.index.tolist()
    matrix_categ=matrix_msg_count_s.columns.tolist()
    matrix_values=matrix_msg_count_s.values

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        
        # percentage_msg_per_categories
        {
            'data': [
                {  
                  'type': 'line',
                   'x': categ_names,
                   'y': perc_categ_msg,

                    
                }
            ],

            'layout': {
                'title': 'Percentage of Messages tagged per Category',
                'yaxis': {
                    'title': "% of Messages",
                    'tickformat': '%'
                },
                'xaxis': {
                    'title': "Categories"
                },
                'barmode':'stack'
                
                
            }
        },
        
        # count_msg_per_categories
        {
            'data': [
                {  
                  'type': 'bar',
                   'x': categ_names,
                   'y': categ_msg_counts,

                    
                }
            ],

            'layout': {
                'title': 'Count of Messages tagged per Category',
                'yaxis': {
                    'title': "Cnt of Messages",
                },
                'xaxis': {
                    'title': "Categories"
                },
                'barmode':'stack'
                
                
            }
        },
        
        # heatmap of msg count within genre, per categories
        {
            'data': [
                {  
                  'type': 'heatmap',
                   'x': matrix_categ,
                   'y': matrix_genre,
                   'z': matrix_values
                    
                }
            ],

            'layout': {
                'title': 'Heatmap of Message Count per Genre and Categories',
                'yaxis': {
                    'title': "Genre"
                },
                'xaxis': {
                    'title': "Categories"
                }
                
            }
        },
        
        # genre_msg_counts
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
                
            }
        }
        
        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()