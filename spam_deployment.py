# Importing essential libraries
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
import nltk
import pandas as pd
import pickle

# Loading the dataset
df = pd.read_csv('./dataset/Spam_SMS_Collection.csv', sep='\t', names=['label', 'message'])

# Importing essential libraries for performing Natural Language Processing on 'SMS Spam Collection' dataset
nltk.download('stopwords')
# Cleaning the messages
corpus = []
ps = PorterStemmer()

for i in range(0, df.shape[0]):

    # Cleaning special character from the message
    message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=df.message[i])

    # Converting the entire message into lower case
    message = message.lower()

    # Tokenizing the review by words
    words = message.split()

    # Removing the stop words
    # Ponemos inglés porque nuestro dataset con el que será entrenado el 
    # proyecto sus palabras están en inglés 
    words = [word for word in words if word not in set(
        stopwords.words('english'))]

    # Este método spem lo que hace es cambiar las palabras de plural a singular
    words = [ps.stem(word) for word in words]

    # Joining the stemmed words
    message = ' '.join(words)

    # Building a corpus of messages
    corpus.append(message)

# Creating the Bag of Words model
cv = CountVectorizer(max_features=2500)
# Este método realiza el ajuste y la transformación en los datos de entrada de una sola vez 
# y convierte los puntos de datos. Si utilizamos el ajuste y la transformación por separado 
# cuando necesitamos ambos, entonces disminuirá la eficiencia del modelo, por lo que utilizamos
# fit_transform() que hará ambos trabajos
X = cv.fit_transform(corpus).toarray()

# Extracting dependent variable from the dataset
# Se utiliza para la manipulación de datos. 
# Convierte los datos categóricos en variables ficticias o indicadores.
y = pd.get_dummies(df['label'])
# es una técnica de selección basada en índices, 
# lo que significa que tenemos que pasar un índice entero
#  en el método para seleccionar una fila/columna específica.
#  Las entradas que se pueden utilizar 
# para . iloc son: Un entero. Una lista de enteros.
y = y.iloc[:, 1].values

# Creating a pickle file for the CountVectorizer
pickle.dump(cv, open('cv-transform.pkl', 'wb'))

# Model Building


#Es una función en la selección de modelos de Sklearn para dividir 
# las matrices de datos en dos subconjuntos: 
# para los datos de entrenamiento y para los datos de prueba. 
# Con esta función, no es necesario dividir el conjunto de datos manualmente.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0)

# Fitting Naive Bayes to the Training set
classifier = MultinomialNB(alpha=0.3)
# utilizamos la fórmula requerida y realizamos el cálculo sobre los valores
#  de las características de los datos de entrada y ajustamos este cálculo al transformador. 
classifier.fit(X_train, y_train)

# Creating a pickle file for the Multinomial Naive Bayes model
filename = 'spam-sms-mnb-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))
