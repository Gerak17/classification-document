# Algorithme pour classifier par catégorie automatiquement des documents (factures, contrats, devoirs)
import os                        
import numpy as np             
import pandas as pd           
import tensorflow as tf       
import fitz                  
import docx                  
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Fonction pour extraire du texte dans un fichier pdf
def extract_text_from_pdf(path):
    text = ""
    doc = fitz.open(path)
    for page in doc:
        text += page.get_text()  
    return text

# Fonction pour extraire du texte dans un fichier word
def extract_text_from_word(path):
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)  

# Fonction pour charger les documents de chaque catégories
def load_documents_from_folder(folder_path):
    documents = []  
    labels = []     
    
    for category in os.listdir(folder_path):
        category_path = os.path.join(folder_path, category)
        if not os.path.isdir(category_path):
            continue
        for file_name in os.listdir(category_path):
            file_path = os.path.abspath(os.path.join(category_path, file_name))
            if not os.path.isfile(file_path):
                print(f"Fichier non trouvé : {file_path}")
                continue
            try:
                if file_name.endswith(".pdf"):
                    text = extract_text_from_pdf(file_path)
                elif file_name.endswith(".docx"):
                    text = extract_text_from_word(file_path)
                else:
                    continue
                if not text.strip():
                    print(f"Fichier vide ou sans texte lisible : {file_path}")
                    continue
                documents.append(text)
                labels.append(category)  
            except Exception as e:
                print(f"Erreur avec {file_path} : {e}")
    return documents, labels

# Données d'entrée 
data_folder = "data_documents"
docs, labels = load_documents_from_folder(data_folder)
print(f"{len(docs)} documents chargés depuis {data_folder}.")

# Vérification avant vectorisation
if not docs:
    raise ValueError("Aucun document n'a pu être chargé. Vérifie que les fichiers existent et contiennent du texte.")

# Transformation des textes en vecteurs numériques 
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(docs).toarray()

# Encodage des catégories textuelles en nombres 
encoder = LabelEncoder()
y = encoder.fit_transform(labels)
y = tf.keras.utils.to_categorical(y, num_classes=len(encoder.classes_))

# Séparation en jeu d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Définition du modèle MLP 
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),  
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(len(encoder.classes_), activation='softmax')  
])

# Compilation du modèle 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entraînement du modèle 
model.fit(X_train, y_train, epochs=15, batch_size=8, validation_split=0.1)

# Evaluation sur le jeu de test 
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n Précision sur le jeu de test : {accuracy:.2f}")

# Fonction de prédiction pour un nouveau document 
def predict_document(path):
    if path.endswith(".pdf"):
        text = extract_text_from_pdf(path)
    elif path.endswith(".docx"):
        text = extract_text_from_word(path)
    else:
        raise ValueError("Format non supporté")

    vec = vectorizer.transform([text]).toarray()
    pred = model.predict(vec)
    category = encoder.inverse_transform([np.argmax(pred)])
    return category[0]

# Utilisation :
predicted = predict_document("test_docs/test_devoir.pdf")
print("Catégorie prédite :", predicted)
