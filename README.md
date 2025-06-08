# Classificateur Automatique de Documents PDF & Word

**Un outil intelligent pour classer automatiquement vos documents** (factures, contrats, devoirs, etc.) grâce à l’apprentissage automatique et au traitement automatique du langage naturel (NLP).

Ce projet utilise la vectorisation TF-IDF pour représenter le contenu textuel des documents, puis entraîne un réseau de neurones dense (MLP) avec TensorFlow afin de prédire la catégorie d’un document donné.



## Fonctionnalités clés

-  **Chargement automatique** des documents organisés par catégories dans des sous-dossiers (`data_documents/<catégorie>/`)
-  **Extraction de texte fiable** depuis des fichiers PDF (`.pdf`) et Word (`.docx`)
-  **Prétraitement du texte** via vectorisation TF-IDF (scikit-learn)
-  **Encodage des labels** pour les catégories cibles
-  **Modélisation et entraînement** d’un réseau de neurones MLP via TensorFlow/Keras
-  **Évaluation précise** du modèle avec métriques d’exactitude (accuracy)
-  **Prédiction simple** d’un nouveau document avec fonction dédiée



##  Technologies utilisées

| Technologie           | Rôle principal                                |
|-----------------------|-----------------------------------------------|
| Python 3.x            | Langage de programmation                      |
| TensorFlow / Keras    | Construction et entraînement du modèle MLP    |
| scikit-learn          | TF-IDF, encodage des labels, division dataset |
| NumPy & Pandas        | Manipulation efficace des données             |
| PyMuPDF (fitz)        | Extraction de texte depuis PDF                |
| python-docx           | Extraction de texte depuis Word               |



## Organisation des données

Pour fonctionner correctement, placez vos documents dans une arborescence comme suit :

data_documents/
├── contrats/
│ ├── contrat1.pdf
│ └── contrat2.docx
├── factures/
│ ├── facture1.pdf
│ └── facture2.docx
└── devoirs/
├── devoir1.pdf
└── devoir2.docx

Chaque dossier correspond à une **catégorie** qui servira d’étiquette pour l’apprentissage.


## Résultats attendus

- Chargement réussi de tous les documents valides
- Précision sur le jeu de test d’environ 92%
- Prédiction cohérente et rapide pour des documents inconnus