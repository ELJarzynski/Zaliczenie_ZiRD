import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


"""Descripton"""
# Projekt ma na celu najepszej klasyfikacji zadowolenia klientow lini lotniczych, database składa się z 22 kolumn,
# w którym są zaprezentowane czynniki które mają wpływ na zadowolenie pasażerów

""" ---------------------------------------- DATA PREPARING ---------------------------------------- """
"""Settings of terminal setup"""
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Wczytywani pliku CSV
file_directory = r"C:\Users\kamil\Desktop\Studia\Semestr VI\MachinLearning\Kamil-Jarzynski-164395\Airline_customer_satisfaction.csv"
df = pd.read_csv(file_directory)
print(df.info())
"""Dropping missing cols"""
# Usuniecie wierszy które maja wybrakowane wartosci w kolumnie Arrival Delay in Minutes jest tak duzo przykladow
# ze nie maja wpływu na końcowy wynik
df = df.dropna(subset=['Arrival Delay in Minutes'])

"""Defining different groups of columns that will be processed using ColumnTransformer"""
# Definiowanie kolumn dla różnych enkoderów
ordinal_cols = ['satisfaction']
onehot_cols = ['Customer Type', 'Class', 'Type of Travel']

# Definiowanie kolumny, które mają być przeskalowane
scaling_cols = ['Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes', 'Departure/Arrival time convenient']

"""Preprocessing and scaling using ColumnTransformer"""
# Inicjalizacja enkoderów
ordinal_preprocessor = OrdinalEncoder()
onehot_preprocessor = OneHotEncoder()

# Inicjalizacja estymatora
scaler_pipeline = make_pipeline(
    MinMaxScaler()
)

# Konfiguracja ColumnTransformer
preprocessing_pipeline = ColumnTransformer(
    transformers=[
        ('ordinal', ordinal_preprocessor, ordinal_cols),
        ('onehot', onehot_preprocessor, onehot_cols),
        ('scaling', scaler_pipeline, scaling_cols)
    ]
)

"""Making new DataFrame"""
# Transformacja danych i utworzenie nowego DataFrame z przekszatłconymi danymi
data_preprocessed = pd.DataFrame(
    preprocessing_pipeline.fit_transform(df),
    columns=ordinal_cols + list(preprocessing_pipeline.named_transformers_['onehot'].get_feature_names_out(onehot_cols)) + scaling_cols,
    index=df.index
)

# Wstawienie kolumn po użyciu ColumnTransform i usuniecie ich starych odpowieników
df = df.drop(columns=scaling_cols + ordinal_cols + onehot_cols).join(data_preprocessed)

# Ustawienie kolumny satisfaction na pierwszy index kolumn
df.insert(0, 'satisfaction', df.pop('satisfaction'))
print(df.head())

"""--------------------------------------------------Modeling----------------------------------------------------------"""
def metrics(y_validation, y_predicted_val):
    """Define for predictions errors"""
    # Stworzyłem fukcje metrics do wyświetlania wyników czułościi, precyzji, dokładności i macierzy pomyłek
    print("Recall:", recall_score(y_validation, y_predicted_val))
    print("Precision:", precision_score(y_validation, y_predicted_val))
    print("Accuracy:", accuracy_score(y_val, y_predicted_val))
    print("Confusion Matrix:", confusion_matrix(y_validation, y_predicted_val))


"""Train test split was used to split dataset into training and testing sets"""
# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['satisfaction']), df['satisfaction'],
                                                    test_size=0.2, random_state=42)

"""Splitting the training set into training and validation sets"""
# Zastosowanie walidacji krzyżowej, ponieważ pozwala ona na zminimalizowanie wpływu losowego podziału danych
# na jakość modelu oraz wyniki są uśredniane, co daje bardziej stabilną ocenę jakości modelu
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

"""Model initialization"""
# Używam RandomForestClassifier do predykcji wyników, najlepiej odnalazł się dla tego zbioru danych z najmniejszym błędem
model = RandomForestClassifier(
    n_estimators=1000,
    criterion='log_loss',
    max_depth=40,
    min_samples_split=2,
    min_samples_leaf=4,
    class_weight='balanced_subsample',
    n_jobs=-1,
)

# Trenowanie modelu
model.fit(X_train, y_train)

# Predykcja wartości dla zbioru walidacyjnego
y_predict_val = model.predict(X_val)

# Użycie funkcji metrics do końcowego wyświetlenia wyników
metrics(y_val, y_predict_val)

# Wizlizacja macierzy pomyłek
cm = confusion_matrix(y_val, y_predict_val)
CMD = ConfusionMatrixDisplay(confusion_matrix=cm)
CMD.plot(cmap='Blues')
plt.show()