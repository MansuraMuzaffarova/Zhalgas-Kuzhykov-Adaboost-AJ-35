# Zhalgas-Kuzhykov-Adaboost-AJ-35
# Импортируем необходимые библиотеки
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Загрузим датасет Diabetes
data = load_diabetes()
X = data.data
y = (data.target > 150).astype(int)  # Бинарная классификация на основе порогового значения

# Разделим данные на обучающий и тестовый набор
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создадим AdaBoostClassifier с DecisionTreeClassifier в качестве базовой модели
clf = AdaBoostClassifier(n_estimators=50, random_state=42)

# Обучим классификатор
clf.fit(X_train, y_train)

# Сделаем предсказания на тестовом наборе
y_pred = clf.predict(X_test)

# Оценим производительность классификатора
accuracy = accuracy_score(y_test, y_pred)
print("Точность AdaBoost на тестовом наборе: {:.2f}%".format(accuracy * 100))
