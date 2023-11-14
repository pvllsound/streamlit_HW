import streamlit as st
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=302)

clf = DecisionTreeClassifier().fit(X_train, y_train)

# Streamlit app
st.title("Iris Species Predictor")

# Input fields
sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, value=4.0)
petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, value=1.0)

# Prediction
if st.button("Predict"):
    prediction = clf.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    species = iris.target_names[prediction][0]
    st.write(f"The predicted species is: {species}")