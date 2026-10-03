from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# Cấu hình trang phải
st.set_page_config(
    page_title="Demo Phân Loại Hoa Iris Dataset",
    layout="wide",
)

# Tải lại dữ liệu và tạo bảng phục vụ phần xem dữ liệu bonus.
def load_iris_dataframe():

    iris_data = load_iris()
    dataframe = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    dataframe["target"] = iris_data.target
    dataframe["species"] = dataframe["target"].map(
        dict({0:iris_data.target_names[0], 
            1:iris_data.target_names[1], 
            2:iris_data.target_names[2]})
    )
    return iris_data, dataframe

# Mô hình chính được huấn luyện trên toàn bộ dữ liệu Iris và chỉ tạo một lần.
@st.cache_resource
def train_app_model():
    iris_data = load_iris()
    app_model = GaussianNB()
    app_model.fit(iris_data.data, iris_data.target)
    return app_model

# Mô hình đánh giá riêng dùng đúng phép chia 80/20 của phần bonus.
@st.cache_data
def calculate_test_accuracy():
    iris_data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris_data.data,
        iris_data.target,
        test_size=0.2,
        random_state=42,
        stratify=iris_data.target,
    )
    evaluation_model = GaussianNB()
    evaluation_model.fit(X_train, y_train)
    predictions = evaluation_model.predict(X_test)
    return accuracy_score(y_test, predictions)

# Dự đoán tên loài và xác suất của cả ba lớp.
def predict_species(model, feature_values, target_names):
    input_data = np.array([feature_values], dtype=float)
    predicted_class = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]
    predicted_species = str(target_names[predicted_class])
    return predicted_species, probabilities

# Ánh xạ mỗi loài tới đúng file ảnh trong thư mục images.
def get_image_path(species_name):
    image_paths = {
        "setosa": Path("setosa.jpg"),
        "versicolor": Path("versicolor.jpg"),
        "virginica": Path("virginica.jpg"),
    }
    return Path(__file__).resolve().parent / image_paths[species_name]


iris, iris_df = load_iris_dataframe()
model = train_app_model()
test_accuracy = calculate_test_accuracy()

st.title("Demo Phân Loại Hoa Iris Dataset")
st.subheader("Mô hình: Gaussian Naive Bayes")
st.write(
    "Điều chỉnh bốn thông số ở cột bên trái, sau đó nhấn "
    "**Dự đoán ngay** để xem kết quả."
)

with st.sidebar:
    st.header("Thông tin mô hình")
    st.write("**Thuật toán:** Gaussian Naive Bayes")
    st.write("**Dữ liệu:** Iris Dataset")
    st.write("**Đánh giá:** Train/Test = 80/20")
    st.write("**random_state:** 42")
    st.write("**stratify:** y")
    st.metric("Độ chính xác trên tập Test", f"{test_accuracy:.2%}")
    st.caption(
        "Accuracy do mô hình đánh giá riêng tính trên tập Test; "
        "mô hình dự đoán của ứng dụng được huấn luyện trên toàn bộ dữ liệu."
    )

input_col, result_col = st.columns(2, gap="large")

with input_col:
    st.header("Thông số đầu vào")
    sepal_length = st.slider(
        "Chiều dài lá đài (Sepal Length - cm)",
        min_value=4.0, max_value=8.0, value=5.1, step=0.1,
    )
    sepal_width = st.slider(
        "Chiều rộng lá đài (Sepal Width - cm)",
        min_value=2.0, max_value=4.5, value=3.5, step=0.1,
    )
    petal_length = st.slider(
        "Chiều dài cánh hoa (Petal Length - cm)",
        min_value=1.0, max_value=7.0, value=1.4, step=0.1,
    )
    petal_width = st.slider(
        "Chiều rộng cánh hoa (Petal Width - cm)",
        min_value=0.1, max_value=2.5, value=0.2, step=0.1,
    )
    predict_button = st.button("Dự đoán ngay", type="primary", use_container_width=True)

with result_col:
    st.header("Kết quả dự đoán")
    if predict_button:
        feature_values = [sepal_length, sepal_width, petal_length, petal_width]
        species, probabilities = predict_species(
            model, feature_values, iris.target_names
        )

        st.markdown(f"## **{species.upper()}**")

        image_path = get_image_path(species)
        if image_path.is_file():
            st.image(str(image_path), caption=f"Hoa Iris {species.capitalize()}")
        else:
            st.warning(
                f"Không tìm thấy ảnh minh họa: {image_path}. "
                "Vui lòng thêm ảnh đúng tên vào thư mục images/."
            )

        st.subheader("Xác suất của từng loài")
        probability_df = pd.DataFrame(
            {"Loài": [name.capitalize() for name in iris.target_names],
             "Xác suất": probabilities}
        ).set_index("Loài")
        st.bar_chart(probability_df, y="Xác suất")
        for species_name, probability in probability_df["Xác suất"].items():
            st.write(f"**{species_name}:** {probability:.2%}")
    else:
        st.info("Nhấn **Dự đoán ngay** để hiển thị kết quả.")

with st.expander("Xem bảng dữ liệu Iris"):
    st.dataframe(iris_df, use_container_width=True)
