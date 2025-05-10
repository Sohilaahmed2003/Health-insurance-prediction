import joblib
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

@st.cache_data()
def load_data():
    try:
        df = pd.read_csv(r"c:\Users\sohila\Downloads\cleaned_data.csv")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load the dataset
df = load_data()

# Check if data loaded correctly
if df is not None:
    try:
        model = joblib.load(r"c:\Users\sohila\Documents\charges_model.joblib")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    st.markdown("""
        <style>
            body {
                background-color: #F4F6F7;
            }
            h1 {
                font-size: 40px;
                color: #2E86C1;
                text-align: center;
                margin-bottom: 20px;
            }
            h2 {
                font-size: 24px;
                color: #2874A6;
            }
            .stButton>button {
                background-color: #3498DB;
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 18px;
            }
            .stButton>button:hover {
                background-color: #5DADE2;
            }
            .prediction-box {
                background-color: #D6EAF8;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                font-size: 28px;
                font-weight: bold;
                color: #1A5276;
                margin-top: 20px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("\U0001F3E5 Medical Insurance Charges Prediction")
    st.subheader("Fill in your details:")

    def user_input_features():
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", min_value=16, max_value=80, value=30)
            sex = st.selectbox("Sex", df['sex'].unique())
            bmi = st.slider("BMI", min_value=14.0, max_value=80.0, value=25.0)

        with col2:
            children = st.slider("Number of Children", min_value=0, max_value=10, value=0)
            smoker = st.selectbox("Smoker", df['smoker'].unique())
            region = st.selectbox("Region", df['region'].unique())

        data = {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region
        }
        return pd.DataFrame(data, index=[0])

    user_input = user_input_features()

    if st.button("\U0001F52E Predict Charges"):
        try:
            prediction = model.predict(user_input)[0].round(2)
            st.markdown(f"""
                <div class="prediction-box">
                    \U0001F3F7️ Estimated Insurance Charges: <br> <span style='color:#154360;'>${prediction:,.2f}</span>
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error during prediction: {e}")

    # Visualization Section
    st.markdown("---")
    st.markdown("### \U0001F4CA Interactive Data Visualizations")
    with st.expander("Click to select features and show charts"):
        st.markdown("#### Univariate Analysis")
        uni_col = st.selectbox("Select numerical column for univariate chart:", df.select_dtypes(include=['int64', 'float64']).columns)
        fig_uni = px.histogram(df, x=uni_col, marginal="box", title=f"Distribution of {uni_col}")
        st.plotly_chart(fig_uni)

        st.markdown("#### Bivariate Analysis")
        x_bi = st.selectbox("Select X-axis column:", df.columns)
        y_bi = st.selectbox("Select Y-axis column:", df.columns)
        color_bi = st.selectbox("Select categorical column for color:", df.select_dtypes(include='object').columns)
        fig_bi = px.scatter(df, x=x_bi, y=y_bi, color=color_bi, title=f"{x_bi} vs {y_bi} by {color_bi}")
        st.plotly_chart(fig_bi)

        st.markdown("#### Multivariate Analysis")
        multi_cols = st.multiselect("Select multiple columns for scatter matrix:", df.select_dtypes(include=['int64', 'float64']).columns, default=['age', 'bmi', 'charges'])
        color_multi = st.selectbox("Select categorical column for color coding:", df.select_dtypes(include='object').columns)
        if len(multi_cols) >= 2:
            fig_multi = px.scatter_matrix(df, dimensions=multi_cols, color=color_multi, title="Scatter Matrix")
            st.plotly_chart(fig_multi)
else:
    st.error("Failed to load data. Please check your CSV file path or content.")

