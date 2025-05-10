import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import joblib

# Load dataset and model
df = pd.read_csv("cleaned_data.csv")
model = joblib.load("charges_model.joblib")

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# App layout
app.layout = dbc.Container([
    html.H1("Medical Insurance Charges Prediction", style={"fontSize": "40px", "textAlign": "center", "marginBottom": "20px"}),
    
    html.H2("User Input Parameters", style={"fontSize": "22px", "color": "#2980B9", "marginBottom": "20px"}),
    
    dbc.Row([
        dbc.Col([
            dbc.Label("Age"),
            dcc.Slider(id='age', min=16, max=80, step=1, value=30, marks={i: str(i) for i in range(16, 81, 8)}),
            
            dbc.Label("Sex", style={"marginTop": "20px"}),
            dcc.Dropdown(id='sex', options=[{"label": i, "value": i} for i in df.sex.unique()], value=df.sex.unique()[0]),
            
            dbc.Label("BMI", style={"marginTop": "20px"}),
            dcc.Slider(id='bmi', min=14.0, max=80.0, step=0.1, value=25.0, marks={i: str(i) for i in range(14, 81, 8)}),
        ], width=6),
        
        dbc.Col([
            dbc.Label("Children"),
            dcc.Slider(id='children', min=0, max=10, step=1, value=0, marks={i: str(i) for i in range(0, 11)}),
            
            dbc.Label("Smoker", style={"marginTop": "20px"}),
            dcc.Dropdown(id='smoker', options=[{"label": i, "value": i} for i in df.smoker.unique()], value=df.smoker.unique()[0]),
            
            dbc.Label("Region", style={"marginTop": "20px"}),
            dcc.Dropdown(id='region', options=[{"label": i, "value": i} for i in df.region.unique()], value=df.region.unique()[0]),
        ], width=6)
    ]),
    
    dbc.Button("Predict", id="predict-button", color="primary", style={"marginTop": "30px", "width": "100%"}),
    
    html.Br(),
    html.Br(),
    
    html.Div(id='prediction-output', style={"textAlign": "center", "fontSize": "24px", "color": "#2C3E50", "fontWeight": "bold"})
], fluid=True)


# Callback to handle prediction
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('age', 'value'),
    State('sex', 'value'),
    State('bmi', 'value'),
    State('children', 'value'),
    State('smoker', 'value'),
    State('region', 'value')
)
def predict_charges(n_clicks, age, sex, bmi, children, smoker, region):
    if n_clicks is None:
        return ""
    
    input_df = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })
    
    prediction = model.predict(input_df)[0].round(2)
    return f"Predicted Charges: $ {prediction:,.2f}"


if __name__ == '__main__':
    app.run_server(debug=True)
