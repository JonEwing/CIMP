import plotly.express as px
import pandas as pd

df = pd.read_csv("rules_ap.csv")

fig = px.scatter_3d(df, x = "support", y = 'confidence', z ='# Rule mutations', color = 'Support * Confidence')

fig.write_html("graphic.html")