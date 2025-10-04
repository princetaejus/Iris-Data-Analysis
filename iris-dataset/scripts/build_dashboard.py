import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
import plotly.express as px

# Make dirs
os.makedirs('reports/images', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Load data
iris = datasets.load_iris(as_frame=True)
iris_df = iris.frame.copy()
iris_df['species'] = iris_df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
iris_df.to_csv('data/iris_sample.csv', index=False)

# Static images
sns.pairplot(iris_df, hue='species', corner=True)
plt.show()
plt.savefig('reports/images/pairplot.png')

corr = iris_df.drop(columns=['target','species']).corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='viridis')
plt.show()
plt.savefig('reports/images/corr_heatmap.png')

for col in ['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']:
    plt.figure(figsize=(5,3))
    sns.histplot(iris_df, x=col, hue='species', element='step', stat='density', common_norm=False)
    plt.title(col)
    plt.tight_layout()
    plt.show()
    plt.savefig('reports/images/dist_' + col.replace(' ','_').replace('(','').replace(')','') + '.png')

# Interactive HTML
scatter = px.scatter_matrix(iris_df, dimensions=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)'], color='species', title='Iris Scatter Matrix')
heatmap_fig = px.imshow(corr.values, x=corr.columns, y=corr.columns, color_continuous_scale='Viridis', title='Correlation Heatmap')
dists = [px.histogram(iris_df, x=c, color='species', barmode='overlay', nbins=20, title=c) for c in ['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']]
from plotly.io import to_html
scatter_html = to_html(scatter, include_plotlyjs='cdn', full_html=False)
heatmap_html = to_html(heatmap_fig, include_plotlyjs=False, full_html=False)
dist_html = ''.join([to_html(fig, include_plotlyjs=False, full_html=False) for fig in dists])
html = '<!DOCTYPE html>
<html>
<head>
<meta charset=utf-8 />
<title>Iris Dashboard</title>
<style>body { font-family: Arial, sans-serif; margin: 0; } .header { padding: 14px 20px; background: #111827; color: #fff; } .tabs { display: flex; gap: 8px; padding: 10px 12px; background: #f3f4f6; position: sticky; top: 0; z-index: 2; } .tab { padding: 8px 12px; border-radius: 6px; background: #e5e7eb; cursor: pointer; user-select: none; } .tab.active { background: #2563eb; color: #fff; } .section { display: none; padding: 10px 12px; } .section.active { display: block; } .footer { padding: 10px 12px; color: #6b7280; font-size: 12px; border-top: 1px solid #e5e7eb; }</style>
</head>
<body>
<div class=header><h2>Iris Mini-Dashboard</h2><div>Interactive views: scatter matrix, correlations, and feature distributions.</div></div>
<div class=tabs><div class='tab active' data-target='scatter'>Scatter Matrix</div><div class=tab data-target=heatmap>Correlations</div><div class=tab data-target=dists>Distributions</div></div>
<div id=scatter class='section active'>' + scatter_html + '</div>
<div id=heatmap class=section>' + heatmap_html + '</div>
<div id=dists class=section>' + dist_html + '</div>
<div class=footer>Built by Julius (https://julius.ai)</div>
<script>const tabs = document.querySelectorAll('.tab'); const sections = document.querySelectorAll('.section'); tabs.forEach(function(t){ t.addEventListener('click', function(){ tabs.forEach(function(x){ x.classList.remove('active'); }); sections.forEach(function(s){ s.classList.remove('active'); }); t.classList.add('active'); document.getElementById(t.getAttribute('data-target')).classList.add('active'); }); });</script>
</body>
</html>'
with open('reports/iris_dashboard.html', 'w', encoding='utf-8') as f:
    f.write(html)
print('reports/iris_dashboard.html')
