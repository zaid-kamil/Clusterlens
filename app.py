from flask import Flask, render_template, redirect, request, flash, session
from database import User, add_to_db, open_db, Dataset
from werkzeug.utils import secure_filename


import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from kneed import KneeLocator
os.environ["OMP_NUM_THREADS"] = '1'
# dbscan


def find_optimal_number_of_clusters(X, max_clusters=10):
    distortions = []
    storage_path = os.path.join('static', 'uploads')
    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)
    kn = KneeLocator(range(1, max_clusters + 1), distortions, curve='convex', direction='decreasing')
    plt.plot(range(1, max_clusters + 1), distortions, marker='o', markersize=10, alpha=0.5)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    print(f'Optimal number of clusters: {kn.knee}')
    plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', colors='orange')
    plt.xticks(range(1, max_clusters + 1))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.savefig(os.path.join(storage_path, 'elbow_method.png'), dpi=150, bbox_inches='tight')

    return kn.knee, os.path.join(storage_path, 'elbow_method.png')


def preprocess_data(X):
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    pca = PCA(n_components=2)
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca

def plot_clusters(X, labels, centers, title):
    storage_path = os.path.join('static', 'uploads')
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='rainbow', edgecolors='k',)
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.title(title)
    # remove spline
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(storage_path, title + '.png'), dpi=150, bbox_inches='tight')
    return os.path.join(storage_path, title + '.png')

def kmeans_clustering(X, n_clusters=2):
    X_pca = preprocess_data(X)
    print(X_pca.shape)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X_pca)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    graphpath = plot_clusters(X_pca, labels, centers, 'KMeans Clustering')
    return labels, centers, graphpath


def fuzzy_cmeans_clustering(X, n_clusters):
    X_pca = preprocess_data(X)
    gmm = GaussianMixture(n_components=n_clusters)
    gmm.fit(X_pca)
    labels = gmm.predict(X_pca)
    centers = gmm.means_
    graphpath = plot_clusters(X_pca, labels, centers, 'Fuzzy CMeans Clustering')
    return labels, centers, graphpath

def dbscan_clustering(X, min_samples=10, eps=0.3):
    storage_path = os.path.join('static', 'uploads')
    X_pca = preprocess_data(X)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_pca)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    fig, ax = plt.subplots(figsize=(12, 8))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        xy = X_pca[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=10)
        xy = X_pca[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('DBSCAN Clustering')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.savefig(os.path.join(storage_path, 'DBSCAN Clustering.png'), dpi=150, bbox_inches='tight')
    return labels, n_clusters_, n_noise_, os.path.join(storage_path, 'DBSCAN Clustering.png')

# df = pd.read_csv(r'../static\uploads\imdb_clean.csv')
# columns = df.select_dtypes(include='number').columns
# X = df[columns].values
# n = find_optimal_number_of_clusters(X, 10)
# kmeans_clustering(X, n)
# fuzzy_cmeans_clustering(X, n)
# dbscan_clustering(X, 10, 0.3)

app = Flask(__name__)
app.secret_key = 'thisissupersecretkeyfornoone'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx', 'xls'}


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if len(email) == 0 or len(password) == 0:
            flash("All fields are required", 'danger')
        db = open_db()
        user = db.query(User).filter_by(email=email, password=password).first()
        if user:
            session['username'] = user.username
            session['email'] = user.email
            session['id'] = user.id
            session['isauth'] = True
            flash("Login successful", 'success')
            return redirect('/')
        else:
            flash("Invalid credentials", 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        cpassword = request.form.get('cpassword')
        if len(username) == 0 or len(email) == 0 or len(password) == 0 or len(cpassword) == 0:
            flash("All fields are required", 'danger')
        user = User(username=username, email=email, password=password)
        add_to_db(user)
    return render_template('register.html')

# upload datasets
@app.route('/dataset/upload', methods=['GET', 'POST'])
def datasets():
    if request.method == 'POST':
        name = request.form.get('name')
        paths = request.files.getlist('path')
        if paths:
           for item in paths:
                filename = secure_filename(item.filename)
                if filename.split('.')[-1] not in app.config['ALLOWED_EXTENSIONS']:
                    flash("Invalid file format", 'danger')
                    return redirect(request.url)
                path = f"{app.config['UPLOAD_FOLDER']}/{filename}"
                # save
                item.save(path)     
                save_to_db = Dataset(name=name, path=path)
                add_to_db(save_to_db)
        flash("Dataset uploaded successfully", 'success')
        return redirect('/dataset/list')
       
    return render_template('datasets.html')

@app.route('/dataset/list')
def dataset_list():
    db = open_db()
    datasets = db.query(Dataset).all()
    return render_template('dataset_list.html', datasets=datasets)

@app.route('/dataset/delete/<int:id>')
def delete_dataset(id):
    db = open_db()
    dataset = db.query(Dataset).filter_by(id=id).first()
    db.delete(dataset)
    db.commit()
    return redirect('/dataset/list')

@app.route('/dataset/clustering/<int:id>')
def clustering(id):
    c_clusters = request.args.get('clusters', 2, type=int)
    db = open_db()
    dataset = db.query(Dataset).filter_by(id=id).first()
    path = dataset.path
    df = pd.read_csv(path)
    # drop unnamed columns like Unnamed: 0
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    columns = df.select_dtypes(include='number').columns
    X = df[columns].values
    n, elbow_graph = find_optimal_number_of_clusters(X, 10)
    kmeans, centers, kmeans_graph = kmeans_clustering(X, c_clusters)
    fuzzy, centers, fuzzy_graph = fuzzy_cmeans_clustering(X, c_clusters)
    dbscan, n_clusters, n_noise, dbscan_graph = dbscan_clustering(X, 10, 0.3)
    name = dataset.name
    table = df.select_dtypes(include='number').sample(100).to_html(classes='table table-striped table-hover table-bordered text-center')
    return render_template('clustering.html' , kmeans=kmeans_graph, dbscan=dbscan_graph, fuzzy=fuzzy_graph, name=name, table=table, elbow=elbow_graph, clusters=n, chosen_clusters=c_clusters)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
