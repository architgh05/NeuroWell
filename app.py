import os
import sqlite3
import numpy as np
import tensorflow as tf
import requests
from flask import Flask, render_template, request, redirect, url_for, session, g, jsonify
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "your_secret_key"
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load models
model1 = tf.keras.models.load_model("model1.h5")  # Alzheimer's model
model2 = tf.keras.models.load_model("model2.h5")  # Brain Tumor model

# Database setup
DATABASE = "users.db"

def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        db.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        email TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL)''')
        db.commit()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/both")
def both():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("both.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        user = get_db().execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if user and check_password_hash(user[2], password):
            session["user_id"] = user[0]
            return redirect(url_for("both"))
        else:
            return render_template("login.html", message="Invalid email or password.")
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])
        try:
            db = get_db()
            db.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, password))
            db.commit()
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            return render_template("signup.html", message="Email already exists.")
    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/alzy", methods=["GET", "POST"])
def image_check():
    if "user_id" not in session:
        return redirect(url_for("login"))
    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            return render_template("alzy.html", error="No file selected")
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        image = tf.keras.preprocessing.image.load_img(filepath, target_size=(128, 128))
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = model1.predict(img_array)
        result = "Positive for Alzheimer's" if prediction[0][0] > 0.5 else "Negative for Alzheimer's"
        return render_template("alzy.html", prediction=result, filepath='/' + filepath.replace('\\', '/'))

    return render_template("alzy.html")

@app.route("/braintumor", methods=["GET", "POST"])
def brain_tumor():
    if "user_id" not in session:
        return redirect(url_for("login"))
    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            return render_template("braintumor.html", error="No file selected")
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        image = tf.keras.preprocessing.image.load_img(filepath, target_size=(128, 128))
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = model2.predict(img_array)
        class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
        result = class_names[np.argmax(prediction)]
        return render_template("braintumor.html", prediction=result, filepath='/' + filepath.replace('\\', '/'))

    return render_template("braintumor.html")

@app.route("/about")
def about():
    return render_template("about.html")


@app.route('/library')
def library():
    return render_template('library.html')



@app.route('/libcard1')
def libcard1():
    return render_template('libcard1.html')


@app.route('/libcard2')
def libcard2():
    return render_template('libcard2.html')


@app.route('/libcard3')
def libcard3():
    return render_template('libcard3.html')

@app.route("/get_live_data")
def get_live_data():
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "country": "in",
        "category":"general",
        "apiKey": "c63a6d326c244f179e8e3b3f3bba0e7e" # Make sure to export this in your environment
    }
    response = requests.get(url, params=params)
    return jsonify(response.json())


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
