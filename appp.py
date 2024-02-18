from flask import Flask, render_template, request, redirect, session
from flask_sqlalchemy import SQLAlchemy
from Mudole import process_question_with_answer

app = Flask(__name__)

# Configure the SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SECRET_KEY'] = 'your_secret_key'  # Add a secret key for session management
db = SQLAlchemy(app)

# Define a User model for the database
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)

# Create the database tables (run this once to initialize the database)
with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('index.html')  # Use index.html as the login page

@app.route('/stufff')
def stufff():
    return render_template('stufff.html')

app.route('/submit_quiz', methods=['POST'])
def submit_quiz():
    print("Received a request to submit quiz.")
    quiz_id = request.form.get('quiz_id')
    question_answers = {key: request.form[key] for key in request.form.keys() if key.startswith('q')}
    
    print("Quiz ID:", quiz_id)
    print("Question Answers:", question_answers)

    # Process the quiz using GPT-3 or any other method
    result = process_question_with_answer(question_answers)
    
    print("Result:", result)

    return result  # Assuming process_question_with_answer returns the result as a string

@app.route('/upload_mp3', methods=['POST'])
def upload_mp3():
    # Handle the MP3 file upload logic here
    # You can access the uploaded file using request.files
    # Example: mp3_file = request.files['mp3_file']
    
    # Save the file to a location on your server
    # Example: mp3_file.save('path/to/save/mp3_file.mp3')
    
    return redirect('/dash')  # Redirect to the dashboard after file upload

@app.route('/dash')
def dash():
    # Check if the user is logged in
    if 'user_name' in session:
        user_name = session['user_name']
        return render_template('dash.html', user_name=user_name)
    else:
        return redirect('/')  # Redirect to the login page if not logged in

@app.route('/logout')
def logout():
    # Clear the user's session
    session.pop('user_name', None)
    return redirect('/')

@app.route('/register')
def register_page():
    return render_template('register.html')  # Assuming createAccount.html is your registration page

@app.route('/register', methods=['POST'])
def register():
    name = request.form.get('name')
    email = request.form.get('email')
    password = request.form.get('password')

    # Check if any of the required fields is empty
    if not name or not email or not password:
        return 'Invalid registration data. Please fill out all fields.'

    # Create a new User instance and add it to the database
    new_user = User(name=name, email=email, password=password)
    db.session.add(new_user)
    db.session.commit()

    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email')
    password = request.form.get('password')

    # Check if the provided email exists in the database and the password matches
    user = User.query.filter_by(email=email, password=password).first()
    if user:
        # Store the user's name in the session
        session['user_name'] = user.name
        # Redirect to the dash page upon successful login
        return redirect('/dash')
    else:
        return 'Invalid login credentials. Please try again.'

if __name__ == '__main__':
    app.run(debug=True)
