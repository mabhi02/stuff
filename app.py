from flask import Flask, render_template, request, redirect, session, jsonify, url_for  
import os
import uuid
from flask_sqlalchemy import SQLAlchemy
song = ""
import numpy as np

app = Flask(__name__)
# Configure the SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SECRET_KEY'] = 'your_secret_key'  # Add a secret key for session management
db = SQLAlchemy(app)

ajay = np.array([0,1])


# Define a User model for the database
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    


def update_array(update):
    global yo
    yo = update

# Create the database tables (run this once to initialize the database)
with app.app_context():
    db.create_all()

personAVM = {
    'email': 'avm@avm.com',
    'total': 15,
    'avgComplexity': 10,
    'historicalData': [
        [4, 2, 9, 3, 2],
        [2, 2, 3, 4, 9]
    ]
}

personAylin = {
    'email': 'aylin@aylin.com',
    'total': 30,
    'avgComplexity': 10,
    'historicalData': [
        [1, 8, 3, 2, 9],
        [2, 3, 4, 5, 6]
    ]
}

personAtharva = {
    'email': 'atharva@atharva.com',
    'total': 10,
    'avgComplexity': 10,
    'historicalData': [
        [3, 5, 6, 2, 1],
        [1, 2, 4, 7, 3]
    ]
}

yolo = ["yo"]

storedPeople = [personAVM, personAylin, personAtharva]

email = yolo[-1]
print(yolo)
print(email)
numb = 0
for i in range(len(storedPeople)):
    if storedPeople[i]["email"] == email:
        numb = i
    



"""

prticipants = [
    {'AVM': [len(historicalData[0]), total, np.mean(historicalData[1])]},
    {'Joe': [20, 30, 3]},
    {'John': [18, 28, 5]}
]

"""

    


@app.route('/')
def index():
    return render_template('index.html')  # Use index.html as the login page


@app.route('/update_song', methods=['POST'])
def update_song():
    global song
    new_song = request.json.get('song')
    if new_song:
        song = new_song
        return jsonify({'message': 'Song updated successfully', 'song': song}), 200
    else:
        return jsonify({'error': 'No song provided in the request'}), 400

@app.route('/current_song', methods=['GET'])
def current_song():
    global song
    return jsonify({'song': song})

A=storedPeople[ajay[0]]['historicalData'][0]
B=storedPeople[ajay[0]]['historicalData'][1]
print("A")
print("B")
@app.route('/dashboard.js')
def dashboard():
    print("A:")
    print(A)
    print("B: ")
    print(B)
    return render_template('dashboard.js', A=A, B=B)

@app.route('/stufff')
def stufff():
    question1 = session.pop('question1', None)
    question2 = session.pop('question2', None)
    question3 = session.pop('question3', None)
    question4 = session.pop('question4', None)
    question5 = session.pop('question5', None)
    question6 = session.pop('question6', None)
    question7 = session.pop('question7', None)
    question8 = session.pop('question8', None)
    question9 = session.pop('question9', None)
    question10 = session.pop('question10', None)
    hint1 = session.pop('hint1', None)
    hint2 = session.pop('hint2', None)
    hint3 = session.pop('hint3', None)
    hint4 = session.pop('hint4', None)
    hint5 = session.pop('hint5', None)
    hint6 = session.pop('hint6', None)
    hint7 = session.pop('hint7', None)
    hint8 = session.pop('hint8', None)
    hint9 = session.pop('hint9', None)
    hint10 = session.pop('hint10', None)
    
    rendered_stufff = render_template('stufff.html', 
        question1=question1, 
        question2=question2, 
        question3=question3, 
        question4=question4, 
        question5=question5, 
        question6=question6, 
        question7=question7, 
        question8=question8, 
        question9=question9,
        question10=question10,   
        hint1 = hint1,
        hint2 = hint2,
        hint3 = hint3,
        hint4 = hint4,
        hint5 = hint5,
        hint6 = hint6,
        hint7 = hint7,
        hint8 = hint8,
        hint9 = hint9,
        hint10 = hint10, 
    )

    return rendered_stufff

@app.route('/Final')
def Final():
    return render_template('finalQuiz.html')

@app.route('/tests')
def tests():
    return render_template('test.html')

@app.route('/submit_quiz', methods=['POST'])
def submit_quiz():
    
    from model import checkAnswerWriting,  checkAnswerVocab, writingQ, vocabQ
    

    data = request.get_json()
    quiz_id = data.get('quiz_id')

    user_answers = [data['answers'][f'q{i+1}'] for i in range(10)]
    print(user_answers)
    count = 0
    for i in range(5):
        print(user_answers[i])
        print()
        var = checkAnswerWriting(i, user_answers[i], writingQ)
        if var == True:
            count+=1

    for i in range(5):
        k = i + 5
        var = checkAnswerVocab(i, user_answers[k], vocabQ)
        if var == True:
            count+=1
            
    is_correct = count
    from model import complexityWriting
    
    mean = sum(complexityWriting) / len(complexityWriting)

    print("email: ")
    email = session.get('user_email')
    print(email)
    
    for i in range(len(storedPeople)):
        if storedPeople[i] == email:
            storedPeople[i]['total'] = storedPeople[i]['total'] + count
            del storedPeople[i]['historicalData'][0][0]
            storedPeople[i]['historicalData'][0].append(count)
            del storedPeople[i]['historicalData'][1][0]
            storedPeople[i]['historicalData'][1].append(mean)            
            storedPeople[i]['avgComplexity'] = ((storedPeople[i]['avgComplexity']) + mean)/2
           
    print(storedPeople)
    print("Number that are corrent is:")
    print(is_correct)
    print("-----------------------------")
    flash(is_correct, 'is_correct')

    # Redirect the user to the /stufff route
    return redirect(url_for('stufff'))

@app.route('/submitFinalquiz', methods=['POST'])
def submitFinalquiz():
    try:
        print("Received POST request")
        data = request.get_json()
        print("Received JSON data:", data)

        quiz_id = data.get('quiz_id')
        answers = {
            'quiz': ['Acepta los desafios de la vida y vive al maximo, baila, rie y disfruta', 'Abrace la vida, rie, baila, y deja ir el dolor', 'Afronta los desafios de la vida con risas y baile', 'Abrace la vida, rie, baila, y deja ir el dolor', 'Acepta los desafios de la vida y vive al maximo, baila, rie y disfruta', 'llorar', 'encontrar', 'vivirla', 'gozar', 'limpiar']
        }

        user_answers = [data['answers'][f'q{i+1}'] for i in range(10)]

        correct_count = sum(user_ans.lower() == correct_ans.lower() for user_ans, correct_ans in zip(user_answers, answers[quiz_id]))

        result_message = f'You got {correct_count} out of 10 questions correct!'

        # Return a proper JSON response
        return jsonify({'result': result_message})
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()  # Add this line to print the traceback
    return jsonify({'result': f'An error occurred during quiz submission. {str(e)}'}), 500

UPLOAD_FOLDER = 'songs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'mp3'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_mp3', methods=['POST'])
def upload_mp3():
    myGoat = True
    # Delete existing files in the 'static/songs' folder
    for existing_file in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], existing_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting file: {e}")

    if 'mp3_file' not in request.files:
        return redirect(request.url)

    mp3_file = request.files['mp3_file']

    if mp3_file.filename == '':
        return redirect(request.url)

    if not allowed_file(mp3_file.filename):
        return redirect(request.url)

    # Generate a unique filename
    filename = str(uuid.uuid4()) + '.' + mp3_file.filename.rsplit('.', 1)[1].lower()

    # Save the file to the upload directory
    mp3_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    from model import myGoat, questions, answers, complexityWriting, writingH, hints, vocabH, writingQ, vocabQ, audiotoText, translate_text, getWritingDict, checkAnswerWriting, checkAnswerVocab, get_normalized_complexity, doVocab, hintVocab, hintWriting, questionNumber
    
    email = session.get('user_email')

    
    boool = False
    for i in range(len(storedPeople)):
        if storedPeople[i] == email:
            boool == True
    
    if boool == True:
        newPerson = {
        'email': email,
        'total': 0,
        'avgComplexity': 0,
        'historicalData': [
            [0],
            [0]]}
        
        storedPeople.append(newPerson)
    
    # Redirect to /dash (assuming this is where you want to go after rendering the templates)
    session['question1'] = questions[0]
    session['question2'] = questions[1]
    session['question3'] = questions[2]
    session['question4'] = questions[3]
    session['question5'] = questions[4]
    session['question6'] = questions[5]
    session['question7'] = questions[6]
    session['question8'] = questions[7]
    session['question9'] = questions[8]
    session['question10'] = questions[9]
    
    session['answer1'] = answers[0]
    session['answer2'] = answers[1]
    session['answer3'] = answers[2]
    session['answer4'] = answers[3]
    session['answer5'] = answers[4]
    session['answer6'] = answers[5]
    session['answer7'] = answers[6]
    session['answer8'] = answers[7]
    session['answer9'] = answers[8]
    session['answer10'] = answers[9]
    print(answers)
    session['hint1'] = hints[0]
    session['hint2'] = hints[1]
    session['hint3'] = hints[2]
    session['hint4'] = hints[3]
    session['hint5'] = hints[4]
    session['hint6'] = hints[5]
    session['hint7'] = hints[6]
    session['hint8'] = hints[7]
    session['hint9'] = hints[8]
    session['hint10'] = hints[9]
    
    
    # Redirect to /dash (assuming this is where you want to go after rendering the templates)
    return redirect(url_for('dash'))
           

@app.route('/dash')
def dash():
    # Check if the user is logged in
    if 'user_name' in session:
        user_name = session['user_name']
        
    if 'user_email' in session:
        user_email = session['user_email']
        email = user_email
        
         
        numb = 0
        for i in range(len(storedPeople)):
            if storedPeople[i]["email"] == email:
                numb = i
            
        
        print('aightlol')
        print(email)
        
        A=storedPeople[ajay[0]]['historicalData'][0]
        B=storedPeople[ajay[0]]['historicalData'][1]
        
        print("----A&B-------")
        print(A)
        print(B)
        
        return render_template('dash.html', user_name=user_name, A=A, B=B)
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
        session['user_email'] = user.email
        user_email = session.get('user_email')
        yolo.append(user_email)
        print(yolo)
        print("sdlksdfj")
        print(user_email)
        update_array(user_email)
        
        # Redirect to the dash page upon successful login
        return redirect('/dash')
    else:
        return 'Invalid login credentials. Please try again.'

if __name__ == '__main__':
    app.run(debug=True)
