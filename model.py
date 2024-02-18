from tqdm.auto import tqdm  # this is our progress bar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pinecone
from pinecone import PodSpec
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone
import openai
import spacy
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import random
from googletrans import Translator
from translate import Translator

myGoat = True
openai.api_key = "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"   #4 Key


import os



def songInfo():
    if myGoat == True:
  # Get the current script directory
      script_dir = os.path.dirname(os.path.realpath(__file__))

      # Set the path to the folder
      folder_path = os.path.join(script_dir, 'songs')

      # Check if the directory exists
      if os.path.exists(folder_path):
          # Get the list of files in the folder
          files_in_folder = os.listdir(folder_path)

          # Filter out only MP3 files
          mp3_files = [file for file in files_in_folder if file.lower().endswith('.mp3')]

          # Check if there is exactly one MP3 file
          if len(mp3_files) == 1:
              ename = mp3_files[0]
              print(f'The single MP3 file in the folder is: {ename}')
          else:
              print('Either there is no MP3 file or there are multiple MP3 files in the folder.')
              ename = None
      else:
          print(f'The directory {folder_path} does not exist.')
          ename = None
      return ename
    else:
      ename ='marc.mp3'

name = songInfo()

def audiotoText(path):
  audio_file = open(path, "rb")
  transcript = openai.Audio.transcribe(
    model="whisper-1",
    file=audio_file,
    response_format="text"
  )
  text = transcript
  return text

def translate_text(text, from_lang, to_lang):
    from_lang = 'es'
    to_lang = 'en'
    translator = Translator(to_lang=to_lang, from_lang=from_lang)

    # Split the text into chunks of 500 characters
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    translated_chunks = []
    for chunk in chunks:
        translated_chunk = translator.translate(chunk)
        translated_chunks.append(translated_chunk)

    translated_text = ' '.join(translated_chunks)
    return translated_text



#Writing functions



def get_normalized_complexity(text):
    # Function to get normalized complexity rating
    prompts = f"Rate the complexity of the following text from a scale of 0-1 with 0 being an easy statement to understand and 1 being a complex phrase:\n{text}\nComplexity: "
    # Use OpenAI's GPT-4 to rate the complexity of the text
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompts}],
    )
    complexity_rating = float(response["choices"][0]["message"]["content"].strip())
    return complexity_rating

def getWritingDict(lyrics):
    lyrics_eng = translate_text(lyrics, from_lang = 'es', to_lang = 'en')
    finalVar = []

    for j in range(1, 6):
        # Generate English phrase
        additional_prompt = f"Create an English phrase (5-10 words) based on the following lyrics:\n{lyrics_eng}\nPhrase {j}: "
        additional_response = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=0.9,
            top_p=0.9,
            messages=[{"role": "user", "content": additional_prompt}]
        )
        additional_generated_phrase = additional_response["choices"][0]["message"]["content"].strip()

        # Translate the English phrase to Spanish
        additional_question = {
            "input": f'Translate the following English phrase to Spanish: "{additional_generated_phrase}"',
            "output": ""
        }
        additional_response_spanish = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=0.9,
            top_p=0.9,
            messages=[{"role": "user", "content": additional_question["input"]}]
        )
        additional_answer = additional_response_spanish["choices"][0]["message"]["content"].strip()
        additional_question["output"] = additional_answer

        # Get normalized complexity rating for the English phrase
        normalized_complexity = get_normalized_complexity(additional_generated_phrase)

        phrase = {
            "English": additional_generated_phrase,
            "Spanish": additional_answer,
            "Complexity": normalized_complexity
        }

        # Add elements to finalVar
        finalVar.append(phrase)
        
    return finalVar

def checkAnswerWriting(i, userInput, finalVar):
  i = i #get from the user
  corrAnswer = finalVar[i]['English']
  userInput = userInput
  comparisons = "Here is the correct answer" + corrAnswer +  "Here is the given answer" + userInput + "Is the answer correct (does the given answer convey the same thing as the correct answer). Make sure to include either the word 'Yes' or 'No only once' in your response to indicate whether the answer is correct"
  response_combined = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0.9,
        top_p=0.9,
        messages= [{ "role": "user", "content": comparisons}] )
  finalAnswer = response_combined.choices[0].message.content.strip()
  if 'yes' in finalAnswer.lower():
    #return true to alogorithm keeping track of score
    checkerSpeaker = True
  else:
    #return false to alogorithm keeping track of score
    checkerSpeaker = False
  return checkerSpeaker

def hintWriting(finalVar, i):
  # Your two strings
  realAnswer= finalVar[i]['English']

  # Construct the prompt
  prompt1 = f"Here is the real answer: {realAnswer} Create a hint without revealing the answer:"

  # Generate hints using OpenAI GPT-3
  response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0.9,
        top_p=0.9,
        messages= [{ "role": "user", "content": prompt1}] )

  hint1 = response["choices"][0]["message"]["content"].strip()

  prompt2 = f"Here is the real answer: {realAnswer}. Here is the previous hint: {hint1} Create another more helpful hint without revealing the answer:"

  # Generate hints using OpenAI GPT-3
  response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0.9,
        top_p=0.9,
        messages= [{ "role": "user", "content": prompt2}] )

  hint2 = response["choices"][0]["message"]["content"].strip()

  allHintsW = [hint1, hint2]

  return allHintsW

def questionNumber(i):
    qNumber = "Question " + str(i)
    return qNumber



#Vocabulary



def doVocab(lyrics):
  # Split the lyrics string into a list of words
  word_list = lyrics.split()

  # Initialize an empty list to store questions
  questions_list = []

  # Iterate 5 times to generate 5 questions
  for _ in range(5):
      # Generate random indices for word selection
      rn1 = random.randint(0, len(word_list))
      rn2 = rn1 - 12
      rn3 = rn1 + 12

      # Create a sublist using the random indices
      selected_words = word_list[rn2:rn1] + word_list[rn1:rn3]
      selected_words = [word.lower().replace(',', '').replace(' ', '') for word in selected_words]

      # Find the two longest words in the selected sublist
      longest_words = sorted(selected_words, key=len, reverse=True)[:1]

      # Replace the two longest words with "___" in the selected sublist
      for i, word in enumerate(selected_words):
          if word in longest_words:
              selected_words[i] = "___"

      # Store the two longest words in the answers list
      answers = longest_words[0]


      # Concatenate the words in the selected sublist into a single text variable
      question_text = ' '.join(selected_words)

      # Store the question and answers in a dictionary
      question_dict = {"question": question_text, "answers": answers}

      # Append the dictionary to the questions_list
      questions_list.append(question_dict)

  return questions_list

def checkAnswerVocab(i,userList, valueVocab):
  i = i #Question number you want to ask
  valueVocab = valueVocab[i]['answers'] #return this to the user

  userAnswer = userList #Get both words from the user and store as a list and make it userList
  print(userAnswer, valueVocab)
  bools = userAnswer.lower().strip() == valueVocab.lower().strip()
  if bools:
    #return true to alogorithm keeping track of score AND return correct answer
    checkerSpeaker = True
  else:
    #return false to alogorithm keeping track of score AND return correct answer
    checkerSpeaker = False

  return checkerSpeaker

def hintVocab(finalVar, i):
  # Your two strings
  realAnswer = finalVar[i]['answers']

  # Construct the prompt
  prompt1 = f"Here is the real answer: {realAnswer}. Create a hint without revealing the answer:"

  # Generate hints using OpenAI GPT-3
  response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0.9,
        top_p=0.9,
        messages= [{ "role": "user", "content": prompt1}] )

  hint1 = response["choices"][0]["message"]["content"].strip()

  return hint1


import os

print(f"Current Working Directory: {os.getcwd()}")
print("lsdlksdfjlks")

name = os.path.join("songs", name)

lyrics = audiotoText(name)
print(lyrics)

questions = []
answers = []
complexityWriting = []

writingQ = getWritingDict(lyrics)

for i in range(5):
  questions.append(writingQ[i]['English'])
  answers.append(writingQ[i]['Spanish'])
  complexityWriting.append(writingQ[i]['Complexity'])

vocabQ = doVocab(lyrics)

for i in range(5):
  questions.append(vocabQ[i]['question'])
  answers.append(vocabQ[i]['answers'])

hints = []

for i in range(5):
  writingH = hintWriting(writingQ, i)
  hints.append(writingH)

for i in range(5):
  vocabH = hintVocab(vocabQ, i)
  hints.append(vocabH)




#VectorDB

import PyPDF2

def vectorDB(path, query, api_key, context):
  def getData(path, n):
    def extract_text_from_pdf(pdf_path):
        text = ""
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        return text

    # Replace "your_pdf_file.pdf" with the actual file name you uploaded
    pdf_path = path
    text = extract_text_from_pdf(pdf_path)
    #print(text)

    def split_text_into_parts(text, num_parts=10):
        # Calculate the length of each part
        part_length = len(text) // num_parts
        # Split the text into parts
        parts = [text[i * part_length:(i + 1) * part_length] for i in range(num_parts)]
        return parts

    # Example usage:
    # Assuming you have already extracted the text and stored it in the 'text' variable
    text_parts = split_text_into_parts(text, num_parts=n)
    return text_parts

  my_list = getData(path, 20)
  my_list


  def getIndex(api_key, index):
    pc = Pinecone(api_key)
    index = pc.Index(index)
    return index

  index = getIndex(api_key, 'quickstart')

  def upserts(q, values, index):
    index = index
    my_list = values

    query = q
    MODEL = "text-embedding-3-small"

    res = openai.Embedding.create(
        input=[query], engine=MODEL
    )

    embeds = [record['embedding'] for record in res['data']]

    # load the first 1K rows of the TREC dataset
    #trec = load_dataset('trec', split='train[:1000]')

    batch_size = 32  # process everything in batches of 32
    for i in tqdm(range(0, len(my_list), batch_size)):
        # set end position of batch
        i_end = min(i+batch_size, len(my_list))
        # get batch of lines and IDs
        lines_batch = my_list[i: i+batch_size]
        ids_batch = [str(n) for n in range(i, i_end)]
        # create embeddings
        res = openai.Embedding.create(input=lines_batch, engine=MODEL)
        embeds = [record['embedding'] for record in res['data']]
        # prep metadata and upsert batch
        meta = [{'text': line} for line in lines_batch]
        to_upsert = zip(ids_batch, embeds, meta)
        # upsert to Pinecone
        index.upsert(vectors=list(to_upsert))

  upserts(query, my_list, index)

  def getRes(query):
    query = query
    MODEL = "text-embedding-3-small"

    xq = openai.Embedding.create(input=query, engine=MODEL)['data'][0]['embedding']

    res = index.query(vector = [xq], top_k=5, include_metadata=True)

    return res

  def vectorQuotes(query):
    similarity = getRes(query)
    #justQuotes just uses what the query results from Pinecone itself
    justQuotes = []
    for i in range(len(similarity['matches'])):
      justQuotes.append(similarity['matches'][i]['metadata']['text'])
    return justQuotes

  def getFinalSummaryGPT4(my_list, queryContext):
    my_list = my_list
    queryContext = queryContext

    # Function to split a list into equal sublists
    def split_list(lst, num_sublists):
        avg = len(lst) // num_sublists
        remainder = len(lst) % num_sublists
        return [lst[i * avg + min(i, remainder):(i + 1) * avg + min(i + 1, remainder)] for i in range(num_sublists)]

    # Split 'my_list' into n equal sublists
    n = 5
    sublists = split_list(my_list, n)

    # Generate summaries for each sublist using the OpenAI API
    sublist_summaries = []

    for i, sublist in enumerate(sublists):
        sublist_text = ' '.join(sublist)
        response = openai.ChatCompletion.create(
          model="gpt-4",
          temperature=0.9,
          top_p=0.9,
          messages= [{ "role": "user", "content": queryContext+sublist_text }] )

        # Extract the summary from the API response
        summary = response.choices[0].message.content
        sublist_summaries.append(summary)

    # Combine the 10 summaries into one variable
    combined_summary = ' '.join(sublist_summaries)

    # Add a specific prompt tailored to your data
    specific_prompt = f"Given the following summaries:\n{combined_summary}\n\nGenerate a coherent final summary that captures the essence of the provided information."

    specific_prompt = queryContext + specific_prompt
    # Use OpenAI API to generate the final coherent summary

    response_combined = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0.9,
        top_p=0.9,
        messages= [{ "role": "user", "content": specific_prompt}] )

    # Extract the final summary from the API response
    final_summary = response_combined.choices[0].message.content.strip()

    return final_summary


  def translate_to_spanish(text):
      translator = Translator()
      translated = translator.translate(text, src='en', dest='es')
      return translated.text

  query = query

  justQuotes = vectorQuotes(query)

  queryContext = context+query

  responseQuotes = getFinalSummaryGPT4(justQuotes, queryContext)

  return responseQuotes


openai.api_key = "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"   #4 Key
api_key = "d403ddc4-dc54-47d5-9c8f-ed19848d06ce"
path = "Full.pdf" 

topic = 'Present tense and preterite' #get from user textbox

'''
context =  "I want to improve my spanish so I want to do practice probelms. Give me JUST the practice problems. Keep the instructions in English though. Here is the topic"
answer1 = vectorDB(path, topic, api_key, context)


context = 'Create a lesson from the pdf on this topic that includes concepts in Spanish and mock Spanish questions. The lesson should be in English but there should be chunks of Spanish examples of these topics. Make sure to explain all the endings and give practice problems on this lesson'
answer4 = vectorDB(path, topic, api_key, context)
'''







#ML stuff



from app import storedPeople

empty = []
for i in range(len(storedPeople)):
  mini = []
  name = storedPeople[i]['email']
  mini.append(name)
  songsM = storedPeople[i]['total']
  mini.append(songsM)
  questionsC = storedPeople[i]['historicalData'][1][0]
  mini.append(questionsC)
  avgComp = storedPeople[i]['avgComplexity']
  mini.append(avgComp)
  empty.append(mini)


participants = np.array(empty)
og_participants = np.copy(participants)

'''
participants = np.array([
      [15, 25, 4],  # songs mastered, questions correct, average complexity for Player1
      [20, 30, 3],  # songs mastered, questions correct, average complexity for Player2
      [18, 28, 5],  # songs mastered, questions correct, average complexity for Player3
      # Add more participants as needed
])
'''


import numpy as np


def leaderBoard(participants):
    # Extract the scores from the participants
    scores = np.array([list(map(int, participant[1:])) for participant in participants])

    # Normalize the scores to represent probabilities
    probabilities = scores / scores.sum(axis=0)

    # Create the transition matrix for the Markov chain
    transition_matrix = probabilities.T

    # Set up the initial probability distribution (assuming equal probability for each participant)
    initial_distribution = np.ones(len(participants)) / len(participants)

    # Iterate to find the stationary distribution
    for _ in range(100):  # Perform 100 iterations (adjust as needed)
        initial_distribution = np.dot(initial_distribution, transition_matrix)

    # Find the participant with the highest PageRank score
    max_score_index = np.argmax(initial_distribution)
    max_score = initial_distribution[max_score_index]
    participant_with_highest_score = participants[max_score_index][0]

    return participant_with_highest_score, max_score

result = leaderBoard(participants)
displayStr = "The participant with the highest PageRank score is {result[0]} with a score of {result[1]}."



from sklearn.cluster import KMeans
import numpy as np

def cluster_participants(participants, num_clusters=2):
    # Extract numerical values for clustering
    data_for_clustering = np.array([participant[1:] for participant in participants])

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(data_for_clustering)

    return clusters


result = cluster_participants(participants)
participants_with_new_column = np.hstack((participants, result.reshape(-1, 1)))







