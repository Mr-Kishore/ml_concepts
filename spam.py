import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Function to collect email data from user
def collect_email_data():
    emails = []
    labels = []
    
    print("Enter email texts and label them as 'spam' or 'valid'. Type 'done' when finished.")
    
    while True:
        email = input("Enter email text (or type 'done' to finish): ")
        if email.lower() == 'done':
            break
        label = input("Is this email 'spam' or 'valid'? ").strip().lower()
        
        if label not in ['spam', 'valid']:
            print("Please enter a valid label ('spam' or 'valid').")
            continue
        
        emails.append(email)
        labels.append(label)
    
    return emails, labels

#  Collect email data
user_emails, user_labels = collect_email_data()

# Check if there are enough samples
if len(user_emails) < 2:  # At least 2 samples to split
    print("Not enough data to train the model. Please enter at least 2 emails.")
else:
    #  Load the dataset into a DataFrame
    df = pd.DataFrame({'text': user_emails, 'label': user_labels})

    #  Split the dataset into features and labels
    X = df['text']
    y = df['label']

    #  Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #  Text preprocessing and feature extraction
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    #  Train the model
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    #  Evaluate the model
    y_pred = model.predict(X_test_tfidf)
    print(classification_report(y_test, y_pred))

    #  Function to predict if an email is spam
    def predict_email(email):
        email_tfidf = vectorizer.transform([email])  
        prediction = model.predict(email_tfidf)
        return prediction[0]  

    # Main loop to ask for user input for prediction
    while True:
        user_input = input("Please enter an email string (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        result = predict_email(user_input)
        print(f'The email is: {result}')
