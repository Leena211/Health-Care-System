Health Care System
About the Project
This web application is a preliminary symptom checker that uses a machine learning model to provide users with a potential disease prediction, a severity rating, and a list of recommended precautions based on their symptoms. The goal of this project is to create a helpful, easy-to-use tool that empowers users with information about their health.

Disclaimer: This tool is for informational purposes only and is not a substitute for professional medical advice. Always consult a healthcare professional for diagnosis and treatment.

Features
Intuitive Symptom Input: Users can easily select up to four symptoms from a predefined list or type in a specific symptom for analysis.

Disease Prediction: The system leverages a pre-trained machine learning model to predict the most likely disease based on the combination of symptoms entered.

Severity Rating: Each analysis provides a severity level (Low, Medium, or High) to give users a quick understanding of their potential condition.

Recommended Precautions: The application suggests practical precautions and advice related to the predicted condition.

Visual Data Representation: Results include a dynamic pie chart that visualizes the probability of several possible diseases, making the analysis easy to understand.

User Authentication: The application includes a secure system for user registration and login, allowing users to save their analysis history.

Technologies Used
Backend: Django

Frontend: HTML, CSS (Tailwind CSS for styling)

Machine Learning: scikit-learn (used for the model)

Data Visualization: Chart.js

Database: Django's built-in database (or a similar solution)

How to Use
Select Symptoms: On the main page, choose at least two symptoms from the dropdown menus. You can also enter a symptom in the text box.

Analyze: Click the "Analyze Symptoms" button to submit your information to the model.

View Results: You will be redirected to a results page displaying the disease prediction, severity, precautions, and a pie chart of potential diseases.

