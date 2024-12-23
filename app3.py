import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import classification_report

def call_llm_api(model, input_data):
    """Simulate API call for local testing (replace with real model training)."""
    st.write(f"Simulating API call for {model}...")

    # Dictionary to map model names to their respective responses
    model_responses = {
        "gemma2-9b-it": {"cleaned_data": input_data["dataset"]},
        "gemma-7b-it": {"model": "mock_initial_model", "accuracy": 0.85},
        "llama-3.3-70b-versatile": {"tuned_model": "mock_tuned_model", "accuracy": 0.90}
    }

    # Get the response for the given model
    if model in model_responses:
        return model_responses[model]
    else:
        raise ValueError("Unknown model")

# Agent 1: Preprocessing
def preprocess_dataset(dataset):
    """Preprocess the dataset using Gemma2-9B-IT."""
    st.write("Agent 1: Preprocessing dataset with Gemma2-9B-IT...")

    # Replace invalid values (inf, -inf) and fill NaN with 0
    dataset = dataset.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    def label_encode_columns(dataset, categorical_cols):
        """Label encode the categorical columns in the dataset."""
        for col in categorical_cols:
            label_encoder = LabelEncoder()
            dataset[col] = label_encoder.fit_transform(dataset[col])
        return dataset
    
    # Use label encoding for categorical data
    categorical_cols = dataset.select_dtypes(include=['object']).columns
    dataset = label_encode_columns(dataset, categorical_cols)

    # Prepare input data for API
    input_data = {"dataset": dataset.to_dict(orient="records")}

    # Simulate API call for preprocessing
    result = call_llm_api("gemma2-9b-it", input_data)

    # Convert the cleaned data back to a DataFrame
    cleaned_data = pd.DataFrame(result["cleaned_data"])
    return cleaned_data

# Agent 2: Model Creation
def create_initial_model(cleaned_dataset, target):
    """Create the initial model using Gemma-7B-IT."""
    st.write("Agent 2: Creating initial model with Gemma-7B-IT...")

    X = cleaned_dataset.drop(columns=[target])
    y = cleaned_dataset[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train, y_train)

    # Evaluate the model
    accuracy = classifier.score(X_test, y_test)  # Get accuracy

    st.write(f"Initial Model Accuracy: {accuracy * 100:.2f}%")
    return classifier, accuracy

# Agent 3: Fine-Tuning (mock, can be extended with actual fine-tuning)
def fine_tune_model(initial_model, cleaned_dataset, target):
    """Fine-tune the model using Llama-3.3-70B-Versatile."""
    st.write("Agent 3: Fine-tuning model with Llama-3.3-70B-Versatile...")

    # You could add fine-tuning steps here, but we will use the same model for simplicity.
    tuned_model = initial_model  # No actual fine-tuning is done here for simplicity

    # Get final accuracy (it should be different each time based on the data)
    accuracy = tuned_model.score(cleaned_dataset.drop(columns=[target]), cleaned_dataset[target])
    st.write(f"Tuned Model Accuracy: {accuracy * 100:.2f}%")

    return tuned_model
def preprocess_user_input(input_data, dataset):
    """Preprocess the user input data to match the format required by the model."""
    try:
        # Split input data into a list
        input_list = input_data.split(',')

        # Check if input matches the expected feature count
        expected_features = len(dataset.columns)
        provided_features = len(input_list)

        if provided_features != expected_features:
            st.error(
                f"Error: Input data has {provided_features} values, "
                f"but {expected_features} values are required."
            )
            return None

        # Identify categorical columns
        categorical_cols = dataset.select_dtypes(include=['object']).columns

        # Label encode categorical values
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            le.fit(dataset[col])  # Fit on the dataset used during training
            label_encoders[col] = le

        # Encode or parse each input value
        encoded_input = []
        for value, col in zip(input_list, dataset.columns):
            if col in categorical_cols:
                encoded_input.append(label_encoders[col].transform([value])[0])
            else:
                encoded_input.append(float(value))

        return np.array(encoded_input).reshape(1, -1)
    except Exception as e:
        st.error(f"Error processing input data: {e}")
        return None

# Leader Agent to orchestrate the process
def leader_agent(dataset, target):
    """Orchestrate the entire workflow with all agents."""
    st.write("Leader Agent: Orchestrating the workflow...")

    # Step 1: Preprocessing
    cleaned_data = preprocess_dataset(dataset)

    # Step 2: Model Creation
    initial_model, initial_accuracy = create_initial_model(cleaned_data, target)

    # Step 3: Fine-Tuning
    final_model = fine_tune_model(initial_model, cleaned_data, target)

    # Save the trained model to session state
    st.session_state['final_model'] = final_model
    st.session_state['dataset'] = cleaned_data  # Save the dataset for prediction

    # Show classification report
    X = cleaned_data.drop(columns=[target])
    y = cleaned_data[target]
    st.text("Model Training Completed. Classification Report:")
    st.text("Precision: The ratio of correctly predicted positive observations to the total predicted positives.")
    st.text("Recall: The ratio of correctly predicted positive observations to the all observations in actual class.")
    st.text("F1-Score: The weighted average of Precision and Recall.")
    st.text("Support: The number of actual occurrences of the class in the dataset.")
    st.text(classification_report(y, final_model.predict(X)))
    st.text(f"Model Training Completed. Classification Report:\n{classification_report(y, final_model.predict(X))}")

# Initialize page state
if 'page' not in st.session_state:
    st.session_state['page'] = 'main'

# Streamlit UI
st.title("Multi-Agent AI System with Specialized Models")

# File uploader for dataset
uploaded_file = st.file_uploader("Upload your Dataset (CSV)", type="csv")

# Check if file is uploaded and valid
if uploaded_file:
    try:
        # Check if the file is empty
        if uploaded_file.size == 0:
            st.error("The uploaded CSV file is empty. Please upload a valid dataset.")
            st.stop()

        try:
            dataset = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            st.error("There was an error decoding the file. Please ensure the file is encoded in UTF-8.")
            st.stop()
        except pd.errors.ParserError:
            st.error("There was an error parsing the file. Please ensure the file is a valid CSV.")
            st.stop()

        # Ensure the dataset is not empty and has columns
        if dataset.empty or dataset.columns.empty:
            st.error("The uploaded CSV file is empty or has no columns. Please upload a valid dataset.")
        else:
            st.write("Original Dataset Preview:", dataset.head())

            # Dropdown to select target column for classification or regression
            target_column = st.selectbox(
                "Select the Target Column (for Classification or Regression)",
                dataset.columns
            )

            # Button to execute the workflow
            if st.button("Train Model"):
                try:
                    # Run the multi-agent system
                    leader_agent(dataset, target_column)

                    st.write("Model training completed. Go to prediction page to use the model.")
                    st.session_state['page'] = 'prediction'

                except Exception as e:
                    st.error(f"An error occurred: {e}")
    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")
else:
    st.warning("Please upload a CSV file to get started.")

# Prediction Page
def prediction_page():
    st.title("Predict with Trained Model")
    
    if 'final_model' in st.session_state:
        final_model = st.session_state['final_model']
        
        # Input for prediction
        input_data = st.text_input("Enter Data for Prediction (comma-separated)")

        if input_data:
            try:
                # Preprocess the input data
                processed_input = preprocess_user_input(input_data, st.session_state.get('dataset', []))
                
                # Make the prediction
                if processed_input is not None:
                    prediction = final_model.predict(processed_input)
                    st.write(f"Prediction: {prediction}")
            except Exception as e:
                st.error(f"Invalid input format: {e}")
        
        else:
            st.warning("Please enter the data for prediction.")
        
        # Back button to go to the main page
        if st.button("Go Back"):
            st.session_state['page'] = 'main'
    else:
        st.error("No trained model found. Please train a model first.")

# Main page navigation
if st.session_state['page'] == 'main':
    # Main page content (train model)
    st.write("Train your model here.")
elif st.session_state['page'] == 'prediction':
    prediction_page()
