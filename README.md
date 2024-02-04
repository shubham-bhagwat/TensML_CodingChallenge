# Movie Recommendation App

Welcome to the Movie Recommendation App! This application uses collaborative filtering, specifically K-Nearest Neighbors (KNN), to provide personalized movie recommendations based on user preferences. It incorporates features such as genre information and timestamps for a richer understanding of user behavior.

## Getting Started

To run the Movie Recommendation App locally, follow these steps:

### Prerequisites

- Python 3.6 or later
- Install the required dependencies using:

    ```bash
      pip install -r requirements.txt

### Running the App

1. Clone the repository:
    ```bash
      git clone https://github.com/your-username/movie-recommendation-app.git

2. Navigate to the project directory:
    ```bash
      cd movie-recommendation-app

3. Run the Streamlit app:
    ```bash
      streamlit run app.py
      
### Features

- Data Overview Page: Provides a brief overview of the initial raw data, explaining the use of random sampling during data loading.

- Feature Enhancement Page: Displays transformations and new columns introduced, such as binary encoding for genres and timestamp conversion.

- Test Train Split Overview Page: Offers insights into the custom train-test split logic and mentions that the sizes of the training and testing sets may vary.

- Recommendation Abstract Page: Describes the rationale for choosing collaborative filtering (KNN) and summarizes conclusions from testing.

- Recommendation Demo Page: Allows users to input a user ID to receive personalized movie recommendations. Displays training data for the selected user and evaluates model predictions.

### Additional Notes
The application utilizes the Surprise library for collaborative filtering and Streamlit for the web interface.

Evaluation metrics include sum squared error, accuracy, precision, recall, and F1 score.

Future considerations include fine-tuning models and exploring alternative approaches for continuous improvement.

Feel free to explore and enjoy the movie recommendations!

### Author
Shubham Bhagwat

