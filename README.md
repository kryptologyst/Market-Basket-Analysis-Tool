# Simple Market Basket Analysis Tool

This project is an interactive web application for performing market basket analysis. It uses the Apriori algorithm to identify frequent itemsets and generate association rules from transactional data.

## Features

- Upload transaction data in CSV format.
- Adjust `min_support` and `min_confidence` parameters through a user-friendly interface.
- View the uploaded data and the generated association rules in real-time.

## How to Run

1.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Streamlit application:**

    ```bash
    streamlit run market_basket_analysis.py
    ```

3.  **Use the application:**

    - Open your web browser and navigate to the local URL provided by Streamlit.
    - Use the sidebar to upload your CSV file and adjust the analysis parameters.
    - The results will be displayed automatically.
