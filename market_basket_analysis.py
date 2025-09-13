import pandas as pd
import streamlit as st
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Market Basket Analysis", layout="wide")

def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=None)
            st.success("File successfully uploaded!")
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    return None

def get_transactions(df):
    transactions = []
    for i in range(df.shape[0]):
        transaction = [item for item in df.values[i, :] if str(item) != 'nan']
        transactions.append(transaction)
    return transactions

def run_analysis(transactions, min_support, min_confidence):
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)

    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return None

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return rules

def display_results(rules):
    st.markdown("### Association Rules")
    if rules is not None and not rules.empty:
        rules_display = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
        for _, row in rules_display.iterrows():
            ant = ', '.join(list(row['antecedents']))
            con = ', '.join(list(row['consequents']))
            st.markdown(f"- If a customer buys **[{ant}]**, they are likely to buy **[{con}]**")
            st.markdown(f"  - *Support: {row['support']:.2f}, Confidence: {row['confidence']:.2f}, Lift: {row['lift']:.2f}*")
    else:
        st.warning("No association rules found for the given parameters.")

def main():
    st.title("🛒 Market Basket Analysis Tool")
    st.markdown("Upload your transaction data as a CSV file and discover association rules.")

    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        min_support = st.slider("Minimum Support", 0.01, 1.0, 0.3, 0.01)
        min_confidence = st.slider("Minimum Confidence", 0.01, 1.0, 0.6, 0.01)

    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            st.markdown("### Transaction Data")
            st.dataframe(df)
            transactions = get_transactions(df)
            rules = run_analysis(transactions, min_support, min_confidence)
            display_results(rules)
    else:
        st.info("Awaiting for CSV file to be uploaded.")

if __name__ == '__main__':
    main()
