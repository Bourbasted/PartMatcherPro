
import streamlit as st
import pandas as pd
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity

st.title("Parts Matcher Pro: Smart Column Version")

# Set API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

def get_embedding(text):
    try:
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Embedding failed: {e}")
        return None

def detect_text_columns(df):
    candidates = []
    for col in df.columns:
        if df[col].dtype == object:
            if any(keyword in col.lower() for keyword in ["desc", "detail", "text", "note"]):
                candidates.append(col)
    return candidates if candidates else df.select_dtypes(include="object").columns.tolist()

def get_column_selection(df, label, default_col=None):
    return st.selectbox(
        f"Select the description column for {label}",
        df.columns.tolist(),
        index=df.columns.get_loc(default_col) if default_col and default_col in df.columns else 0
    )

# File uploads
file1 = st.file_uploader("Upload First Spreadsheet", type=["xlsx", "csv"])
file2 = st.file_uploader("Upload Second Spreadsheet", type=["xlsx", "csv"])

threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.6, 0.05)
top_n_matches = st.number_input("Top N Matches per Item", min_value=1, max_value=10, value=3)

if file1 and file2:
    df1 = pd.read_excel(file1) if file1.name.endswith("xlsx") else pd.read_csv(file1)
    df2 = pd.read_excel(file2) if file2.name.endswith("xlsx") else pd.read_csv(file2)

    col1_candidates = detect_text_columns(df1)
    col2_candidates = detect_text_columns(df2)

    col1 = get_column_selection(df1, "File 1", col1_candidates[0] if col1_candidates else None)
    col2 = get_column_selection(df2, "File 2", col2_candidates[0] if col2_candidates else None)

    df1_clean = df1[[col1]].dropna().rename(columns={col1: "Description"})
    df2_clean = df2[[col2]].dropna().rename(columns={col2: "Description"})

    df1_clean["Embedding"] = df1_clean["Description"].apply(get_embedding)
    df2_clean["Embedding"] = df2_clean["Description"].apply(get_embedding)

    emb1 = np.array(df1_clean["Embedding"].tolist())
    emb2 = np.array(df2_clean["Embedding"].tolist())

    if emb1.ndim == 2 and emb2.ndim == 2:
        sim_matrix = cosine_similarity(emb1, emb2)

        results = []
        for i, row in enumerate(sim_matrix):
            top_indices = row.argsort()[-top_n_matches:][::-1]
            for j in top_indices:
                score = row[j]
                if score >= threshold:
                    desc1 = df1_clean.iloc[i]["Description"]
                    desc2 = df2_clean.iloc[j]["Description"]
                    shared = set(desc1.lower().split()).intersection(set(desc2.lower().split()))
                    results.append({
                        "File1_Description": desc1,
                        "File2_Description": desc2,
                        "Similarity": round(score, 3),
                        "Shared_Keywords": ", ".join(shared)
                    })

        df_results = pd.DataFrame(results)
        possible_bin_columns = ["BinLocation", "Location", "Location #", "Bin", "Storage"]
        possible_partnumber_columns = ["PartNumber", "Part #", "Part No", "Part_Number"]

        bin_col = next((col for col in df2.columns if col.strip().lower() in [x.lower() for x in possible_bin_columns]), None)
        pn_col = next((col for col in df2.columns if col.strip().lower() in [x.lower() for x in possible_partnumber_columns]), None)

        if bin_col and pn_col:
            try:
                bin_map = df2[[pn_col, bin_col]].dropna()
                bin_map.columns = ["PartNumber", "BinLocation"]

                # Dynamically find which part number column to use for matching
                merge_key = next((col for col in df_results.columns if "PartNumber" in col and "Catalogue" not in col), None)

                if merge_key:
                    df_results = df_results.merge(
                        bin_map,
                        left_on=merge_key,
                        right_on="PartNumber",
                        how="left"
                    )
                    if "PartNumber" in df_results.columns:
                        df_results = df_results.drop(columns=["PartNumber"])
                else:
                    st.warning("Could not find appropriate part number column for bin location merge.")
            except Exception as e:
                st.warning(f"Bin location merge skipped due to error: {e}")
        
        st.warning(f"Bin location merge skipped due to error: {e}")
        st.success(f"{len(df_results)} matches found.")
        st.dataframe(df_results.head(50))

        csv = df_results.to_csv(index=False)
        st.download_button("Download CSV", csv, "smart_matches.csv", "text/csv")
    else:
        st.error("Embedding failed for one or both files. Please check your API key and input format.")
