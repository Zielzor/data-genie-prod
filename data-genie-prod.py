import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns 
from collections import defaultdict

# Function to get or create session state
def get_session_state():
    session_state = st.session_state
    if "kmeans_page" not in session_state:
        session_state.kmeans_page = False
    if "manipulation_page" not in session_state:
        session_state.manipulation_page = False
    if "data_exploration" not in session_state:
        session_state.data_exploration = False
    if "random_sample_export" not in session_state:
        session_state.random_sample_export = False
    return session_state

def main():
    st.title("Data Genie")

    # Check if the "Go to K-Means" button has been clicked
    session_state = get_session_state()

    # Create the "Data Exploration" button
    if st.sidebar.button("Data Exploration", key="exploration"):
        session_state.data_exploration = True
        session_state.manipulation_page = False
        session_state.random_sample_export = False

    # Create the "Data Manipulation" button
    if st.sidebar.button("Data Manipulation", key="manipulation"):
        session_state.manipulation_page = True
        session_state.data_exploration = False
        session_state.random_sample_export = False

    # Create "Random Sample Export" button
    if st.sidebar.button("Random Sample Export", key="random_sample"):
        session_state.random_sample_export = True
        session_state.data_exploration = False
        session_state.manipulation_page = False

    # Render the appropriate page based on the button click
    if session_state.manipulation_page:
        render_manipulation_page()
    elif session_state.data_exploration:
        render_data_exploration()
    elif session_state.random_sample_export:
        render_random_sample_export()

def render_data_exploration():
    st.subheader("Data Exploration")
    st.write("This section allows for data exploration")

    # File upload
    uploaded_files = st.file_uploader("Choose files", type=["csv", "xlsx", "txt"], accept_multiple_files=True)

    if uploaded_files:
        # Create a dictionary to store data from all uploaded files
        data = defaultdict(pd.DataFrame)

        for uploaded_file in uploaded_files:
            file_extension = uploaded_file.name.split(".")[-1]

            # Read file
            if file_extension == "csv":
                df = pd.read_csv(uploaded_file)
            elif file_extension == "xlsx":
                df = pd.read_excel(uploaded_file)
            elif file_extension == "txt":
                df = pd.read_csv(uploaded_file, delimiter="\t")
            else:
                st.error("Invalid file format. Please upload a CSV, XLSX, or TXT file.")
                return

            # Store the data in the dictionary
            data[uploaded_file.name] = df

        # Select a file to operate on
        selected_file = st.selectbox("Select a file to operate on", list(data.keys()))

        # Get the selected dataframe
        df = data[selected_file]
        
        # Data info
        st.subheader("Dataset Info")
        st.write(df.dtypes)

        # Display the data
        st.subheader("Data Preview")
        st.write(df.head())

        #Displaying Desc and Info side by sid
        # Descriptive statistics
        st.subheader("Descriptive Statistics")
        st.write(df.describe())
        
        
        #Correlactions heatmap
        st.subheader("Detailed Correlations")
        correlation_view = st.radio("Correlation View",("Table", "Heatmap"))

        # Add a checkbox to select all columns
        if st.checkbox('Select all columns'):
            cols = df.columns.tolist()
        else:
        # Add a multiselect widget to allow the user to select columns
            cols = st.multiselect("Select the columns to compute correlations", df.columns.tolist())

        # Only compute correlations for selected columns
        df_selected = df[cols]

        # Compute correlations
        correlations = df_selected.corr()

        # Add an input to select the threshold
        threshold = st.number_input('Correlation threshold', min_value=0.0, max_value=1.0, value=0.5, step=0.01)

        # Add a selectbox to choose the filter type
        filter_type = st.selectbox('Filter type', ['Greater than', 'Less than'])


        # Apply the threshold
        correlations = correlations.where(np.triu(np.ones(correlations.shape)).astype(np.bool))
        if filter_type == 'Greater than':
            correlations = correlations[correlations >= threshold]
        else:
            correlations = correlations[correlations <= threshold]

        if correlation_view == "Table":
            st.write(correlations)
        elif correlation_view == "Heatmap":
            fig,ax = plt.subplots(figsize=(10,8)) #adjust the size
            cmap = sns.diverging_palette(220,20, as_cmap=True) #Color palette
            sns.heatmap(correlations, annot=False,cmap=cmap, linewidths=0.5, annot_kws={"fontsize":10},ax=ax)
            st.pyplot(fig)

        # Display unique values for selected columns
        st.subheader("Unique Values")
        unique_cols = st.multiselect("Select columns for unique values", df.columns)
        for col in unique_cols:
            st.subheader(f"Unique values for column: {col}")
            st.write(df[col].unique())

        # Display values of selected columns as histograms
        st.subheader("Histogram")
        hist_cols = st.multiselect("Select columns for histogram", df.columns)
        for col in hist_cols:
            st.subheader(f"Histogram for column: {col}")
            plt.hist(df[col])
            st.pyplot()

def render_manipulation_page():
    st.subheader("Data Manipulation")
    st.write("This section allows for data manipulation  and export")

    # File upload
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "txt"]) 

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1]

        # Read file
        if file_extension == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_extension == "xlsx":
            df = pd.read_excel(uploaded_file)
        elif file_extension == "txt":
            df = pd.read_csv(uploaded_file, delimiter="\t")
        else:
            st.error("Invalid file format. Please upload a CSV, XLSX, or TXT file.")
            return
    
        # Filter the dataset
        st.subheader("Filter Dataset")
        columns = st.multiselect("Columns: ", df.columns)
        filter = st.radio("Chose by:", ("include","exclude"))

        if filter == "exclude":
            columns = [col for col in df.columns if col not in columns]
        
        filtered_df = df[columns]
        filtered_df

        
        # Export filtered/transformed dataset
        st.sidebar.subheader("Export Filtered/Transformed Dataset")
        export_format = st.sidebar.selectbox("Select export format", ["csv", "xlsx", "txt"],key="export-format")
        export_filename = st.sidebar.text_input("Enter export file name", key="export-name")

        if st.sidebar.button("Export",key="export_filtered"):
            if export_format == "csv":
                filtered_df.to_csv(export_filename + ".csv", index=False)
            elif export_format == "xlsx":
                filtered_df.to_excel(export_filename + ".xlsx", index=False)
            elif export_format == "txt":
                filtered_df.to_csv(export_filename + ".txt", sep="\t", index=False)
            st.sidebar.success("Export successful!")
        
        # Pivot table
        st.subheader("Pivot Table")
        pivot_cols = st.multiselect("Select columns for pivot table", df.columns)
        pivot_values = st.selectbox("Select values for pivot table",df.columns)
        pivot_agg = st.selectbox("Select aggregation function", ["sum","count"])
        
        if pivot_cols and pivot_values and pivot_agg:
            pivot_table = df.pivot_table(index=pivot_cols, values=pivot_values, aggfunc=pivot_agg)
            st.write(pivot_table)

def render_random_sample_export():
    st.write("Test")
    # File upload
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "txt"]) 

    # Check if a file is uploaded
    if uploaded_file is None:
        st.warning("Please upload a file first.")
        return

    file_extension = uploaded_file.name.split(".")[-1]

    # Read file
    if file_extension == "csv":
        df = pd.read_csv(uploaded_file)
    elif file_extension == "xlsx":
        df = pd.read_excel(uploaded_file)
    elif file_extension == "txt":
        df = pd.read_csv(uploaded_file, delimiter="\t")
    else:
        st.error("Invalid file format. Please upload a CSV, XLSX, or TXT file.")
        return

    # Display the random sample of uploaded file
    st.subheader("Random sample")
    st.write("Create and export a random sample of chosen size")
    rows = df.shape[0]
    columns = df.shape[1]
    st.write(f"Total size of rows: {rows}")
    st.write(f"Total size of columns: {columns}")

    sample_size = st.number_input("Enter sample size:", value=100, step=1)
    sample_size_txt = str(sample_size)
    random_sample = df.sample(sample_size)
    st.write(f"Shape of exported data: {random_sample.shape}")
    st.write(random_sample)
    
    # Exporting Random sample
    export_format = st.sidebar.selectbox("Select Export Format", ["csv", "xlsx", "txt"])
    autogenerated_sample_name = f"Random sample from {uploaded_file.name} of size {sample_size_txt}"
    export_filename = st.sidebar.text_input("Enter Filename", value=autogenerated_sample_name)
    

    if st.sidebar.button("Export Sample"):
        if export_format == "csv":
            random_sample.to_csv(export_filename + ".csv", index=False)
        elif export_format == "xlsx":
            random_sample.to_excel(export_filename + ".xlsx", index=False)
        elif export_format == "txt":
            random_sample.to_csv(export_filename + ".txt", sep="\t", index=False)
        st.sidebar.success("Export successful!")

    st.write("Test")
    # File upload
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "txt"]) 

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1]
        st.warning("Please upload a file first.")
    

        # Read file
        if file_extension == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_extension == "xlsx":
            df = pd.read_excel(uploaded_file)
        elif file_extension == "txt":
            df = pd.read_csv(uploaded_file, delimiter="\t")
        else:
            st.error("Invalid file format. Please upload a CSV, XLSX, or TXT file.")
            return
    
    # Display the random sample of uplaoded file
    st.subheader("Random sample")
    st.write("Create and export a random sample of chosen size")
    rows = df.shape[0]
    columns = df.shape[1]
    st.write(f"Total size of rows: {rows}")
    st.write(f"Total size of columns: {columns}")

    sample_size = st.number_input("Enter sample size:",value=100, step=1)
    sample_size_txt = str(sample_size)
    random_sample=df.sample(sample_size)
    st.write(f"Shape of exported data{random_sample.shape}")
    st.write(random_sample)
    
    # Exporting Random sample
    export_format = st.sidebar.selectbox("Select Export Format", ["csv", "xlsx", "txt"])
    autogenerated_sample_name = f"Random sample from {uploaded_file.name} of size {sample_size_txt}"
    export_filename = st.sidebar.text_input("Enter Filename", value=autogenerated_sample_name)
    

    if st.sidebar.button("Export Sample"):
        if export_format == "csv":
            df.to_csv(export_filename + ".csv", index=False)
        elif export_format == "xlsx":
            df.to_excel(export_filename + ".xlsx", index=False)
        elif export_format == "txt":
            df.to_csv(export_filename + ".txt", sep="\t", index=False)
        st.success("Export successful!")
    
    


# Run the app
if __name__ == "__main__":
    main()
