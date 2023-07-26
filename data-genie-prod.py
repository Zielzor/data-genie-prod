import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
from collections import defaultdict
import os 

# Function to get or create session state
def get_session_state():
    session_state = st.session_state
    if "manipulation_page" not in session_state:
        session_state.manipulation_page = False
    if "data_exploration" not in session_state:
        session_state.data_exploration = False
    if "random_sample_export" not in session_state:
        session_state.random_sample_export = False
    if "exported_files" not in session_state:
        session_state.exported_files = False
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
        session_state.exported_files = False

    # Create the "Data Manipulation" button
    if st.sidebar.button("Data Manipulation", key="manipulation"):
        session_state.manipulation_page = True
        session_state.data_exploration = False
        session_state.random_sample_export = False
        session_state.exported_files = False

    # Create "Random Sample Export" button
    if st.sidebar.button("Random Sample Export", key="random_sample"):
        session_state.random_sample_export = True
        session_state.data_exploration = False
        session_state.manipulation_page = False
        session_state.exported_files = False
    
    # Create the list of files
    if st.sidebar.button("Exported Files", key="files"):
        session_state.exported_files = True
        session_state.data_exploration = False
        session_state.manipulation_page = False
        session_state.random_sample_export = False

    # Render the appropriate page based on the button click
    if session_state.manipulation_page:
        render_manipulation_page()
    elif session_state.data_exploration:
        render_data_exploration()
    elif session_state.random_sample_export:
        render_random_sample_export()
    elif session_state.exported_files:
        render_display_files()

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
        filter = st.radio("Chose by:", ("include", "exclude"))

        if filter == "exclude":
            columns = [col for col in df.columns if col not in columns]
        
        filtered_df = df[columns]
        st.write(filtered_df)

        # Export filtered/transformed dataset
        st.sidebar.subheader("Export Filtered/Transformed Dataset")
        export_format = st.sidebar.selectbox("Select export format", ["csv", "xlsx", "txt"], key="export-format")
        export_filename = st.sidebar.text_input("Enter export file name", key="export-name")

        # FileDownloader to enable download and save to "export" directory
        if st.sidebar.button("Export", key="export_filtered"):
            export_directory = os.path.join(os.getcwd(), "export")
            if not os.path.exists(export_directory):
                os.makedirs(export_directory)

            if export_format == "csv":
                file_path = os.path.join(export_directory, export_filename + ".csv")
                filtered_df.to_csv(file_path, index=False)
            elif export_format == "xlsx":
                file_path = os.path.join(export_directory, export_filename + ".xlsx")
                filtered_df.to_excel(file_path, index=False)
            elif export_format == "txt":
                file_path = os.path.join(export_directory, export_filename + ".txt")
                filtered_df.to_csv(file_path, sep="\t", index=False)

            st.sidebar.success(f"Export successful! Navigate to Exported Files for download")
        
        # Pivot table
        st.subheader("Pivot Table")
        pivot_cols = st.multiselect("Select columns for pivot table", df.columns)
        pivot_values = st.selectbox("Select values for pivot table", df.columns)
        pivot_agg = st.selectbox("Select aggregation function", ["sum", "count"])
        
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

    # Export options
    st.write("Export Options")
    export_format = st.selectbox("Select Export Format", ["csv", "xlsx", "txt"])
    autogenerated_sample_name = f"Random_sample_from_{uploaded_file.name}_of_size_{len(df)}"
    export_filename = st.text_input("Enter Filename", value=autogenerated_sample_name)

    # FileDownloader to enable download and save to "export" directory
    if st.button("Export Sample"):
        export_directory = os.path.join(os.getcwd(), "export")
        if not os.path.exists(export_directory):
            os.makedirs(export_directory)

        if export_format == "csv":
            file_path = os.path.join(export_directory, export_filename + ".csv")
            df.to_csv(file_path, index=False)
        elif export_format == "xlsx":
            file_path = os.path.join(export_directory, export_filename + ".xlsx")
            df.to_excel(file_path, index=False)
        elif export_format == "txt":
            file_path = os.path.join(export_directory, export_filename + ".txt")
            df.to_csv(file_path, sep="\t", index=False)

        st.success(f"Export successful! Navigate to Exported Files button for download")
    
def render_display_files():
    # Get the absolute path to the "export" directory inside the app's working directory
    directory_path = os.path.join(os.getcwd(), "export")

    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        st.header("Files in the Directory")
        
        files = os.listdir(directory_path)
        if len(files) > 0:
            for file_name in files:
                # Create a checkbox to select the file for deletion
                delete_file = st.checkbox(f"Delete {file_name}", key=f"checkbox_{file_name}")
                if delete_file:
                    confirm_delete = st.checkbox("Are you sure?")
                    if confirm_delete:
                        # Full path of the file to delete
                        file_path = os.path.join(directory_path, file_name)
                        try:
                            os.remove(file_path)
                            st.success(f"{file_name} has been deleted.")
                        except Exception as e:
                            st.error(f"Error deleting {file_name}: {str(e)}")

                # Create a download link for each file
                file_path = os.path.join(directory_path, file_name)
                st.download_button(label="Download", data=file_path, file_name=file_name)
        else:
            st.write("No files found in the directory.")
    else:
        st.write("Directory does not exist or is not a valid directory.")
    
    



# Run the app
if __name__ == "__main__":
    main()