import streamlit as st
import pandas as pd
import plotly.express as px

# Load dataset
def load_data():
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    else:
        st.warning("Please upload a CSV file.")
        return None

data = load_data()

if data is not None:
    # Streamlit app title
    st.title("Flowshop Scheduling Visualization")

    # Sidebar for filtering
    st.sidebar.header("Filters")
    selected_priority = st.sidebar.multiselect(
        "Select Priority Levels", data['Priority'].unique(), default=data['Priority'].unique()
    )
    selected_family = st.sidebar.multiselect(
        "Select Family IDs", data['Family_ID'].unique(), default=data['Family_ID'].unique()
    )

    # Filter data based on selections
    filtered_data = data[
        (data['Priority'].isin(selected_priority)) & (data['Family_ID'].isin(selected_family))
    ]

    # Display filtered data
    st.subheader("Filtered Dataset")
    st.dataframe(filtered_data)

    # Statistics
    st.subheader("General Statistics")
    st.write(f"Total Jobs: {len(filtered_data)}")
    st.write(f"Average Processing Time (Machine 1): {filtered_data['Processing_Time_Machine_1'].mean():.2f}")
    st.write(f"Average Processing Time (Machine 2): {filtered_data['Processing_Time_Machine_2'].mean():.2f}")
    st.write(f"Average Setup Time (Machine 1): {filtered_data['Setup_Time_Machine_1'].mean():.2f}")
    st.write(f"Average Setup Time (Machine 2): {filtered_data['Setup_Time_Machine_2'].mean():.2f}")

    # Visualization: Priority Distribution
    st.subheader("Priority Distribution")
    priority_counts = filtered_data['Priority'].value_counts().reset_index()
    priority_counts.columns = ['Priority', 'Count']
    fig_priority = px.bar(priority_counts, x='Priority', y='Count', title="Priority Distribution", color='Priority')
    st.plotly_chart(fig_priority)

    # Visualization: Processing and Setup Time
    st.subheader("Processing and Setup Time Distribution")
    fig_processing = px.scatter(
        filtered_data, 
        x='Processing_Time_Machine_1', 
        y='Processing_Time_Machine_2', 
        size='Weight', 
        color='Priority', 
        hover_name='Job_ID', 
        title="Processing Time Comparison (Machine 1 vs Machine 2)"
    )
    st.plotly_chart(fig_processing)

    fig_setup = px.scatter(
        filtered_data, 
        x='Setup_Time_Machine_1', 
        y='Setup_Time_Machine_2', 
        size='Weight', 
        color='Priority', 
        hover_name='Job_ID', 
        title="Setup Time Comparison (Machine 1 vs Machine 2)"
    )
    st.plotly_chart(fig_setup)

    # Visualization: Gantt Chart for Job Scheduling
    st.subheader("Gantt Chart for Job Scheduling")
    gantt_data = filtered_data[['Job_ID', 'Due_Date', 'Processing_Time_Machine_1', 'Processing_Time_Machine_2']]
    gantt_data = gantt_data.melt(id_vars=['Job_ID', 'Due_Date'], 
                                 value_vars=['Processing_Time_Machine_1', 'Processing_Time_Machine_2'],
                                 var_name='Machine', value_name='Processing_Time')
    gantt_data['End_Time'] = gantt_data['Due_Date'] - gantt_data['Processing_Time']
    fig_gantt = px.timeline(
        gantt_data, 
        x_start='End_Time', 
        x_end='Due_Date', 
        y='Job_ID', 
        color='Machine',
        title="Job Scheduling Gantt Chart"
    )
    st.plotly_chart(fig_gantt)

    st.write("\n\n")
    st.info("This visualization uses Ant Colony Optimization results for improved scheduling.")
