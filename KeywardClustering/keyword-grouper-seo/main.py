import streamlit as st
import pandas as pd
from collections import Counter
import itertools
import re

st.set_page_config(page_title="Keyword Grouper SEO App", page_icon="🔑", layout="wide")

# Function to determine keyword groups based on term frequency
def group_keyword(df, stop_words, min_group_size=2, ngram_size=2, keyword_column='Parent Keyword'):
    df[keyword_column] = df[keyword_column].astype(str)
    all_words = list(itertools.chain(*df[keyword_column].str.lower().str.split()))
    word_freq = Counter(all_words)
    common_terms = {word for word, freq in word_freq.items() if freq > 1 and word not in stop_words and len(word) > 2}

    grouped_dfs = []
    for keyword in df[keyword_column]:
        words = re.findall(r'\b\w+\b', keyword.lower())
        if len(words) >= ngram_size:
            ngrams = [tuple(words[i:i + ngram_size]) for i in range(len(words) - ngram_size + 1)]
            groups = set()
            for ngram in ngrams:
                if all(term in common_terms or term.isdigit() for term in ngram):
                    groups.add(" ".join(ngram))
            if groups:
                grouped_dfs.extend([pd.DataFrame({'Group': [group], 'Keyword': [keyword]}) for group in groups])

    grouped_keyword_df = pd.concat(grouped_dfs, ignore_index=True)
    filtered_groups = grouped_keyword_df.groupby('Group').filter(lambda x: len(x) >= min_group_size)
    return filtered_groups

# Function to calculate the total clicks for each group
def calculate_click_totals(df, grouped_df, keyword_column='Parent Keyword', clicks_column='Volume'):
    click_totals = {}
    for group in grouped_df['Group'].unique():
        keyword_in_group = grouped_df[grouped_df['Group'] == group]['Keyword'].tolist()
        click_total = df[df[keyword_column].isin(keyword_in_group)][clicks_column].sum()
        click_totals[group] = click_total
    return click_totals

# Function to calculate additional metrics for each group
def calculate_group_metrics(df, grouped_df, keyword_column='Parent Keyword', clicks_column='Volume', difficulty_column='Difficulty', traffic_potential_column='Traffic potential'):
    metrics = {}
    for group in grouped_df['Group'].unique():
        keyword_in_group = grouped_df[grouped_df['Group'] == group]['Keyword'].tolist()
        total_volume = df[df[keyword_column].isin(keyword_in_group)][clicks_column].sum()
        avg_kd = df[df[keyword_column].isin(keyword_in_group)][difficulty_column].mean()
        traffic_potential = df[df[keyword_column].isin(keyword_in_group)][traffic_potential_column].sum()
        metrics[group] = {
            'Total Volume': total_volume,
            'Avg. KD': avg_kd,
            'Traffic Potential': traffic_potential
        }
    return metrics

# Function to count unique values and sum integer values for selected columns
def count_unique_and_sum(df):
    columns_for_unique_count = ['Parent Keyword', 'Keyword', 'SERP Features', 'Country']
    unique_counts = df[columns_for_unique_count].nunique()
    sum_counts = df.select_dtypes(include=[int, float]).sum()
    unique_parent_keyword = df['Parent Keyword'].unique()
    return unique_counts, sum_counts, unique_parent_keyword

# Function to read CSV file with proper encoding and delimiter
def read_csv_file(uploaded_file):
    encodings = ['utf-8', 'latin1', 'utf-16']
    for encoding in encodings:
        try:
            return pd.read_csv(uploaded_file, encoding=encoding)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    st.error("Error parsing the file. Please check the encoding and the file format.")
    return None

# Streamlit user interface
st.title("Keyword Grouper SEO App")
st.markdown("made in Streamlit 🎈 by [Growth Src](https://growthsrc.com/)")
st.divider()

# Default stop words in English
default_stop_words = [
    'and', 'but', 'is', 'the', 'to', 'in', 'for', 'on', 'with', 'as', 'by', 'at', 'from',
    'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 
    'should', 'now'
]

# Upload CSV file for keyword with clicks
st.subheader("⬆️ Upload Keyword CSV with Clicks")
uploaded_file = st.file_uploader("CSV with Keyword and Clicks", type=["csv"])

# Process uploaded file with keyword and clicks
if uploaded_file is not None:
    data = read_csv_file(uploaded_file)
    if data is not None:
        # Automatically map columns
        keyword_column = 'Parent Keyword'
        clicks_column = 'Volume'
        difficulty_column = 'Difficulty'
        traffic_potential_column = 'Traffic potential'
        cpc_column = 'cpc'
        global_volume_column = 'Global volume'

        # Process the data directly
        with st.spinner("Grouping..."):
            grouped_keyword_df = group_keyword(data, default_stop_words)
            click_totals = calculate_click_totals(data, grouped_keyword_df)
            metrics = calculate_group_metrics(data, grouped_keyword_df)

            # Count unique clusters and sum integer values for all columns
            unique_counts, sum_counts, unique_parent_keyword = count_unique_and_sum(data)

            sorted_groups = sorted(click_totals.items(), key=lambda x: x[1], reverse=True)
            top_groups = sorted_groups[:20]  # Display top 20 groups

            top_groups_df = pd.DataFrame(top_groups, columns=['Cluster', 'Total Volume'])
            top_groups_df['Avg. KD'] = top_groups_df['Cluster'].map(lambda x: metrics[x]['Avg. KD'])
            top_groups_df['Traffic Potential'] = top_groups_df['Cluster'].map(lambda x: metrics[x]['Traffic Potential'])

            # Ensure numerical columns are of numeric type
            top_groups_df['Total Volume'] = pd.to_numeric(top_groups_df['Total Volume'], errors='coerce')
            top_groups_df['Avg. KD'] = pd.to_numeric(top_groups_df['Avg. KD'], errors='coerce')
            top_groups_df['Traffic Potential'] = pd.to_numeric(top_groups_df['Traffic Potential'], errors='coerce')

            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("🔑 Top 20 Clusters by Volume")
                st.dataframe(top_groups_df.style.format({'Total Volume': '{:,.0f}', 'Avg. KD': '{:.0f}', 'Traffic Potential': '{:,.0f}'}).background_gradient(cmap='viridis'))

            # Independently calculate and display Top 20 Clusters by Avg KD
            sorted_groups_by_kd = sorted(metrics.items(), key=lambda x: x[1]['Avg. KD'])
            top_groups_by_kd = sorted_groups_by_kd[:20]

            top_groups_df_by_kd = pd.DataFrame(top_groups_by_kd, columns=['Cluster', 'Metrics'])
            top_groups_df_by_kd['Avg. KD'] = top_groups_df_by_kd['Metrics'].map(lambda x: x['Avg. KD'])
            top_groups_df_by_kd['Total Volume'] = top_groups_df_by_kd['Metrics'].map(lambda x: x['Total Volume'])
            top_groups_df_by_kd['Traffic Potential'] = top_groups_df_by_kd['Metrics'].map(lambda x: x['Traffic Potential'])

            with col2:
                st.subheader("🔑 Top 20 Clusters by Avg KD")
                st.dataframe(top_groups_df_by_kd.drop(columns=['Metrics']).style.format({'Total Volume': '{:,.0f}', 'Avg. KD': '{:.0f}', 'Traffic Potential': '{:,.0f}'}).background_gradient(cmap='viridis'))

            # Independently calculate and display Top 20 Clusters by Traffic Potential
            sorted_groups_by_tp = sorted(metrics.items(), key=lambda x: x[1]['Traffic Potential'], reverse=True)
            top_groups_by_tp = sorted_groups_by_tp[:20]

            top_groups_df_by_tp = pd.DataFrame(top_groups_by_tp, columns=['Cluster', 'Metrics'])
            top_groups_df_by_tp['Traffic Potential'] = top_groups_df_by_tp['Metrics'].map(lambda x: x['Traffic Potential'])
            top_groups_df_by_tp['Total Volume'] = top_groups_df_by_tp['Metrics'].map(lambda x: x['Total Volume'])
            top_groups_df_by_tp['Avg. KD'] = top_groups_df_by_tp['Metrics'].map(lambda x: x['Avg. KD'])

            with col3:
                st.subheader("🔑 Top 20 Clusters by Potential")
                st.dataframe(top_groups_df_by_tp.drop(columns=['Metrics']).style.format({'Traffic Potential': '{:,.0f}', 'Total Volume': '{:,.0f}', 'Avg. KD': '{:.0f}'}).background_gradient(cmap='viridis'))

            # Display sorting options for any column
            st.subheader("Filter Full Sheet")
            col_sort, col_order = st.columns(2)
            with col_sort:
                sort_column = st.selectbox("Select the column to sort by:", data.columns)
            with col_order:
                sort_order = st.radio("Select the order:", ["Ascending", "Descending"], index=1)

            # Apply sorting based on user selection
            ascending = True if sort_order == "Ascending" else False
            filtered_data = data.sort_values(by=sort_column, ascending=ascending)

            # Hide the "#" column if it exists
            if "#" in filtered_data.columns:
                filtered_data = filtered_data.drop(columns=["#"])

            st.subheader(f"📄 Filtered Full Sheet by {sort_column}")
            st.dataframe(filtered_data.style.format({
                'Volume': '{:,.0f}', 'Difficulty': '{:,.0f}', 'Traffic potential': '{:,.0f}', 'Global volume': '{:,.0f}',
                'CPC': '{:.2f}', 'CPS': '{:.2f}'
            }).background_gradient(cmap='viridis'))

            # User input for sorting by Parent Keyword
            st.subheader("📄 Sorted Full Sheet by cluster containing")
            col_keyword, col_keyword_input = st.columns([1, 2])
            with col_keyword:
                st.text("Enter a value to sort cluster by:")
            with col_keyword_input:
                keyword = st.text_input("", help="Enter the keyword to filter cluster.", key="keyword_input")

            # Add custom CSS for styling
            st.markdown("""
                <style>
                div[role="textbox"] input {
                    width: 50% !important;
                    margin-top: -20px !important;
                }
                </style>
            """, unsafe_allow_html=True)

            if keyword:
                keyword_sorted_data = data[data[keyword_column].str.contains(keyword, case=False, na=False)]
                if not keyword_sorted_data.empty:
                    st.subheader(f"📄 Sorted Full Sheet by cluster containing '{keyword}'")
                    if "#" in keyword_sorted_data.columns:
                        keyword_sorted_data = keyword_sorted_data.drop(columns=["#"])
                    keyword_sorted_data.insert(0, 'Parent Keyword', keyword_sorted_data.pop('Parent Keyword'))
                    st.dataframe(keyword_sorted_data.style.format({
                'Volume': '{:,.0f}', 'Difficulty': '{:,.0f}', 'Traffic potential': '{:,.0f}', 'Global volume': '{:,.0f}',
                'CPC': '{:.2f}', 'CPS': '{:.2f}'
            }).background_gradient(cmap='viridis'))
                    
                    filtered_data = keyword_sorted_data

                else:
                    st.write("No matches found.")

            # Calculate unique counts and sum values for the filtered data
            unique_counts_filtered, sum_counts_filtered, unique_parent_keyword_filtered = count_unique_and_sum(filtered_data)
            unique_counts_filtered_df = pd.DataFrame(unique_counts_filtered, columns=['Unique Counts']).transpose()
            sum_counts_filtered_df = pd.DataFrame(sum_counts_filtered, columns=['Sum Counts']).transpose()

            # Convert unique parent keyword to DataFrame and transpose it for horizontal display
            unique_parent_keyword_filtered_df = pd.DataFrame(unique_parent_keyword_filtered, columns=['Unique Cluster']).transpose()

            # Place the tables side by side
            col_left, col_right = st.columns(2)
            with col_left:
                st.subheader("Unique Counts and Sum Counts")
                combined_counts_filtered_df = pd.concat([unique_counts_filtered_df, sum_counts_filtered_df])
                st.dataframe(combined_counts_filtered_df.style.format({
                    'Parent Keyword': '{:,.0f}', 'Keyword': '{:,.0f}', 'SERP Features': '{:,.0f}', 'Country': '{:,.0f}',
                    'Volume': '{:,.0f}', 'Difficulty': '{:,.0f}', 'Traffic potential': '{:,.0f}', 'Global volume': '{:,.0f}', 'CPC': '{:,.2f}', 'CPS': '{:,.2f}'
                }).background_gradient(cmap='viridis'))
            with col_right:
                st.subheader("Unique Clusters")
                st.dataframe(unique_parent_keyword_filtered_df)

            # Calculate Traffic for filtered data
            traffic_monthly_filtered = filtered_data['Volume'].sum()
            traffic_potential_desktop_filtered = int(traffic_monthly_filtered * 0.15)
            traffic_potential_mobile_filtered = int(traffic_monthly_filtered * 0.10)

            # Calculate Conversions for filtered data
            conversion_rates = [1.00, 0.50, 0.25]
            conversion_data = {
                "CTR": [f"{rate:.2f}%" for rate in conversion_rates],
                "Potential Desktop": [int(traffic_potential_desktop_filtered * (rate / 100)) for rate in conversion_rates],
                "Potential Mobile": [int(traffic_potential_mobile_filtered * (rate / 100)) for rate in conversion_rates]
            }
            conversion_df_filtered = pd.DataFrame(conversion_data)

            # Section for selecting Average Order Value (AOV)
            col_centered = st.columns([1, 1, 1])
            with col_centered[1]:
                st.subheader("Select Average Order Value (AOV):")
                selected_aov = st.number_input(" ", min_value=0, value=100, step=1, format="%d")
                st.write(f"You entered AOV value: ${selected_aov}")
           
            # Calculate Revenue for filtered data
            high_conversion_rate = 0.10
            medium_conversion_rate = 0.05
            low_conversion_rate = 0.025

            revenue_data = {
                "Ranges": ["High", "Medium", "Low"],
                "Potential Desktop": [
                    f"${traffic_potential_desktop_filtered * high_conversion_rate * selected_aov:,.2f}",
                    f"${traffic_potential_desktop_filtered * medium_conversion_rate * selected_aov:,.2f}",
                    f"${traffic_potential_desktop_filtered * low_conversion_rate * selected_aov:,.2f}"
                ],
                "Potential Mobile": [
                    f"${traffic_potential_mobile_filtered * high_conversion_rate * selected_aov:,.2f}",
                    f"${traffic_potential_mobile_filtered * medium_conversion_rate * selected_aov:,.2f}",
                    f"${traffic_potential_mobile_filtered * low_conversion_rate * selected_aov:,.2f}"
                ]
            }

            revenue_df = pd.DataFrame(revenue_data)

            # Display Traffic, Conversions, and Revenue side by side
            col_traffic, col_conversions, col_revenue = st.columns(3)
            with col_traffic:
                st.subheader("TRAFFIC")
                traffic_data_filtered = {
                    "": ["Traffic Monthly"],
                    "Potential Desktop": [traffic_potential_desktop_filtered],
                    "Potential Mobile": [traffic_potential_mobile_filtered]
                }
                traffic_df_filtered = pd.DataFrame(traffic_data_filtered)
                st.table(traffic_df_filtered)

            with col_conversions:
                st.subheader("CONVERSIONS")
                st.table(conversion_df_filtered)

            with col_revenue:
                st.subheader("REVENUE")
                st.table(revenue_df.style.set_table_styles([{
                    'selector': 'th',
                    'props': [('background-color', '#ffffff'), ('color', 'black'), ('text-align', 'center')]
                }, {
                    'selector': 'td',
                    'props': [('text-align', 'center')]
                }]))
            
