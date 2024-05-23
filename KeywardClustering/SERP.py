import pandas as pd
# Function to process the uploaded CSV file
def process_csv(input_file, output_file):
    try:
        # Reading the uploaded CSV file
        data = pd.read_csv(input_file)
        # Checking for required columns
        required_columns = ['keyword', 'search_volume', 'position', 'ranking_url']
        if not all(col in data.columns for col in required_columns):
            print("Error: CSV file doesn't contain all required columns.")
            return
        # Converting 'position' column to numeric values
        data['position'] = pd.to_numeric(data['position'], errors='coerce')
        # Filtering URLs with at least one keyword in top 10
        top_10_urls = data[data['position'] <= 19]['ranking_url'].unique()
        # List to store results
        results = []
        for url in top_10_urls:
            url_data = data[data['ranking_url'] == url]
            top_keyword = url_data[url_data['position'] <= 19].nlargest(1, 'search_volume')
            if not top_keyword.empty:
                best_ranking = top_keyword.iloc[0]['position']
                highest_ranking_keyword = top_keyword.iloc[0]['keyword']
                search_volume = top_keyword.iloc[0]['search_volume']
                results.append({'ranking_url': url, 'best_ranking': best_ranking,
                                'highest_ranking_keyword': highest_ranking_keyword,
                                'search_volume': search_volume})
        # Creating DataFrame from the results list
        result_df = pd.DataFrame(results, columns=['ranking_url', 'best_ranking', 'highest_ranking_keyword', 'search_volume'])
        # Writing results to a new CSV file
        result_df.to_csv(output_file, index=False)
        print(f"Exported data to '{output_file}' successfully.")
    except Exception as e:  
        print(f"An error occurred: {str(e)}")
# Replace 'import_file_path.csv' and 'output.csv' with your file names
process_csv('Keyward.CSV.csv', 'output.csv')
