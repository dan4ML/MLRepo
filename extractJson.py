import json

def load_spider_data(file_path, num_rows=None):
    """Load and preprocess Spider dataset from the given file path."""
    print("Loading data for: ", file_path)

    data = []

    with open(file_path, 'r', encoding='utf-8') as f:
        spider_data = json.load(f)

    # Loop through each entry and format as {source, target}
    for i, entry in enumerate(spider_data):
        if num_rows != "all" and num_rows is not None and i >= int(num_rows):
            break

        question = entry['question']
        query = entry['query']

        # Create a dictionary for each entry
        data.append({
            "source": f"question: {question}",
            "target": query
        })

        print("question: ", question, " query: ", query)

    return data
