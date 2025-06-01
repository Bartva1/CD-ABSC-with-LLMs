import json
import os 

def process_file(input_path, output_path, toPositive=True):
    with open(input_path, 'r', encoding='latin1') as file:
        lines = [line.strip() for line in file if line.strip()]

    output_lines = []
    for i in range(0, len(lines), 3):
        sentence = lines[i]
        target = lines[i + 1]
        sentiment = int(lines[i + 2])

        if sentiment == 0:
            sentiment = 1 if toPositive else -1

        output_lines.extend([sentence, target, str(sentiment)])

    with open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.write('\n'.join(output_lines) + '\n')


for domain in ["book", "laptop", "restaurant"]:
    for polarity in [True, False]:
        year = 2019 if domain == "book" else 2014
        polarity_text = "positive" if polarity else "negative"
        input_directory = f"data_out/{domain}"
        output_directory = f"data_out/{domain}/no_neutral"
        os.makedirs(output_directory, exist_ok=True)
        
        train_input = os.path.join(input_directory, f"raw_data_{domain}_train_{year}.txt")
        test_input = os.path.join(input_directory, f"raw_data_{domain}_test_{year}.txt")
        train_output = os.path.join(output_directory, f"{polarity_text}_data_{domain}_train_{year}.txt")
        test_output = os.path.join(output_directory, f"{polarity_text}_data_{domain}_test_{year}.txt")


        process_file(train_input, train_output, polarity)
        process_file(test_input, test_output, polarity)