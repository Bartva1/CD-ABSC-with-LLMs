import re
def get_domain_data(domain,data):
    key_to_check = 'id'
    
    # Initialize a new list to store dictionaries with same values for the key
    filtered_list = []
    
    # Iterate through the list and filter dictionaries with same values
    for dictionary in data:
        if domain in dictionary.get(key_to_check):
            
            filtered_list.append(dictionary)
    
    # Print the new list of dictionaries
    return filtered_list


def get_descriptive(data):
    key_to_check = 'output'
    
    # Initialize a new list to store dictionaries with same values for the key
    filtered_list = []
     
    positive = 0
    neutral = 0
    negative = 0
    # Iterate through the list and filter dictionaries with same values
    for dictionary in data:
        if dictionary.get(key_to_check):
            # Split the input string into individual pairs
                        # Extract pairs using regular expressions
            
            pairs = re.findall(r'\[([^]]+)\]', dictionary.get(key_to_check))

            # Extract the first and second elements of each pair and store them in tuples
            result = [tuple(pair.split(', ')) for pair in pairs]

            for tup in result:
                word = tup[0]
                sentiment = tup[1]
                
                if not word:
                    print('rip')
                if 'positive' in sentiment:
                    positive+=1
                if 'negative' in sentiment:
                    negative +=1 
                if 'neutral' in sentiment:
                    neutral +=1
            # Print the result
            filtered_list.append(result)

    print(f"{data[0].get('domain')} - number of positive: {positive}, number of negative: {negative}, number of neutral:{neutral}, total: {positive + neutral + negative}")
    print(f"{data[0].get('domain')} & {positive} & {round(100 *positive / (positive+negative+neutral))}\% & {negative} &{round(100 *negative / (positive+negative+neutral))}\% & {neutral}&{round(100 *neutral / (positive+negative+neutral))}\%")
    # Print the new list of dictionaries
    return filtered_list
    