from load_data import divide_samples,get_stats_from_file
import random
def main():
    book = 'raw_data_book_2019'
    elec = 'raw_data_electronics_reviews_2004'
    laptop = 'raw_data_laptop_2014'
    rest = 'raw_data_restaurant_2014'

    # all book 
    print('book dataset')

    book_path = 'data_out/' + book + '.txt'
    train_book = 'train/' + book + '.txt'
    test_book = 'test/' + book + '.txt'
    divide_samples(book_path,train_book,test_book)

    print('train')
    get_stats_from_file(train_book)
    print('test')
    get_stats_from_file(test_book)

    

    # all elec 
    print('electronics dataset')

    elec_path = 'data_out/'+elec+'.txt'
    train_elec = 'train/'+elec+'.txt'
    test_elec = 'test/'+elec+'.txt'
    divide_samples(elec_path,train_elec,test_elec)

    print('train')
    get_stats_from_file(train_elec)
    print('test')
    get_stats_from_file(test_elec)

   

    # all laptop 
    print('laptop dataset')

    laptop_path = 'data_out/'+laptop+'.txt'
    train_laptop = 'train/'+laptop+'.txt'
    test_laptop = 'test/'+laptop+'.txt'
    divide_samples(laptop_path,train_laptop,test_laptop)

    print('train')
    get_stats_from_file(train_laptop)
    print('test')
    get_stats_from_file(test_laptop)

    # all rest 
    print('restauarant dataset')

    rest_path = 'data_out/'+rest+'.txt'
    train_rest = 'train/'+rest+'.txt'
    test_rest = 'test/'+rest+'.txt'
    divide_samples(rest_path,train_rest,test_rest)

    print('train')
    get_stats_from_file(train_rest)
    print('test')
    get_stats_from_file(test_rest)

    # validation set
    train_book = 'train/' + book + '.txt'
    val_book = 'val/' + book + '.txt'
    train_book2 = 'train_small/' + book + '.txt'
    divide_samples(train_book,train_book2,val_book)

    train_elec = 'train/' + elec + '.txt'
    val_elec = 'val/' + elec + '.txt'
    train_elec2 = 'train_small/' + elec + '.txt'
    divide_samples(train_elec,train_elec2,val_elec)

    train_laptop = 'train/' + laptop + '.txt'
    val_laptop = 'val/' + laptop + '.txt'
    train_laptop2 = 'train_small/' + laptop + '.txt'
    divide_samples(train_laptop,train_laptop2,val_laptop)

    train_rest = 'train/' + rest + '.txt'
    val_rest = 'val/' + rest + '.txt'
    train_rest2 = 'train_small/' + rest + '.txt'
    divide_samples(train_rest,train_rest2,val_rest)

def add_domain(input_file_path,output_file_path,domain):
    # Open input file for reading and output file for writing
    with open(input_file_path, 'r', encoding="latin-1") as input_file, open(output_file_path, 'w', encoding="latin-1") as output_file:
        line_count = 0
        for line in input_file:
            # Write the current line to the output file
            output_file.write(line)
            # Increment line count
            line_count += 1
            # If the current line count is a multiple of three, add a new line
            if line_count % 3 == 0:
                output_file.write(domain + "\n")

def concat_datasets(input_file_path1,input_file_path2,input_file_path3,output_file_path):
    random.seed(123)
    # Read lines from the input file

    with open(input_file_path1, "r", encoding='latin-1') as input_file:
        lines = input_file.readlines()

    # Split lines into groups of four (samples)
    samples1 = [lines[i:i+4] for i in range(0, len(lines), 4)]

    with open(input_file_path2, "r", encoding='latin-1') as input_file:
        lines = input_file.readlines()

    # Split lines into groups of four (samples)
    samples2 = [lines[i:i+4] for i in range(0, len(lines), 4)]

    with open(input_file_path3, "r", encoding='latin-1') as input_file:
        lines = input_file.readlines()

    # Split lines into groups of four (samples)
    samples3 = [lines[i:i+4] for i in range(0, len(lines), 4)]


    samples = samples1 + samples2 + samples3
    # Shuffle the samples randomly
    random.shuffle(samples)

    # Flatten the list of samples back into lines
    shuffled_lines = [line for sample in samples for line in sample]

    # Write the shuffled lines to the output file
    with open(output_file_path, "w", encoding='utf-8') as output_file:
        output_file.writelines(shuffled_lines)

if __name__ == '__main__':
    main()
    datasets = ['train','train_small', 'val', 'test']
    reviews = [('book_2019','book'), ('restaurant_2014','restaurant'),('laptop_2014','laptop'),('electronics_reviews_2004','electronic')]
    for rev,dom in reviews:
        for data in datasets:
            train_book = ''+data+'/raw_data_'+rev+'.txt'
            domain_train_book = ''+data+'/domain_raw_data_'+rev+'.txt'
            
            add_domain(train_book,domain_train_book,dom)
    
    for data in datasets:
        train_book  = ''+data+'/domain_raw_data_book_2019.txt'
        train_laptop  = ''+data+'/domain_raw_data_laptop_2014.txt'
        train_restaurant  = ''+data+'/domain_raw_data_restaurant_2014.txt'
        #train_elec  = ''+data+'/domain_raw_data_electronics_reviews_2004.txt'
        out = ''+data+'/domain_raw_data_all.txt'
        concat_datasets(train_book,train_laptop,train_restaurant,out)