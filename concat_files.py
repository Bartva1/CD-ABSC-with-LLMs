from load_data import concatenate_four_files, concatenate_two_files

def main():

    domain = "book"
    year = 2019

    if domain == "restaurant":
        train_file = "data_out/" + domain + "/raw_data_" + domain + "_train_" + str(year) + ".txt"
        test_file = "data_out/" + domain + "/raw_data_" + domain + "_test_" + str(year) + ".txt"
        out_file = "data_out/" + "raw_data_" + domain + "_" + str(year) + ".txt"
        concatenate_two_files(train_file,test_file,out_file)
    elif domain == "laptop":
        train_file = "data_out/" + domain + "/raw_data_" + domain + "_train_" + str(year) + ".txt"
        test_file = "data_out/" + domain + "/raw_data_" + domain + "_test_" + str(year) + ".txt"
        out_file = "data_out/" + "raw_data_" + domain + "_" + str(year) + ".txt"
        concatenate_two_files(train_file,test_file,out_file)
    elif domain == "electronics_reviews":
        Apex = "data_out/" + domain + "_"+str(year) + "/Apex/raw_data_Apex" + "_" + str(year) + ".txt"
        Camera = "data_out/" + domain +"_"+str(year) + "/Camera/raw_data_Camera" + "_" + str(year) + ".txt"
        Creative = "data_out/" + domain +"_"+str(year) + "/Creative/raw_data_Creative" + "_" + str(year) + ".txt"
        Nokia = "data_out/" + domain +"_"+str(year) + "/Nokia/raw_data_Nokia" + "_" + str(year) + ".txt"
        out_file = "data_out/" + "raw_data_" + domain + "_" + str(year) + ".txt"
        concatenate_four_files(Apex,Camera,Creative,Nokia,out_file)
    elif domain == 'book':
        source_file = 'data_out/book/raw_data_book_2019.txt'
        destination_file = 'data_out/raw_data_book_2019.txt'
                # Open the source file for reading
        with open(source_file, 'r', encoding="utf-8") as source:
            # Read the content of the source file
            content = source.read()

        # Open the destination file for writing
        with open(destination_file, 'w', encoding="utf-8") as destination:
            # Write the content to the destination file
            destination.write(content)
    else:
        print('error, domain not found')        
if __name__ == '__main__':
    main()
