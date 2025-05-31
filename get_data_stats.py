# Method for preparing raw data for the restaurant, laptop, book and hotel datasets.
#
# https://github.com/jorisknoester/DAT-LCR-Rot-hop-PLUS-PLUS
#
# Adapted from van Berkum, van Megen, Savelkoul, and Weterman (2021).
# https://github.com/stefanvanberkum/CD-ABSC
#
# van Berkum, S., van Megen, S., Savelkoul, M., and Weterman, P. (2021) Fine-Tuning for Cross-Domain
# Aspect-Based Sentiment Classification. Theoretical report Erasmus University Rotterdam.

from data_book_hotel import read_book_hotel
from data_rest_lapt import read_rest_lapt
from load_data import divide_samples, get_stats_from_file
import os 


def main():
        train_laptop = 'data_out/laptop/raw_data_laptop_train_2014.txt'
        test_laptop = 'data_out/laptop/raw_data_laptop_test_2014.txt'
        train_restaurant = 'data_out/restaurant/raw_data_restaurant_train_2014.txt'
        test_restaurant = 'data_out/restaurant/raw_data_restaurant_test_2014.txt'
        train_book = 'data_out/book/raw_data_book_train_2019.txt'
        test_book = 'data_out/book/raw_data_book_test_2019.txt'
        print("Train laptop:")
        get_stats_from_file(train_laptop)
        print("Test laptop: ")
        get_stats_from_file(test_laptop)
        print("Train restaurant: ")
        get_stats_from_file(train_restaurant)
        print("Test restaurant: ")
        get_stats_from_file(test_restaurant)
        print("Train book: ")
        get_stats_from_file(train_book)
        print("Test book:")
        get_stats_from_file(test_book)
   

if __name__ == '__main__':
    main()
