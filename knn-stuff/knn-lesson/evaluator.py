import argparse
import pandas as pd

from joblib import dump, load


def evaluate_data(data=None):
    """
    input
    data - pandas dataframe of the iris dataset
    
    return 
    predicted labels
    """
    if data is None:
        print("You have not passed in data")
        return None
    features = ['petal length (cm)', 'petal width (cm)']
    data = data[features]
    knn_loaded = load("./models/knn_07072020.pkl")
    labels = knn_loaded.predict(data)
    return labels 


def evaluate_dataframe(full_filepath):
    df = pd.read_csv(full_filepath)
    labels = evaluate_data(data=df)
    return labels


def main():
    parser = argparse.ArgumentParser(description='predict labels on an iris csv file')
    parser.add_argument('-f', 
                        type=str,
                        help='full file path of iris csv')
    args = parser.parse_args()
    args = vars(args)
    labels = evaluate_dataframe(args['f'])
    print(labels) 
    
    
if __name__=="__main__":
    main()

