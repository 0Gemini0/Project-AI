from data import Data

test = Data("unsupervised", preprocess=False)
test.save_csv_separate("test\\", "_test")
