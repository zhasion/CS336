import pandas as pd

def main():
    fpath = 'data/test.jsonl.gz'
    df = pd.read_json(fpath, lines=True, compression='gzip')
    print(df.head())

if __name__ == '__main__':
    main()