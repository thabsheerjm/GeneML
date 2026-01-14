import chembl_downloader

if __name__ == '__main__':
    db_path = chembl_downloader.download_extract_sqlite()
    print(f"Downloaded Chembl dataset to :{db_path}")
