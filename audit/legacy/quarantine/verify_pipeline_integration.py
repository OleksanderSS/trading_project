import pandas as pd
import glob
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataVerifier")

def verify_and_fix_parquet(file_path):
    try:
        df = pd.read_parquet(file_path)
        modified = False
        
        # Перевірка datetime
        if 'datetime' not in df.columns:
            logger.warning(f"Missing 'datetime' in {file_path}")

            # Шукаємо джерело для datetime
            found_date = False
            for col in ['publishedAt', 'published_date', 'date']:
                if col in df.columns:
                    df['datetime'] = pd.to_datetime(df[col], errors='coerce')
                    modified = True
                    found_date = True
                    break

            if not found_date:
                logger.error(f"Cannot fix 'datetime' in {file_path}")
                return False
        else:
            # Навіть якщо є, переконаємось, що формат правильний
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            modified = True


        # Перевірка ticker
        if 'ticker' not in df.columns:
            logger.warning(f"Missing 'ticker' in {file_path}")
            df['ticker'] = 'UNKNOWN'
            modified = True
            
        if modified:
            df.to_parquet(file_path, index=False)
            logger.info(f"Fixed and saved: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return False

def run_audit():
    data_dir = "data/colab/accumulated"
    files = glob.glob(os.path.join(data_dir, "*.parquet"))
    logger.info(f"Found {len(files)} files to audit.")
    
    for f in files:
        verify_and_fix_parquet(f)

if __name__ == "__main__":
    run_audit()
