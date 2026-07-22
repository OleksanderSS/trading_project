import multiprocessing
import time
from src.data.management.data_manager import DataManager

def open_db(worker_id):
    try:
        print(f"Worker {worker_id} attempting to connect...")
        dm = DataManager(db_path="data/trading_data.duckdb")
        print(f"Worker {worker_id} successfully connected!")
        time.sleep(2) # Hold the lock
    except Exception as e:
        print(f"Worker {worker_id} failed: {e}")

if __name__ == "__main__":
    processes = []
    for i in range(3):
        p = multiprocessing.Process(target=open_db, args=(i,))
        processes.append(p)
        p.start()
        time.sleep(0.1) # Stagger slightly to guarantee conflict

    for p in processes:
        p.join()
    
    print("Concurrent DB connection test finished.")
