import os
import glob
from mpi4py import MPI

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "dataset")
    consulta_path = os.path.join(dataset_dir, "consulta.txt")

    TAG_REQUEST = 1
    TAG_FILE = 2
    TAG_STOP = 3

    if rank == 0:
        start_total = MPI.Wtime()
        with open(consulta_path, "r", encoding="utf-8") as f:
            query_words = [line.strip() for line in f if line.strip()]
        files = sorted(glob.glob(os.path.join(dataset_dir, "file_*.txt")))
    else:
        query_words = None
        files = None

    query_words = comm.bcast(query_words, root=0)
    query_set = set(query_words)

    local_start_time = MPI.Wtime()
    local_counts = {word: 0 for word in query_words}
    assigned_files_count = 0

    if rank == 0:
        file_iterator = iter(files)
        active_workers = size - 1
        
        while active_workers > 0:
            status = MPI.Status()
            comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            worker_rank = status.Get_source()
            
            try:
                next_file = next(file_iterator)
                comm.send(next_file, dest=worker_rank, tag=TAG_FILE)
            except StopIteration:
                comm.send(None, dest=worker_rank, tag=TAG_STOP)
                active_workers -= 1
    else:
        while True:
            comm.send(None, dest=0, tag=TAG_REQUEST)
            
            status = MPI.Status()
            file_path = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            
            if status.Get_tag() == TAG_STOP:
                break
                
            assigned_files_count += 1
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    for word in line.strip().split():
                        if word in query_set:
                            local_counts[word] += 1

    local_end_time = MPI.Wtime()
    local_processing_time = local_end_time - local_start_time

    all_local_counts = comm.gather(local_counts, root=0)
    all_local_times = comm.gather(local_processing_time, root=0)
    all_assigned_files = comm.gather(assigned_files_count, root=0)

    if rank == 0:
        end_total = MPI.Wtime()
        total_execution_time = end_total - start_total

        global_counts = {word: 0 for word in query_words}
        for r_idx, local_dict in enumerate(all_local_counts):
            if r_idx > 0:
                for word, count in local_dict.items():
                    global_counts[word] += count

        top_10 = sorted(global_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        print(f"\nTiempo de ejecución total MPI (Version 2): {total_execution_time:.6f} segundos")
        print("\nReporte por proceso:")
        for r in range(size):
            print(f"  - Proceso {r}: {all_assigned_files[r]} archivos procesados | Tiempo local: {all_local_times[r]:.6f} segundos")

        print("\nTop 10 palabras de consulta en los archivos (MPI v2):")
        for word, count in top_10:
            print(f"  {word}: {count}")

if __name__ == "__main__":
    main()