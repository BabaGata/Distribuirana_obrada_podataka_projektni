from motif_finding_bf import *
import time
import os
from mpi4py import MPI

start_time = time.time()
start_process = time.process_time()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(rank)

adjacency_matrix_size = 10
motif_size = 4
# CUDA, MPI, NORMAL
method = 'CUDA'

np.random.seed(123)
adjacency_matrix = np.random.randint(3, size=(adjacency_matrix_size, adjacency_matrix_size))
# nek generira brojeve do 10, a samo 1 neka racuna kao vezu
for i, row in enumerate(adjacency_matrix):
    for j, el in enumerate(row):
        if el != 1:
            adjacency_matrix[i, j] = 0
        if i == j:
            adjacency_matrix[i, j] = 0

G = nx.from_numpy_array(adjacency_matrix)

# if rank == 0:
#     sendmsg = G
# else:
#     sendmsg = None
# recvmsg = comm.bcast(sendmsg, root = 0)
# comm.Barrier()
# if rank != 0:
#     G = recvmsg
# print("Proces ranga", rank, "prošao je barijeru")

#print(nx.adjacency_matrix(G).todense())
# is_subgraph_normal / is_subgraph_cuda
motifs = find_network_motifs(G.edges, motif_size, is_subgraph_cuda)

end_time = time.time()
end_process = time.process_time()
total_time = end_time - start_time
total_process = end_process - start_process

print("Network Motifs:")
file = open("izlaz.txt", "a")
for motif in motifs:
    # print(motif)
    file.write(str(motif) + "\n")
file.close()

print('Total number of motifs: ', len(motifs))
print('Start time: ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
print('End time: ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
print('Total run time: ', total_time)
print('Total process time: ', total_process)
print('One motif sample: ', motifs[0])

write_string = '{0:<7} {1:<25} {2:<25} {3:<20} {4:<20} {5:<7} {6:<8}\r\n'.format(
    method, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)),
    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)), total_time,
    total_process, motif_size, adjacency_matrix_size)

if os.path.isfile('./stats.txt'):
    file = open("stats.txt", "a")
    file.write(write_string)
    file.close()
else:
    file = open("stats.txt", "a")
    file.write('{0:<7} {1:<25} {2:<25} {3:<20} {4:<20} {5:<7} {6:<8}\r\n'.format(
        'method', 'start_time', 'end_time', 'total_time', 'total_process', 'motif N', 'matrix N'))
    file.write(write_string)
    file.close()


write_csv = (f"{method},{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))},\\"
             f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))},\\"
             f"{total_time},{total_process},{motif_size},{adjacency_matrix_size}\r\n")

if os.path.isfile('./stats.txt'):
    file = open("stats.csv", "a")
    file.write(write_string)
    file.close()
else:
    file = open("stats.csv", "a")
    file.write('method,start_time,end_time,total_time,total_process,motif_N,matrix_N')
    file.write(write_string)
    file.close()