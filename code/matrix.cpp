#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <mpi.h>
#include <unistd.h> 
#include <sstream>

using namespace std;

const int MSG_TAG = 0;

int getRandomDestination(int my_rank, int world_size) {
    if (world_size == 1) return my_rank;
    int dest_rank;
    do {
        dest_rank = rand() % world_size;
    } while (dest_rank == my_rank); 
    return dest_rank;
}

void print_matrix(const vector<int>& matrix, int world_size) {
    cout << "{\n";
    for (int i = 0; i < world_size; ++i) {
        cout << "    P" << i << ": [";
        
        for (int j = 0; j < world_size; ++j) {
            cout << matrix[i * world_size + j];
            if (j < world_size - 1) {
                cout << ", ";
            }
        }
        cout << "]";
        if (i < world_size - 1) {
            cout << ",\n";
        }
    }
    cout << "\n}\n";
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    srand(time(NULL) + world_rank);
    
    int matrix_size = world_size * world_size;
    
    vector<int> matrix_clock(matrix_size, 0);
    
    int messages_sent = 0;
    const int MAX_MESSAGES_PER_PROCESS = 10;
    bool terminated = false;
    
    vector<int> received_matrix(matrix_size); 
    MPI_Request recv_request;
    MPI_Status recv_status;
    bool recv_posted = false;

    if (world_size > 1) {
        MPI_Irecv(received_matrix.data(), matrix_size, MPI_INT, MPI_ANY_SOURCE, MSG_TAG, MPI_COMM_WORLD, &recv_request);
        recv_posted = true;
    }

    while (!terminated) {
        usleep(500000); 

        matrix_clock[world_rank * world_size + world_rank]++;
        cout << "rank " << world_rank << ": internal event. new clock M[" << world_rank << "][" << world_rank << "] = " 
             << matrix_clock[world_rank * world_size + world_rank] << endl;
        
        if (messages_sent < MAX_MESSAGES_PER_PROCESS && world_size > 1) {
            matrix_clock[world_rank * world_size + world_rank]++; 
            
            int dest_rank = getRandomDestination(world_rank, world_size);
            
            MPI_Request send_request;
    
            MPI_Isend(matrix_clock.data(), matrix_size, MPI_INT, dest_rank, MSG_TAG, MPI_COMM_WORLD, &send_request);
            
            cout << "rank " << world_rank << ": sent message with clock to rank " << dest_rank << ". M[" 
                 << world_rank << "][" << world_rank << "] = " 
                 << matrix_clock[world_rank * world_size + world_rank] << endl;
                  print_matrix(matrix_clock, world_size);
            messages_sent++;
        }
        
        if (recv_posted) {
            int flag = 0;
            MPI_Test(&recv_request, &flag, &recv_status);

            if (flag) { 
                int sender_rank = recv_status.MPI_SOURCE;
                
                matrix_clock[world_rank * world_size + world_rank]++; 
                
                for (int j = 0; j < world_size; ++j) {
                    for (int k = 0; k < world_size; ++k) {
                        int index = j * world_size + k;
                        matrix_clock[index] = max(matrix_clock[index], received_matrix[index]);
                    }
                }
                
                cout << "rank " << world_rank << ": received message from rank " << sender_rank
                          << ". Merged clock. New M[" << world_rank << "][" << world_rank << "] = " 
                          << matrix_clock[world_rank * world_size + world_rank] << endl;

                           print_matrix(matrix_clock, world_size);
              
                MPI_Irecv(received_matrix.data(), matrix_size, MPI_INT, MPI_ANY_SOURCE, MSG_TAG, MPI_COMM_WORLD, &recv_request);
            }
        }

        int global_messages_sent;
        MPI_Reduce(&messages_sent, &global_messages_sent, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (world_rank == 0) {
            if (global_messages_sent >= MAX_MESSAGES_PER_PROCESS * world_size) {
                terminated = true;
                cout << "\nRank 0: Total messages sent limit reached. Initiating termination.\n" << endl;
            }
        }
    
        MPI_Bcast(&terminated, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    } 

    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0) {
        cout << "\nFinal Matrix Clock State:\n";
        print_matrix(matrix_clock, world_size);
    }
    
    MPI_Finalize();
    return 0;
}