#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <mpi.h>
#include <unistd.h> 

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


void print_vector(const vector<int>& clock) {
    cout << "[";
    for (size_t i = 0; i < clock.size(); ++i) {
        cout << clock[i];
        if (i < clock.size() - 1) {
            cout << ", ";
        }
    }
    cout << "]";
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    srand(time(NULL) + world_rank);

    vector<int> vector_clock(world_size, 0);
    
    int messages_sent = 0;
    const int MAX_MESSAGES_PER_PROCESS = 10;
    bool terminated = false;
    
    vector<int> received_vector(world_size); 
    MPI_Request recv_request;
    MPI_Status recv_status;
    bool recv_posted = false;

    if (world_size > 1) {
        MPI_Irecv(received_vector.data(), world_size, MPI_INT, MPI_ANY_SOURCE, MSG_TAG, MPI_COMM_WORLD, &recv_request);
        recv_posted = true;
    }

    while (!terminated) {
        usleep(500000);

        vector_clock[world_rank]++;
        cout << "rank " << world_rank << ": internal event. new clock = ";
        print_vector(vector_clock);
        cout << endl;
        
        if (messages_sent < MAX_MESSAGES_PER_PROCESS && world_size > 1) {
            
            vector_clock[world_rank]++; 
            
            int dest_rank = getRandomDestination(world_rank, world_size);
            
            MPI_Request send_request;
            MPI_Isend(vector_clock.data(), world_size, MPI_INT, dest_rank, MSG_TAG, MPI_COMM_WORLD, &send_request);
            
            MPI_Request_free(&send_request); 

            cout << "rank " << world_rank << ": sent message with clock ";
            print_vector(vector_clock);
            cout << " to rank " << dest_rank << endl;
            messages_sent++;
        }
        
        if (recv_posted) {
            int flag = 0;
            MPI_Test(&recv_request, &flag, &recv_status);

            if (flag) { 
                
                vector_clock[world_rank]++; 
                
                for (int i = 0; i < world_size; ++i) {
                    vector_clock[i] = max(vector_clock[i], received_vector[i]);
                }
                
                cout << "rank " << world_rank << ": received message from rank " << recv_status.MPI_SOURCE
                          << " with clock ";
                print_vector(received_vector);
                cout << ". new clock = ";
                print_vector(vector_clock);
                cout << endl;

                MPI_Irecv(received_vector.data(), world_size, MPI_INT, MPI_ANY_SOURCE, MSG_TAG, MPI_COMM_WORLD, &recv_request);
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

    if (recv_posted) {
        MPI_Cancel(&recv_request);
        MPI_Status status;
        MPI_Wait(&recv_request, &status); 
        MPI_Request_free(&recv_request);
    }
    
    MPI_Finalize();
    return 0;
}