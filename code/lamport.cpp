#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <mpi.h>
#include <unistd.h> 

using namespace std;

const int MSG_TAG = 0;
const int TERM_TAG = 1;

struct MessageData {
    int logical_timestamp;
};

int getRandomDestination(int my_rank, int world_size) {
    if (world_size == 1) return my_rank;
    int dest_rank;
    do {
        dest_rank = rand() % world_size;
    } while (dest_rank == my_rank); 
    return dest_rank;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    srand(time(NULL) + world_rank);

    int local_clock = 0;
    int messages_sent = 0;
    const int MAX_MESSAGES_PER_PROCESS = 10;
    bool terminated = false;
    
    MessageData received_msg;
    MPI_Request recv_request;
    MPI_Status recv_status;
    bool recv_posted = false;

    if (world_size > 1) {
        MPI_Irecv(&received_msg, sizeof(MessageData), MPI_BYTE, MPI_ANY_SOURCE, MSG_TAG, MPI_COMM_WORLD, &recv_request);
        recv_posted = true;
    }

    while (!terminated) {
        
        usleep(500000);

        local_clock++;
        cout << "rank " << world_rank << ": internal event. new clock = " << local_clock << endl;
        
    
        if (messages_sent < MAX_MESSAGES_PER_PROCESS && world_size > 1) {
            
            local_clock++; 
            
            int dest_rank = getRandomDestination(world_rank, world_size);
            MessageData send_msg;
            send_msg.logical_timestamp = local_clock;

            MPI_Request send_request;
            MPI_Isend(&send_msg, sizeof(MessageData), MPI_BYTE, dest_rank, MSG_TAG, MPI_COMM_WORLD, &send_request);
            
            MPI_Request_free(&send_request); 

            cout << "rank " << world_rank << ": sent message with timestamp " << local_clock 
             << " to rank " << dest_rank << endl;
            messages_sent++;
        }
        
        if (recv_posted) {
            int flag = 0;
            MPI_Test(&recv_request, &flag, &recv_status);

            if (flag) { 
                
                local_clock = max(local_clock, received_msg.logical_timestamp) + 1;
                
                cout << "rank " << world_rank << ": received message from rank " << recv_status.MPI_SOURCE
                          << " with timestamp " << received_msg.logical_timestamp
                          << ". new clock = " << local_clock << endl;

                MPI_Irecv(&received_msg, sizeof(MessageData), MPI_BYTE, MPI_ANY_SOURCE, MSG_TAG, MPI_COMM_WORLD, &recv_request);
            }
        }

        int global_messages_sent;
        MPI_Reduce(&messages_sent, &global_messages_sent, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (world_rank == 0) {
            if (global_messages_sent >= MAX_MESSAGES_PER_PROCESS * world_size) {
                terminated = true;
                cout << "\nRank 0: All " << global_messages_sent << " messages sent. Initiating termination.\n" << endl;
            }
        }
    
        MPI_Bcast(&terminated, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    } 

     
    // if (recv_posted) {
    //     MPI_Cancel(&recv_request);
    //     MPI_Status status;
    //     MPI_Wait(&recv_request, &status);
    //     MPI_Request_free(&recv_request);
    // }
    
    MPI_Finalize();
    return 0;
}