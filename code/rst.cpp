#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <mpi.h>
#include <unistd.h> 

using namespace std;

const int RST_MSG_TAG = 10;
const int ROOT_RANK = 0;

struct RSTMessage {
    int sender_rank;
};

vector<vector<int>> get_graph_topology(int world_size) {


    if (world_size == 2) {
        return {
            {1},       
            {0}  
                  
        };
    } 
    if (world_size == 4) {
        return {
            {1, 3},       
            {0, 2},       
            {1, 3},       
            {0, 2}        
        };
    } 
    // if (world_size > 1) {
    //     vector<vector<int>> adj(world_size);
    //     for (int i = 0; i < world_size; ++i) {
    //         if (i > 0) adj[i].push_back(i - 1); 
    //         if (i < world_size - 1) adj[i].push_back(i + 1); 
    //     }
    //     return adj;
    // }
    return vector<vector<int>>(world_size); 
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < 2) {
        cerr << "Need at least 2 processes for this simulation." << endl;
        MPI_Finalize();
        return 0;
    }

    vector<vector<int>> adjacency_list = get_graph_topology(world_size);
    const vector<int>& neighbors = adjacency_list[world_rank];

    int parent_rank = -1; 
    vector<int> children; 

    bool message_received = false;
    bool terminated = false;
    
    RSTMessage received_msg;
    MPI_Request recv_request = MPI_REQUEST_NULL;
    MPI_Status recv_status;
    
    if (world_rank != ROOT_RANK) {
        MPI_Irecv(&received_msg, sizeof(RSTMessage), MPI_BYTE, MPI_ANY_SOURCE, RST_MSG_TAG, MPI_COMM_WORLD, &recv_request);
    }

    cout << "Rank " << world_rank << " started with neighbors: ";
    for(int n : neighbors) cout << n << " ";
    cout << endl;

    if (world_rank == ROOT_RANK) {
        cout << "\nRank " << world_rank << " Root: initiating RST construction." << endl;
        RSTMessage send_msg;
        send_msg.sender_rank = world_rank;
        
        for (int dest_rank : neighbors) {
            MPI_Request send_request;
            MPI_Isend(&send_msg, sizeof(RSTMessage), MPI_BYTE, dest_rank, RST_MSG_TAG, MPI_COMM_WORLD, &send_request);
            
            cout << "Rank " << world_rank << " Root: sent RST message to neighbor " << dest_rank << endl;
        }
        terminated = true; 
    } 
    
    else {
        
        while (!terminated) {
            
            usleep(100000); 

            int flag = 0;
            if (recv_request != MPI_REQUEST_NULL) {
                MPI_Test(&recv_request, &flag, &recv_status);
            }

            if (flag) { 
                if (!message_received) {
                    parent_rank = received_msg.sender_rank;
                    message_received = true;

                    cout << "\nrank " << world_rank << ": received RST message from " << parent_rank 
                         << ". parent is set to " << parent_rank << "**." << endl;
                    
                    RSTMessage send_msg;
                    send_msg.sender_rank = world_rank;

                    for (int dest_rank : neighbors) {
                        if (dest_rank != parent_rank) {
                            MPI_Request send_request;
                            MPI_Isend(&send_msg, sizeof(RSTMessage), MPI_BYTE, dest_rank, RST_MSG_TAG, MPI_COMM_WORLD, &send_request);
                            MPI_Request_free(&send_request); 
                            cout << "rank " << world_rank << ": sent RST message to child neighbor " << dest_rank << endl;
                            children.push_back(dest_rank);
                        }
                    }
                    
                    terminated = true; 
                    
                    MPI_Cancel(&recv_request);
                    MPI_Status status_cancel;
                    MPI_Wait(&recv_request, &status_cancel); 
                } else {
                    cout << "Rank " << world_rank << ": Ignoring message from " << received_msg.sender_rank 
                         << " (Already terminated its role)." << endl;
                }
            }
        } 
    }

    MPI_Barrier(MPI_COMM_WORLD); 

    if (world_rank != ROOT_RANK) {
        cout << "Rank " << world_rank << ": Final RST result: Parent=" << parent_rank << endl;
    } else {
        cout << "Rank " << world_rank << ": Final RST result: Root (Parent=-1)" << endl;
    }
    
    MPI_Finalize();
    return 0;
}