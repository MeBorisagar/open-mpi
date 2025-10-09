#include <iostream>
#include <vector>
#include <mpi.h>
#include <unistd.h> 

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size == 1) {
        cout << "Need more than one process for broadcast simulation." << endl;
        MPI_Finalize();
        return 0;
    }

    int received_id;
    

    vector<int> all_ids(world_size);

    cout << "Rank " << world_rank << " started." << endl;

  
    for (int root_rank = 0; root_rank < world_size; ++root_rank) {
        
        
        int broadcast_data;

        if (world_rank == root_rank) {
           
            broadcast_data = world_rank;
        }
        
        MPI_Bcast(&broadcast_data, 1,MPI_INT,root_rank, MPI_COMM_WORLD  );
        
      
        all_ids[root_rank] = broadcast_data;

        cout << "Rank " << world_rank << ": Received ID " << broadcast_data << " from Rank " << root_rank << endl;
        usleep(50000); 
    }


    cout << "\nRank " << world_rank << " finished and collected all IDs: [";
    for (int i = 0; i < world_size; ++i) {
        cout << all_ids[i];
        if (i < world_size - 1) {
            cout << ", ";
        }
    }
    cout << "]" << endl;

    MPI_Finalize();
    return 0;
}