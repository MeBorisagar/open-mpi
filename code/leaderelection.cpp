#include <iostream>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include <unistd.h> 

using namespace std;


const int ELECTION_TAG = 20;
const int ELECTED_TAG = 21;


struct ElectionMessage {
    int candidate_id;
};


struct ElectedMessage {
    int leader_id;
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < 2) {
        cerr << "Need at least 2 processes for ring election." << endl;
        MPI_Finalize();
        return 0;
    }

   
    const int next_rank = (world_rank + 1) % world_size; 
    const int prev_rank = (world_rank - 1 + world_size) % world_size; 

    int leader_id = -1;
    bool has_forwarded_own_id = false; 


    ElectionMessage recv_election_msg;
    MPI_Request election_recv_request;
    
   
    MPI_Irecv(&recv_election_msg, sizeof(ElectionMessage), MPI_BYTE, prev_rank, ELECTION_TAG, MPI_COMM_WORLD, &election_recv_request);

    ElectedMessage recv_elected_msg;
    MPI_Request elected_recv_request;
    
  
    MPI_Irecv(&recv_elected_msg, sizeof(ElectedMessage), MPI_BYTE, prev_rank, ELECTED_TAG, MPI_COMM_WORLD, &elected_recv_request);

    MPI_Status status;

    ElectionMessage init_msg = {world_rank};
    MPI_Request init_send_request;
    MPI_Isend(&init_msg, sizeof(ElectionMessage), MPI_BYTE, next_rank, ELECTION_TAG, MPI_COMM_WORLD, &init_send_request);
    MPI_Request_free(&init_send_request);
    
    has_forwarded_own_id = true;
    cout << "Rank " << world_rank << ": Initiated election with ID " << world_rank << ". Sent to " << next_rank << endl;
    

    while (leader_id == -1) {
        usleep(100000); 


        int election_flag = 0;
        MPI_Test(&election_recv_request, &election_flag, &status);

        if (election_flag) {
            int arrived_id = recv_election_msg.candidate_id;
            

            MPI_Irecv(&recv_election_msg, sizeof(ElectionMessage), MPI_BYTE, prev_rank, ELECTION_TAG, MPI_COMM_WORLD, &election_recv_request);

  
            ElectionMessage outgoing_msg;
            bool should_forward = false;

            if (arrived_id > world_rank) {
                outgoing_msg.candidate_id = arrived_id;
                should_forward = true;
            } else if (arrived_id < world_rank && !has_forwarded_own_id) {
                outgoing_msg.candidate_id = world_rank;
                should_forward = true;
                has_forwarded_own_id = true;
            } else if (arrived_id == world_rank) {
                leader_id = world_rank;
                cout << "\nRank " << world_rank << ": Received own ID. **I AM THE NEW LEADER.**" << endl;
                
                ElectedMessage elected_msg = {leader_id};
                MPI_Request send_request;
                MPI_Isend(&elected_msg, sizeof(ElectedMessage), MPI_BYTE, next_rank, ELECTED_TAG, MPI_COMM_WORLD, &send_request);
                MPI_Request_free(&send_request);
                
                cout << "Rank " << world_rank << ": Sent ELECTED message to " << next_rank << endl;
            }
            
            if (should_forward) {
                MPI_Request send_request;
                MPI_Isend(&outgoing_msg, sizeof(ElectionMessage), MPI_BYTE, next_rank, ELECTION_TAG, MPI_COMM_WORLD, &send_request);
                MPI_Request_free(&send_request);
                cout << "Rank " << world_rank << ": Forwarding ID " << outgoing_msg.candidate_id << " to " << next_rank << endl;
            }
        } 

        
        int elected_flag = 0;
        MPI_Test(&elected_recv_request, &elected_flag, &status);

        if (elected_flag) {
            int announced_leader = recv_elected_msg.leader_id;
            
            if (announced_leader != world_rank) {
            
                leader_id = announced_leader;
                
    
                MPI_Request send_request;
                MPI_Isend(&recv_elected_msg, sizeof(ElectedMessage), MPI_BYTE, next_rank, ELECTED_TAG, MPI_COMM_WORLD, &send_request);
                MPI_Request_free(&send_request);
                
                cout << "Rank " << world_rank << ": Received and forwarded ELECTED message. Leader is **" << leader_id << "**." << endl;
            }
         
        }
    } 

    MPI_Cancel(&election_recv_request);
    MPI_Cancel(&elected_recv_request);
    
    MPI_Wait(&election_recv_request, &status);
    MPI_Wait(&elected_recv_request, &status); 

    MPI_Request_free(&election_recv_request);
    MPI_Request_free(&elected_recv_request);
    
    MPI_Barrier(MPI_COMM_WORLD);
    cout << "\nRank " << world_rank << " terminated. Final Leader: " << leader_id << endl;
    
    MPI_Finalize();
    return 0;
}