#include <iostream>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include <unistd.h> 

using namespace std;

// Tags for different message types
const int ELECTION_TAG = 20;
const int ELECTED_TAG = 21;

// Message structure for the Election phase
struct ElectionMessage {
    int candidate_id;
};

// Message structure for the Elected/Coordinator phase
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

    // --- Topology and State ---
    const int next_rank = (world_rank + 1) % world_size; 
    const int prev_rank = (world_rank - 1 + world_size) % world_size; 

    int leader_id = -1;
    bool has_forwarded_own_id = false; 

    // --- Non-blocking Receive Setup (Listening from prev_rank) ---
    ElectionMessage recv_election_msg;
    MPI_Request election_recv_request;
    
    // Post Irecv for Election messages
    MPI_Irecv(&recv_election_msg, sizeof(ElectionMessage), MPI_BYTE, prev_rank, ELECTION_TAG, MPI_COMM_WORLD, &election_recv_request);

    ElectedMessage recv_elected_msg;
    MPI_Request elected_recv_request;
    
    // Post Irecv for Elected messages
    MPI_Irecv(&recv_elected_msg, sizeof(ElectedMessage), MPI_BYTE, prev_rank, ELECTED_TAG, MPI_COMM_WORLD, &elected_recv_request);

    MPI_Status status;

    // --- Initiation (Every process starts its own election simultaneously) ---
    // Every process acts as an initiator, sending its ID.
    ElectionMessage init_msg = {world_rank};
    MPI_Request init_send_request;
    MPI_Isend(&init_msg, sizeof(ElectionMessage), MPI_BYTE, next_rank, ELECTION_TAG, MPI_COMM_WORLD, &init_send_request);
    MPI_Request_free(&init_send_request);
    
    has_forwarded_own_id = true;
    cout << "Rank " << world_rank << ": Initiated election with ID " << world_rank << ". Sent to " << next_rank << endl;
    
    // --- Main Election Loop ---
    while (leader_id == -1) {
        usleep(100000); 

        // 1. Check for **ELECTION** message arrival
        int election_flag = 0;
        MPI_Test(&election_recv_request, &election_flag, &status);

        if (election_flag) {
            int arrived_id = recv_election_msg.candidate_id;
            
            // Repost Irecv immediately to keep listening
            MPI_Irecv(&recv_election_msg, sizeof(ElectionMessage), MPI_BYTE, prev_rank, ELECTION_TAG, MPI_COMM_WORLD, &election_recv_request);

            // --- Election Process Logic ---
            ElectionMessage outgoing_msg;
            bool should_forward = false;

            if (arrived_id > world_rank) {
                // Case 1: Arrived ID is greater, forward it.
                outgoing_msg.candidate_id = arrived_id;
                should_forward = true;
            } else if (arrived_id < world_rank && !has_forwarded_own_id) {
                // Case 2: Arrived ID is smaller, overwrite with own ID, and forward.
                outgoing_msg.candidate_id = world_rank;
                should_forward = true;
                has_forwarded_own_id = true;
            } else if (arrived_id == world_rank) {
                // Case 3: Arrived ID matches own ID. I am the Leader!
                leader_id = world_rank;
                cout << "\nRank " << world_rank << ": Received own ID. **I AM THE NEW LEADER.**" << endl;
                
                // Send Elected message
                ElectedMessage elected_msg = {leader_id};
                MPI_Request send_request;
                MPI_Isend(&elected_msg, sizeof(ElectedMessage), MPI_BYTE, next_rank, ELECTED_TAG, MPI_COMM_WORLD, &send_request);
                MPI_Request_free(&send_request);
                
                cout << "Rank " << world_rank << ": Sent ELECTED message to " << next_rank << endl;
            }
            
            // Forward the message if necessary
            if (should_forward) {
                MPI_Request send_request;
                MPI_Isend(&outgoing_msg, sizeof(ElectionMessage), MPI_BYTE, next_rank, ELECTION_TAG, MPI_COMM_WORLD, &send_request);
                MPI_Request_free(&send_request);
                cout << "Rank " << world_rank << ": Forwarding ID " << outgoing_msg.candidate_id << " to " << next_rank << endl;
            }
        } 

        // 2. Check for **ELECTED** message arrival
        int elected_flag = 0;
        MPI_Test(&elected_recv_request, &elected_flag, &status);

        if (elected_flag) {
            int announced_leader = recv_elected_msg.leader_id;
            
            if (announced_leader != world_rank) {
                // Set the leader and forward the message
                leader_id = announced_leader;
                
                // Forward the Elected message
                MPI_Request send_request;
                MPI_Isend(&recv_elected_msg, sizeof(ElectedMessage), MPI_BYTE, next_rank, ELECTED_TAG, MPI_COMM_WORLD, &send_request);
                MPI_Request_free(&send_request);
                
                cout << "Rank " << world_rank << ": Received and forwarded ELECTED message. Leader is **" << leader_id << "**." << endl;
            }
            // Once the leader is set, the loop terminates.
        }
    } 

    // --- Final Cleanup ---
    // Cancel and free the outstanding receive requests before finalizing
    MPI_Cancel(&election_recv_request);
    MPI_Cancel(&elected_recv_request);
    
    // Note: MPI_Wait is necessary on cancelled requests for proper cleanup in some MPI implementations
    MPI_Wait(&election_recv_request, &status);
    MPI_Wait(&elected_recv_request, &status); 

    MPI_Request_free(&election_recv_request);
    MPI_Request_free(&elected_recv_request);
    
    MPI_Barrier(MPI_COMM_WORLD);
    cout << "\nRank " << world_rank << " terminated. Final Leader: " << leader_id << endl;
    
    MPI_Finalize();
    return 0;
}