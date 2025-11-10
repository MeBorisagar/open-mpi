#include <iostream>
#include <vector>
#include <mpi.h>
#include <unistd.h>
#include <cstdlib>
#include <ctime>

using namespace std;

// Message Tags
const int REQ_TAG = 100;  // Request message
const int REP_TAG = 101;  // Reply message
const int DONE_TAG = 102; // Done with all CS executions

// Message Structures
struct RequestMessage {
    int timestamp;      // Lamport's clock value
    int process_id;     // Sender's process ID
};

struct ReplyMessage {
    int process_id;     // Sender's process ID
};

// Global Variables (per process)
int Ts_current = 0;              // Current Lamport's clock value
int Num_expected = 0;            // Expected number of REPLY messages
bool Cs_requested = false;       // Is CS requested?
vector<bool> Rep_deferred;       // Deferred reply flags
bool Priority = false;           // Priority flag

int world_rank;
int world_size;
int cs_executions = 0;           // Counter for CS executions
const int MAX_CS_EXECUTIONS = 3; // Each process enters CS this many times

// Function to update Lamport's clock on message receipt
void update_lamport_clock(int received_timestamp) {
    Ts_current = max(Ts_current, received_timestamp) + 1;
}

// Function to simulate Critical Section execution
void execute_critical_section() {
    cs_executions++;
    cout << "\n╔════════════════════════════════════════════════════╗" << endl;
    cout << "║ Process " << world_rank << " ENTERED Critical Section [" 
         << cs_executions << "/" << MAX_CS_EXECUTIONS << "]" << endl;
    cout << "║ Timestamp: " << Ts_current << endl;
    cout << "╚════════════════════════════════════════════════════╝\n" << endl;
    
    // Simulate some work in CS
    usleep(500000 + (rand() % 500000)); // 0.5 - 1 second
    
    cout << "Process " << world_rank << " EXITING Critical Section [" 
         << cs_executions << "/" << MAX_CS_EXECUTIONS << "]" << endl;
}

// EnterCS(): Request to enter critical section
void EnterCS() {
    Cs_requested = true;
    Ts_current++;
    Num_expected = world_size - 1;
    
    cout << "Process " << world_rank << " requesting CS (Timestamp: " 
         << Ts_current << ")" << endl;
    
    // Send REQUEST to all other processes
    RequestMessage req_msg;
    req_msg.timestamp = Ts_current;
    req_msg.process_id = world_rank;
    
    for (int j = 0; j < world_size; j++) {
        if (j != world_rank) {
            MPI_Send(&req_msg, sizeof(RequestMessage), MPI_BYTE, 
                    j, REQ_TAG, MPI_COMM_WORLD);
            cout << "  → Process " << world_rank << " sent REQ(" 
                 << Ts_current << ", " << world_rank << ") to Process " 
                 << j << endl;
        }
    }
    
    // Wait until all REPLY messages are received
    cout << "Process " << world_rank << " waiting for " << Num_expected 
         << " replies..." << endl;
    
    while (Num_expected > 0) {
        usleep(10000); // Small sleep to prevent busy waiting
        
        // Check for incoming messages
        MPI_Status status;
        int flag;
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
        
        if (flag) {
            if (status.MPI_TAG == REP_TAG) {
                ReplyMessage rep_msg;
                MPI_Recv(&rep_msg, sizeof(ReplyMessage), MPI_BYTE, 
                        status.MPI_SOURCE, REP_TAG, MPI_COMM_WORLD, &status);
                
                Num_expected--;
                cout << "  ← Process " << world_rank << " received REP from Process " 
                     << rep_msg.process_id << " (Remaining: " << Num_expected << ")" << endl;
            }
            else if (status.MPI_TAG == REQ_TAG) {
                // Handle incoming REQUEST while waiting
                RequestMessage req_msg;
                MPI_Recv(&req_msg, sizeof(RequestMessage), MPI_BYTE, 
                        status.MPI_SOURCE, REQ_TAG, MPI_COMM_WORLD, &status);
                
                cout << "  ← Process " << world_rank << " received REQ(" 
                     << req_msg.timestamp << ", " << req_msg.process_id 
                     << ") while waiting" << endl;
                
                // Calculate priority
                Priority = Cs_requested && 
                          ((req_msg.timestamp > Ts_current) || 
                           ((req_msg.timestamp == Ts_current) && 
                            (world_rank < req_msg.process_id)));
                
                if (Priority) {
                    // Defer the reply
                    Rep_deferred[req_msg.process_id] = true;
                    cout << "    Process " << world_rank << " DEFERRED reply to Process " 
                         << req_msg.process_id << " (Priority: mine)" << endl;
                } else {
                    // Send immediate reply
                    Rep_deferred[req_msg.process_id] = false;
                    ReplyMessage reply_msg;
                    reply_msg.process_id = world_rank;
                    MPI_Send(&reply_msg, sizeof(ReplyMessage), MPI_BYTE, 
                            req_msg.process_id, REP_TAG, MPI_COMM_WORLD);
                    cout << "    Process " << world_rank << " sent immediate REP to Process " 
                         << req_msg.process_id << endl;
                }
                
                update_lamport_clock(req_msg.timestamp);
            }
        }
    }
    
    cout << "✓ Process " << world_rank << " received all replies, entering CS!" << endl;
}

// ExitCS(): Exit critical section and send deferred replies
void ExitCS() {
    Cs_requested = false;
    
    cout << "Process " << world_rank << " sending deferred replies..." << endl;
    
    // Send deferred replies
    for (int j = 0; j < world_size; j++) {
        if (j != world_rank && Rep_deferred[j]) {
            Rep_deferred[j] = false;
            ReplyMessage reply_msg;
            reply_msg.process_id = world_rank;
            MPI_Send(&reply_msg, sizeof(ReplyMessage), MPI_BYTE, 
                    j, REP_TAG, MPI_COMM_WORLD);
            cout << "  → Process " << world_rank << " sent deferred REP to Process " 
                 << j << endl;
        }
    }
}

// Background message handler for incoming REQUESTs when not in CS
void handle_background_requests() {
    MPI_Status status;
    int flag;
    
    MPI_Iprobe(MPI_ANY_SOURCE, REQ_TAG, MPI_COMM_WORLD, &flag, &status);
    
    if (flag) {
        RequestMessage req_msg;
        MPI_Recv(&req_msg, sizeof(RequestMessage), MPI_BYTE, 
                status.MPI_SOURCE, REQ_TAG, MPI_COMM_WORLD, &status);
        
        cout << "  ← Process " << world_rank << " received REQ(" 
             << req_msg.timestamp << ", " << req_msg.process_id 
             << ") in background" << endl;
        
        // Calculate priority
        Priority = Cs_requested && 
                  ((req_msg.timestamp > Ts_current) || 
                   ((req_msg.timestamp == Ts_current) && 
                    (world_rank < req_msg.process_id)));
        
        if (Priority) {
            // Defer the reply
            Rep_deferred[req_msg.process_id] = true;
            cout << "    Process " << world_rank << " DEFERRED reply to Process " 
                 << req_msg.process_id << endl;
        } else {
            // Send immediate reply
            Rep_deferred[req_msg.process_id] = false;
            ReplyMessage reply_msg;
            reply_msg.process_id = world_rank;
            MPI_Send(&reply_msg, sizeof(ReplyMessage), MPI_BYTE, 
                    req_msg.process_id, REP_TAG, MPI_COMM_WORLD);
            cout << "    Process " << world_rank << " sent immediate REP to Process " 
                 << req_msg.process_id << endl;
        }
        
        update_lamport_clock(req_msg.timestamp);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    if (world_size < 2) {
        if (world_rank == 0) {
            cerr << "Need at least 2 processes for mutual exclusion." << endl;
        }
        MPI_Finalize();
        return 0;
    }
    
    // Initialize
    Rep_deferred.resize(world_size, false);
    srand(time(NULL) + world_rank); // Different seed for each process
    
    cout << "\n╔═══════════════════════════════════════════════════╗" << endl;
    cout << "║  Ricart-Agrawala Mutual Exclusion Algorithm     ║" << endl;
    cout << "║  Process " << world_rank << " of " << world_size 
         << " started                              ║" << endl;
    cout << "╚═══════════════════════════════════════════════════╝\n" << endl;
    
    // Each process will try to enter CS multiple times
    for (int execution = 0; execution < MAX_CS_EXECUTIONS; execution++) {
        // Random delay before requesting CS
        int delay = rand() % 2000000; // 0-2 seconds
        usleep(delay);
        
        // Handle any background requests during delay
        for (int i = 0; i < 10; i++) {
            handle_background_requests();
            usleep(delay / 10);
        }
        
        // Request Critical Section
        EnterCS();
        
        // Execute Critical Section
        execute_critical_section();
        
        // Exit Critical Section
        ExitCS();
        
        // Small delay after exiting CS
        usleep(200000); // 0.2 seconds
    }
    
    cout << "\n✓ Process " << world_rank << " completed all " 
         << MAX_CS_EXECUTIONS << " CS executions" << endl;
    
    // Wait for all processes to finish
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Final summary
    if (world_rank == 0) {
        cout << "\n╔═══════════════════════════════════════════════════╗" << endl;
        cout << "║  All processes completed mutual exclusion test   ║" << endl;
        cout << "║  Total CS executions: " << (world_size * MAX_CS_EXECUTIONS) 
             << "                           ║" << endl;
        cout << "╚═══════════════════════════════════════════════════╝\n" << endl;
    }
    
    MPI_Finalize();
    return 0;
}