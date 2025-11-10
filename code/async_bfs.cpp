#include <iostream>
#include <vector>
#include <algorithm>
#include <set>
#include <mpi.h>
#include <unistd.h>

using namespace std;

// Message Tags
const int EXPLORE_TAG = 10;
const int ACCEPT_TAG = 11;
const int REJECT_TAG = 12;
const int LEVEL_COMPLETE_TAG = 13;
const int PROCEED_TAG = 14;
const int TERMINATE_TAG = 15;

const int ROOT_RANK = 0;
const int MAX_CHILDREN = 100;

// Message structures
struct ExploreMessage {
    int sender_id;
    int sender_level;
};

struct AcceptMessage {
    int sender_id;
    int sender_level;
};

struct RejectMessage {
    int sender_id;
};

struct LevelCompleteMessage {
    int sender_id;
    int sender_level;
    int num_children;
    int children[MAX_CHILDREN];
};

struct ProceedMessage {
    int level;
};

// Graph topology
vector<vector<int>> get_graph_topology(int world_size) {
    if (world_size == 4) {
        return {
            {1, 3},
            {0, 2},
            {1, 3},
            {0, 2}
        };
    }
    if (world_size > 1) {
        vector<vector<int>> adj(world_size);
        for (int i = 0; i < world_size; ++i) {
            if (i > 0) adj[i].push_back(i - 1);
            if (i < world_size - 1) adj[i].push_back(i + 1);
        }
        return adj;
    }
    return vector<vector<int>>(world_size);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < 2) {
        if (world_rank == 0) {
            cerr << "Need at least 2 processes for this simulation." << endl;
        }
        MPI_Finalize();
        return 0;
    }

    // Get topology
    const vector<vector<int>> adjacency_list = get_graph_topology(world_size);
    const vector<int>& neighbors = adjacency_list[world_rank];

    // Node state
    int parent = -2; // -1 for root, -2 for unassigned
    int level = -1;  // -1 for unassigned
    vector<int> children;
    set<int> pending_neighbors;
    bool explored = false;
    bool received_proceed = false;

    // Root-specific state
    vector<set<int>> nodes_at_level;
    vector<int> completed_at_level;
    if (world_rank == ROOT_RANK) {
        nodes_at_level.resize(world_size);
        completed_at_level.resize(world_size, 0);
        nodes_at_level[0].insert(ROOT_RANK);
        parent = -1;
        level = 0;
    }

    cout << "Rank " << world_rank << ": Starting BFS tree algorithm with " 
         << neighbors.size() << " neighbors." << endl;

    // ==================== ROOT INITIALIZATION ====================
    if (world_rank == ROOT_RANK) {
        cout << "\n=== ROOT " << world_rank << ": Initiating BFS construction ===" << endl;
        
        // Send EXPLORE to all neighbors
        ExploreMessage explore_msg = {ROOT_RANK, 0};
        for (int neighbor : neighbors) {
            MPI_Send(&explore_msg, sizeof(ExploreMessage), MPI_BYTE, 
                    neighbor, EXPLORE_TAG, MPI_COMM_WORLD);
            pending_neighbors.insert(neighbor);
            cout << "ROOT: Sent EXPLORE(0, 0) to neighbor " << neighbor << endl;
        }

        if (pending_neighbors.empty()) {
            explored = true;
            cout << "ROOT: No neighbors, sending LEVEL_COMPLETE to self" << endl;
        }
    }

    // ==================== MAIN MESSAGE LOOP ====================
    bool algorithm_running = true;
    MPI_Status status;

    while (algorithm_running) {
        usleep(50000); // 50ms polling interval

        int flag;
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);

        if (!flag) continue;

        int tag = status.MPI_TAG;
        int source = status.MPI_SOURCE;

        // ==================== HANDLE EXPLORE ====================
        if (tag == EXPLORE_TAG) {
            ExploreMessage msg;
            MPI_Recv(&msg, sizeof(ExploreMessage), MPI_BYTE, source, 
                    EXPLORE_TAG, MPI_COMM_WORLD, &status);
            
            cout << "Rank " << world_rank << ": Received EXPLORE(" 
                 << msg.sender_id << ", " << msg.sender_level << ")" << endl;

            if (world_rank == ROOT_RANK) {
                // Root rejects all EXPLORE
                RejectMessage reject_msg = {world_rank};
                MPI_Send(&reject_msg, sizeof(RejectMessage), MPI_BYTE, 
                        source, REJECT_TAG, MPI_COMM_WORLD);
                cout << "ROOT: Rejected EXPLORE from " << source << endl;
            }
            else if (parent == -2) {
                // First EXPLORE - set parent
                parent = msg.sender_id;
                level = msg.sender_level + 1;
                
                cout << "Rank " << world_rank << ": **Set parent to " << parent 
                     << ", level to " << level << "**" << endl;

                // Send ACCEPT to parent
                AcceptMessage accept_msg = {world_rank, level};
                MPI_Send(&accept_msg, sizeof(AcceptMessage), MPI_BYTE, 
                        parent, ACCEPT_TAG, MPI_COMM_WORLD);
                cout << "Rank " << world_rank << ": Sent ACCEPT(" << world_rank 
                     << ", " << level << ") to parent " << parent << endl;
            }
            else {
                // Already have parent - reject
                RejectMessage reject_msg = {world_rank};
                MPI_Send(&reject_msg, sizeof(RejectMessage), MPI_BYTE, 
                        source, REJECT_TAG, MPI_COMM_WORLD);
                cout << "Rank " << world_rank << ": Rejected EXPLORE from " 
                     << source << " (already have parent)" << endl;
            }
        }

        // ==================== HANDLE ACCEPT ====================
        else if (tag == ACCEPT_TAG) {
            AcceptMessage msg;
            MPI_Recv(&msg, sizeof(AcceptMessage), MPI_BYTE, source, 
                    ACCEPT_TAG, MPI_COMM_WORLD, &status);
            
            cout << "Rank " << world_rank << ": Received ACCEPT(" 
                 << msg.sender_id << ", " << msg.sender_level << ")" << endl;

            children.push_back(msg.sender_id);
            pending_neighbors.erase(msg.sender_id);

            // Root updates nodes_at_level
            if (world_rank == ROOT_RANK) {
                nodes_at_level[msg.sender_level].insert(msg.sender_id);
                cout << "ROOT: Added node " << msg.sender_id << " to level " 
                     << msg.sender_level << endl;
            }

            // Check if exploration complete
            if (pending_neighbors.empty() && received_proceed) {
                explored = true;
                
                // Send LEVEL_COMPLETE
                LevelCompleteMessage lc_msg;
                lc_msg.sender_id = world_rank;
                lc_msg.sender_level = level;
                lc_msg.num_children = children.size();
                for (size_t i = 0; i < children.size(); ++i) {
                    lc_msg.children[i] = children[i];
                }
                
                MPI_Send(&lc_msg, sizeof(LevelCompleteMessage), MPI_BYTE, 
                        ROOT_RANK, LEVEL_COMPLETE_TAG, MPI_COMM_WORLD);
                cout << "Rank " << world_rank << ": Sent LEVEL_COMPLETE(" 
                     << world_rank << ", " << level << ", " << children.size() 
                     << " children) to ROOT" << endl;
            }
        }

        // ==================== HANDLE REJECT ====================
        else if (tag == REJECT_TAG) {
            RejectMessage msg;
            MPI_Recv(&msg, sizeof(RejectMessage), MPI_BYTE, source, 
                    REJECT_TAG, MPI_COMM_WORLD, &status);
            
            cout << "Rank " << world_rank << ": Received REJECT from " 
                 << msg.sender_id << endl;

            pending_neighbors.erase(msg.sender_id);

            // Check if exploration complete
            if (pending_neighbors.empty() && received_proceed) {
                explored = true;
                
                // Send LEVEL_COMPLETE
                LevelCompleteMessage lc_msg;
                lc_msg.sender_id = world_rank;
                lc_msg.sender_level = level;
                lc_msg.num_children = children.size();
                for (size_t i = 0; i < children.size(); ++i) {
                    lc_msg.children[i] = children[i];
                }
                
                MPI_Send(&lc_msg, sizeof(LevelCompleteMessage), MPI_BYTE, 
                        ROOT_RANK, LEVEL_COMPLETE_TAG, MPI_COMM_WORLD);
                cout << "Rank " << world_rank << ": Sent LEVEL_COMPLETE(" 
                     << world_rank << ", " << level << ", " << children.size() 
                     << " children) to ROOT" << endl;
            }
        }

        // ==================== HANDLE PROCEED ====================
        else if (tag == PROCEED_TAG) {
            ProceedMessage msg;
            MPI_Recv(&msg, sizeof(ProceedMessage), MPI_BYTE, source, 
                    PROCEED_TAG, MPI_COMM_WORLD, &status);
            
            cout << "Rank " << world_rank << ": Received PROCEED_NEXT_LEVEL(" 
                 << msg.level << ")" << endl;

            received_proceed = true;

            // Send EXPLORE to all neighbors except parent
            ExploreMessage explore_msg = {world_rank, level};
            for (int neighbor : neighbors) {
                if (neighbor != parent) {
                    MPI_Send(&explore_msg, sizeof(ExploreMessage), MPI_BYTE, 
                            neighbor, EXPLORE_TAG, MPI_COMM_WORLD);
                    pending_neighbors.insert(neighbor);
                    cout << "Rank " << world_rank << ": Sent EXPLORE(" 
                         << world_rank << ", " << level << ") to neighbor " 
                         << neighbor << endl;
                }
            }

            // Check if no neighbors to explore
            if (pending_neighbors.empty()) {
                explored = true;
                
                // Send LEVEL_COMPLETE
                LevelCompleteMessage lc_msg;
                lc_msg.sender_id = world_rank;
                lc_msg.sender_level = level;
                lc_msg.num_children = 0;
                
                MPI_Send(&lc_msg, sizeof(LevelCompleteMessage), MPI_BYTE, 
                        ROOT_RANK, LEVEL_COMPLETE_TAG, MPI_COMM_WORLD);
                cout << "Rank " << world_rank 
                     << ": No neighbors to explore, sent LEVEL_COMPLETE to ROOT" 
                     << endl;
            }
        }

        // ==================== HANDLE LEVEL_COMPLETE (ROOT ONLY) ====================
        else if (tag == LEVEL_COMPLETE_TAG && world_rank == ROOT_RANK) {
            LevelCompleteMessage msg;
            MPI_Recv(&msg, sizeof(LevelCompleteMessage), MPI_BYTE, source, 
                    LEVEL_COMPLETE_TAG, MPI_COMM_WORLD, &status);
            
            cout << "\nROOT: Received LEVEL_COMPLETE(" << msg.sender_id 
                 << ", " << msg.sender_level << ", " << msg.num_children 
                 << " children)" << endl;

            // Add children to next level
            for (int i = 0; i < msg.num_children; ++i) {
                int child = msg.children[i];
                nodes_at_level[msg.sender_level + 1].insert(child);
                cout << "ROOT: Added node " << child << " to level " 
                     << (msg.sender_level + 1) << endl;
            }

            // Increment completion count
            completed_at_level[msg.sender_level]++;
            
            cout << "ROOT: Level " << msg.sender_level << " completion: " 
                 << completed_at_level[msg.sender_level] << "/" 
                 << nodes_at_level[msg.sender_level].size() << endl;

            // Check if level is complete
            if (completed_at_level[msg.sender_level] == 
                (int)nodes_at_level[msg.sender_level].size()) {
                
                cout << "\n*** ROOT: Level " << msg.sender_level 
                     << " COMPLETE! ***" << endl;

                int next_level = msg.sender_level + 1;
                
                if (!nodes_at_level[next_level].empty()) {
                    // Send PROCEED to next level
                    cout << "ROOT: Sending PROCEED(" << next_level 
                         << ") to " << nodes_at_level[next_level].size() 
                         << " nodes" << endl;
                    
                    ProceedMessage proceed_msg = {next_level};
                    for (int node : nodes_at_level[next_level]) {
                        MPI_Send(&proceed_msg, sizeof(ProceedMessage), MPI_BYTE, 
                                node, PROCEED_TAG, MPI_COMM_WORLD);
                        cout << "ROOT: Sent PROCEED to node " << node << endl;
                    }
                } else {
                    // Algorithm complete
                    cout << "\n*** ROOT: BFS TREE CONSTRUCTION COMPLETE! ***\n" 
                         << endl;
                    
                    // Send TERMINATE to all nodes
                    for (int i = 1; i < world_size; ++i) {
                        MPI_Send(NULL, 0, MPI_BYTE, i, TERMINATE_TAG, 
                                MPI_COMM_WORLD);
                    }
                    algorithm_running = false;
                }
            }
        }

        // ==================== HANDLE TERMINATE ====================
        else if (tag == TERMINATE_TAG) {
            MPI_Recv(NULL, 0, MPI_BYTE, ROOT_RANK, TERMINATE_TAG, 
                    MPI_COMM_WORLD, &status);
            cout << "Rank " << world_rank << ": Received TERMINATE" << endl;
            algorithm_running = false;
        }
    }

    // ==================== FINAL SYNCHRONIZATION ====================
    MPI_Barrier(MPI_COMM_WORLD);

    // ==================== OUTPUT RESULTS ====================
    cout << "\n========================================" << endl;
    cout << "Rank " << world_rank << " - FINAL BFS TREE RESULT" << endl;
    cout << "========================================" << endl;
    cout << "Level: " << level << endl;
    cout << "Parent: " << (parent == -1 ? "ROOT" : to_string(parent)) << endl;
    cout << "Children (" << children.size() << "): ";
    if (children.empty()) {
        cout << "None";
    } else {
        for (size_t i = 0; i < children.size(); ++i) {
            cout << children[i];
            if (i < children.size() - 1) cout << ", ";
        }
    }
    cout << endl;
    cout << "========================================\n" << endl;

    // Root prints complete tree structure
    if (world_rank == ROOT_RANK) {
        cout << "\n=======================================" << endl;
        cout << "ROOT: COMPLETE BFS TREE STRUCTURE" << endl;
        cout << "=======================================" << endl;
        for (size_t l = 0; l < nodes_at_level.size(); ++l) {
            if (!nodes_at_level[l].empty()) {
                cout << "Level " << l << ": { ";
                bool first = true;
                for (int node : nodes_at_level[l]) {
                    if (!first) cout << ", ";
                    cout << node;
                    first = false;
                }
                cout << " }" << endl;
            }
        }
        cout << "=======================================" << endl;
    }

    MPI_Finalize();
    return 0;
}