import { z } from "zod";

enum Role {
  Follower = "follower",
  Candidate = "candidate",
  Leader = "leader",
}

type NodeId = string;
type LogEntry = {
  term: number;
  command: string;
};

// Logs Invariants
// 1. If two entries in different logs have the same index and term, then they store the same command.
// 2. If two entries in different logs have the same index and term, then the logs are identical in all preceding entries.

// Raft
// never commits log entries from previous terms by count-
// ing replicas. Only log entries from the leader’s current
// term are committed by counting replicas

type PersistedState = {
  currentTerm: number;
  votedFor: NodeId | null;
  log: LogEntry[];
};

type VolatileState = {
  commitIndex: number;
  lastApplied: number;
};

type LeaderState = {
  nextIndex: Record<NodeId, number>;
  matchIndex: Record<NodeId, number>;
};

type AppendEntriesRequest = {
  term: number;
  leaderId: NodeId;
  prevLogIndex: number;
  prevLogTerm: number;
  entries: LogEntry[];
  leaderCommit: number;
};

type AppendEntriesResponse = {
  term: number;
  success: boolean;
};

type RequestVoteRequest = {
  term: number;
  candidateId: NodeId;
  lastLogIndex: number;
  lastLogTerm: number;
};

type RequestVoteResponse = {
  term: number;
  voteGranted: boolean;
};

class RaftNode {
  host: string;
  port: number;
  id: NodeId;
  constructor(id: NodeId, host: string, port: number) {
    this.host = host;
    this.port = port;
    this.id = id;
  }

  async requestVote(request: RequestVoteRequest): Promise<RequestVoteResponse> {
    const res = await fetch(`http://${this.host}:${this.port}/request-vote`, {
      method: "POST",
      body: JSON.stringify(request),
    });

    return res.json() as unknown as RequestVoteResponse;
  }

  async appendEntries(
    request: AppendEntriesRequest
  ): Promise<AppendEntriesResponse> {
    const res = await fetch(`http://${this.host}:${this.port}/append-entries`, {
      method: "POST",
      body: JSON.stringify(request),
    });

    return res.json() as unknown as AppendEntriesResponse;
  }
}

type RaftCluster = RaftNode[];

function getElectionTimeout() {
  return Math.floor(Math.random() * 150) + 150;
}

class RaftRpcServer {
  private rpc_append_entries: (
    req: AppendEntriesRequest
  ) => AppendEntriesResponse;
  private rpc_request_vote: (req: RequestVoteRequest) => RequestVoteResponse;

  constructor(
    append_entries: (req: AppendEntriesRequest) => AppendEntriesResponse,
    request_vote: (req: RequestVoteRequest) => RequestVoteResponse
  ) {
    this.rpc_append_entries = append_entries;
    this.rpc_request_vote = request_vote;
  }

  run(port: number) {
    Bun.serve({
      routes: {
        "/append-entries": (req) => {
          const b = req.json() as unknown as AppendEntriesRequest;
          return Response.json(this.rpc_append_entries(b));
        },
        "/request-vote": (req) => {
          const b = req.json() as unknown as RequestVoteRequest;
          return Response.json(this.rpc_request_vote(b));
        },
      },

      fetch(req) {
        return new Response("404!");
      },

      port: port,
    });
  }
}

class RaftServer {
  private node: RaftNode;
  private peers: RaftCluster;

  private persistedState: PersistedState;
  private volatileState: VolatileState;
  private leaderState: LeaderState;

  private server: RaftRpcServer;
  private role: Role = Role.Follower;

  private heartbeatInterval: Timer | null = null;
  private electionTimeout: Timer | null = null;

  constructor(node: RaftNode, cluster: RaftCluster) {
    this.node = node;
    this.peers = cluster.filter((n) => n.id !== node.id);

    this.persistedState = {
      currentTerm: 1,
      votedFor: null,
      log: [{ term: 0, command: "init" }],
    };

    this.volatileState = {
      commitIndex: 1,
      lastApplied: 1,
    };
  }

  run() {
    this.server = new RaftRpcServer(
      (req) => {
        throw new Error("Not implemented");
      },
      (req) => {
        throw new Error("Not implemented");
      }
    );

    this.server.run(this.node.port);

    setTimeout(() => {
      this.role = Role.Candidate;
      this.persistedState.currentTerm += 1;
      this.broadcastRequestVote();
    }, getElectionTimeout());
  }

  private async broadcastRequestVote() {
    //if no win set another timeout for election staled
    const request: RequestVoteRequest = {
      term: this.persistedState.currentTerm,
      candidateId: this.node.id,
      lastLogIndex: this.persistedState.log.length,
      lastLogTerm:
        this.persistedState.log[this.persistedState.log.length - 1].term,
    };

    const votes = await Promise.all(
      this.peers.map((peer) => peer.requestVote(request))
    );

    if (this.role !== Role.Candidate) {
      // some other node has become leader
      return;
    }

    const votesGranted = votes.filter((v) => v).length;
    if (votesGranted > this.peers.length / 2) {
      this.role = Role.Leader;

      this.leaderState = {
        nextIndex: Object.fromEntries(
          // initialized to leader last log index + 1
          this.peers.map((p) => [p.id, this.persistedState.log.length])
        ),
        // initialized to 0
        matchIndex: Object.fromEntries(this.peers.map((p) => [p.id, 0])),
      };

      this.startHeartbeats();
    }
  }

  private startHeartbeats() {
    this.heartbeatInterval = setInterval(() => {
      this.broadcastAppendEntries();
    }, 50);
  }

  private async broadcastAppendEntries() {
    const responses = await Promise.all(
      this.peers.map((peer) => {
        const nextIndex = this.leaderState.nextIndex[peer.id];
        const entries = this.persistedState.log.slice(nextIndex);

        const request: AppendEntriesRequest = {
          term: this.persistedState.currentTerm,
          leaderId: this.node.id,
          prevLogIndex: nextIndex - 1 || 0,
          prevLogTerm: this.persistedState.log[nextIndex - 1].term,
          entries: entries,
          leaderCommit: this.volatileState.commitIndex,
        };

        return peer.appendEntries(request);
      })
    );

    if (this.role !== Role.Leader) {
      // some other node has become leader
      return;
    }

    const successCount = responses.filter((r) => r.success).length;
    if (successCount > this.peers.length / 2) {
      // commit entries
    }
  }
}

const cluster = [
  new RaftNode("node1", "localhost", 3000),
  new RaftNode("node2", "localhost", 3001),
  new RaftNode("node3", "localhost", 3002),
];

for (const node of cluster) {
  const server = new RaftServer(node, cluster);
  server.run();
}
