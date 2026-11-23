// Shapes of the backend responses (see echostream/serving/api.py).

export type Timings = Partial<
  Record<
    "features" | "user_vector" | "recall_local" | "recall_ann_wait" | "recall" | "merge" | "rank" | "mmr" | "total",
    number
  >
>;

export interface Attempt {
  replica: number;
  status: string; // "ok" | "timeout" | "error" | ...
  ms: number;
}

export interface ShardCall {
  shard: number;
  ok: boolean;
  replica: number | null;
  ms: number;
  attempts: Attempt[];
}

export interface Item {
  item_id: number;
  title: string;
  category: string;
  source: string;
  sources: Record<string, number>;
  recall_score: number;
  rank_score: number;
}

export interface Recommendation {
  user_id: number;
  model_version: string;
  served_by: string;
  items: Item[];
  timings_ms: Timings;
  shards: ShardCall[];
  degraded: string[];
}

export interface ShardReplica {
  shard: number;
  replica: number;
  up: boolean;
  ejected?: boolean;
  delay_ms?: number;
  versions?: string[];
  n_items?: Record<string, number>;
  pid?: number;
  uptime_s?: number;
  url?: string;
}

export interface Partition {
  partition: number;
  length?: number;
  pending?: number;
  lag?: number | null;
  owner?: string | null;
}

export interface ClusterState {
  shards: ShardReplica[][];
  feature_nodes: Record<string, boolean>;
  events: {
    up: boolean;
    workers_alive?: string[];
    dlq_length?: number;
    partitions?: Partition[];
    workers?: Record<string, Record<string, string>>;
  };
  api_replica: string;
}

export interface Proc {
  pid?: number | null;
  group?: string;
  port?: number | null;
  running?: boolean;
  restarts?: number;
  held_down?: boolean;
  frozen?: boolean;
  uptime_s?: number | null;
}

export type SupervisorStatus = Record<string, Proc>;

/** A measured value with a bootstrap 95% confidence interval. */
export interface CI {
  mean: number;
  ci95?: [number, number];
}

export interface JourneyItem {
  stage: number;
  title: string;
  zh?: string;
  done: boolean;
  result?: string;
}
