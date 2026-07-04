# Vector Databases: Internals and Implementation Guide

## Table of Contents

1. [What is a Vector Database?](#1-what-is-a-vector-database)
2. [Core Concepts](#2-core-concepts)
3. [Distance Metrics](#3-distance-metrics)
4. [Indexing Structures](#4-indexing-structures)
5. [Approximate Nearest Neighbor Search](#5-approximate-nearest-neighbor-search)
6. [Storage and Persistence](#6-storage-and-persistence)
7. [Filtering](#7-filtering)
8. [Distributed Architecture](#8-distributed-architecture)
9. [Components to Build From Scratch](#9-components-to-build-from-scratch)
10. [Trade-offs and Design Decisions](#10-trade-offs-and-design-decisions)
11. [Real-World Systems](#11-real-world-systems)

---

## 1. What is a Vector Database?

A vector database is a storage and retrieval system optimized for **high-dimensional floating-point vectors** and **similarity search** — finding the vectors most similar to a query vector, not exact matches.

### Why it exists

Traditional databases answer questions like:
- "Give me all rows where `user_id = 42`"
- "Give me all products where `price < 100`"

These are **exact match** or **range** queries over scalar values. They don't answer:
- "Give me the 10 documents most semantically similar to this query"
- "Give me the 5 images that look most like this photo"

That's the gap vector databases fill. The "data" is an **embedding** — a dense numeric representation of meaning, produced by a neural network.

### How it differs from traditional databases

| Property | Relational DB | Vector DB |
|---|---|---|
| Query type | Exact / range | Similarity (nearest neighbor) |
| Index structure | B-tree, hash | HNSW, IVF, LSH |
| Data | Scalars, strings | Float32/Float16 arrays |
| Result | Exact matches | Approximate top-K matches |
| Dimensionality | Low (1–10s) | High (128–65536+) |
| Curse of dimensionality | Not relevant | Central design constraint |

---

## 2. Core Concepts

### Embeddings

An **embedding** is a vector `[f₀, f₁, ..., fₙ₋₁]` of `n` floats produced by encoding raw data (text, image, audio) through a neural model. The key property: **similar inputs produce geometrically close vectors** in the high-dimensional space.

```
"cat"  → [0.21, -0.54, 0.88, ...]  (n-dimensional)
"dog"  → [0.19, -0.51, 0.90, ...]  ← close to "cat"
"SQL"  → [-0.73, 0.12, -0.33, ...]  ← far from "cat"
```

Common embedding dimensions:
- `all-MiniLM-L6-v2`: 384
- OpenAI `text-embedding-3-small`: 1536
- OpenAI `text-embedding-3-large`: 3072
- CLIP (images): 512–1024

### Vector Space

All vectors live in **ℝⁿ** (n-dimensional real space). Similarity is measured by geometric proximity. This is where the entire challenge lies — in high dimensions, naively scanning all vectors is O(n·d) per query, where n = number of vectors, d = dimension. At 10M vectors × 1536 dims, brute force is infeasible for low-latency use cases.

### The Curse of Dimensionality

As dimensionality increases:
- Distance between nearest and farthest neighbors converges — everything becomes "equally far"
- Space becomes exponentially sparse
- Tree-based indexes (KD-tree) degrade to O(n) at ~20+ dimensions

This is why specialized ANN (Approximate Nearest Neighbor) algorithms exist.

---

## 3. Distance Metrics

The choice of metric must match how the embedding model was trained.

### Cosine Similarity

Measures the angle between two vectors, ignoring magnitude.

```
cos_sim(A, B) = (A · B) / (|A| × |B|)

distance = 1 - cos_sim(A, B)     range: [0, 2]
```

- Best for: text embeddings (sentence transformers, OpenAI embeddings)
- Invariant to vector scale — only direction matters
- Can be converted to dot product if vectors are L2-normalized

### Euclidean Distance (L2)

Straight-line distance in space.

```
L2(A, B) = sqrt(Σ (Aᵢ - Bᵢ)²)
```

- Best for: image embeddings, some audio models
- Scale-sensitive — normalize inputs if needed

### Dot Product (Inner Product)

```
dot(A, B) = Σ (Aᵢ × Bᵢ)
```

- Best for: models trained with dot product (e.g., DPR, ColBERT)
- Not a true metric (not symmetric w.r.t. similarity)
- Faster to compute than cosine (no normalization step)

### Manhattan Distance (L1)

```
L1(A, B) = Σ |Aᵢ - Bᵢ|
```

- Rarely used in practice for high-dimensional embeddings
- Less sensitive to outlier dimensions than L2

### Comparison Table

| Metric | Formula | Use case | SIMD-friendly |
|---|---|---|---|
| Cosine | angle | Text, NLP | Yes (with normalization) |
| L2 | straight line | Images, general | Yes |
| Dot product | projection | Recommendation, DPR | Yes |
| L1 | city block | Rare | Moderate |

---

## 4. Indexing Structures

The index is the core of a vector database. It trades **recall** (finding the true nearest neighbors) for **speed** and **memory**.

### 4.1 Flat Index (Brute Force)

No index structure. Compare query against every vector.

```
for each vector v in database:
    dist = distance(query, v)
    update top-K heap
return top-K
```

- **Recall**: 100% (exact)
- **Query time**: O(n · d)
- **Build time**: O(n) — just load data
- **Memory**: O(n · d · sizeof(float32))
- **Use case**: Small datasets (<100K), ground-truth benchmarking
- **Implementation**: FAISS `IndexFlat`

### 4.2 IVF — Inverted File Index

Partition vectors into `nlist` Voronoi cells using k-means. At query time, search only `nprobe` nearest cells.

```
BUILD:
  1. Run k-means on training vectors → nlist centroids
  2. Assign each vector to its nearest centroid
  3. Store each vector in the corresponding inverted list

QUERY:
  1. Find nprobe nearest centroids to query
  2. Scan only vectors in those lists
  3. Return top-K from scanned set
```

```
         Centroid₁   Centroid₂   Centroid₃
              │            │            │
         [v₁,v₄,v₇]  [v₂,v₅]   [v₃,v₆,v₈]

Query → closest 2 centroids → scan ~2/nlist of the data
```

- **Recall**: ~95–99% (tunable via `nprobe`)
- **Query time**: O(nprobe/nlist · n · d)
- **Build time**: O(n · d · k-means iterations)
- **Trade-off**: Higher `nprobe` = better recall, slower query
- **Typical `nlist`**: `4·sqrt(n)`

### 4.3 HNSW — Hierarchical Navigable Small World

The dominant algorithm in production systems. Builds a multi-layer proximity graph where upper layers are sparse (long-range links) and lower layers are dense (local links).

```
Layer 2 (sparse):  1 ── 5
                   │
Layer 1:           1 ── 3 ── 5 ── 8
                   │         │
Layer 0 (dense):   1─2─3─4─5─6─7─8─9
```

**Insertion:**
```
INSERT(new_vector q):
  1. Pick random max layer L from exponential distribution
  2. Start at entry point at top layer
  3. Greedy search downward to layer L+1
  4. At each layer L down to 0:
     a. Find ef_construction nearest neighbors
     b. Connect q to M nearest (bidirectional edges)
     c. Prune connections exceeding Mmax
```

**Query:**
```
SEARCH(query q, k):
  1. Start at entry point (top layer)
  2. Greedy descent: at each layer, greedily move to nearest neighbor
  3. At layer 0: beam search with ef candidates
  4. Return top-k from candidates
```

**Key parameters:**
- `M` — number of bidirectional connections per node (typical: 16–64). Higher = better recall, more memory/build time
- `ef_construction` — size of dynamic candidate list during build. Higher = better graph quality, slower build
- `ef_search` — candidate list size at query time. Higher = better recall, slower query

**Complexity:**
- Build: O(n · log(n) · M)
- Query: O(log(n) · ef · M)
- Memory: O(n · M · 2 · sizeof(int) + n · d · sizeof(float32))

**Why it dominates:**
- Excellent recall/speed trade-off
- No training data needed (unlike IVF)
- Handles updates better than IVF
- Used by: Qdrant, Weaviate, Milvus, pgvector

### 4.4 LSH — Locality-Sensitive Hashing

Hash vectors so that similar vectors collide with high probability.

```
For cosine similarity (random projection LSH):
  1. Generate k random hyperplanes h₁...hₖ
  2. Hash(v) = [sign(v·h₁), sign(v·h₂), ..., sign(v·hₖ)]
  3. Vectors with same hash → same bucket
  4. Query: hash query, check bucket(s)
```

- Fast build, low memory
- Lower recall than HNSW at the same speed
- Mostly superseded by HNSW in practice
- Still useful for streaming/online settings

### 4.5 Product Quantization (PQ)

Compression technique, usually combined with IVF (IVF-PQ).

```
1. Split d-dimensional vector into m sub-vectors of d/m dims each
2. Train k-means (k=256) on each sub-space → codebooks
3. Encode each vector as m 8-bit codes (1 byte each)
4. Compressed size: m bytes instead of d×4 bytes

At query time:
1. Precompute distance from query to all k centroids in each sub-space
2. For each stored vector: sum m table lookups → approximate distance
```

- **Compression ratio**: `m / (d×4)` — e.g., 32x compression for d=1536, m=48
- **Trade-off**: Lossy compression → lower recall
- **Use case**: Billion-scale datasets where memory is the bottleneck
- Combined as `IVF-PQ` in FAISS

### 4.6 ScaNN (Scalable Nearest Neighbors)

Google's algorithm. Uses anisotropic quantization — quantization error is weighted by direction (errors along the query direction matter more than perpendicular errors).

- State of the art on ANN benchmarks
- Harder to implement; not widely available outside Google

### Index Comparison Summary

| Index | Recall | Query Speed | Build Time | Memory | Updates | Use case |
|---|---|---|---|---|---|---|
| Flat | 100% | Slowest | O(1) | Baseline | Trivial | Small datasets |
| IVF | High | Fast | Medium | Baseline | Costly | Large, static |
| HNSW | Very high | Very fast | Slow | 1.5–3× | Moderate | Production default |
| LSH | Medium | Fast | Fast | Low | Easy | Streaming |
| IVF-PQ | Medium | Fastest | Medium | 10–100× less | Costly | Billion scale |

---

## 5. Approximate Nearest Neighbor Search

### Why "approximate"?

Exact KNN in high dimensions is O(n·d). ANN algorithms accept small recall loss to achieve sub-linear query time (in practice O(log n) or O(1) amortized).

**Recall@K** = |true top-K ∩ returned top-K| / K

A system with 95% recall@10 means 9.5 of the true top-10 neighbors are returned on average.

### The ANN benchmark

[ann-benchmarks.com](https://ann-benchmarks.com) measures recall vs queries-per-second. HNSW consistently sits on the Pareto frontier across most datasets.

### Beam search in HNSW layer 0

```python
def search_layer(q, entry_points, ef, layer):
    visited = set(entry_points)
    candidates = min_heap(entry_points, key=dist(q))  # to explore
    results   = max_heap(entry_points, key=dist(q))  # best found so far

    while candidates:
        c = candidates.pop_min()
        if dist(q, c) > dist(q, results.peek_max()):
            break  # all unvisited candidates are worse than worst result
        for neighbor in graph[layer][c]:
            if neighbor not in visited:
                visited.add(neighbor)
                if dist(q, neighbor) < dist(q, results.peek_max()) or len(results) < ef:
                    candidates.push(neighbor)
                    results.push(neighbor)
                    if len(results) > ef:
                        results.pop_max()
    return results
```

---

## 6. Storage and Persistence

### Memory Layout

Vectors are stored as flat float32 arrays for SIMD-friendly access:

```
vectors: [f32; n * d]   // row-major: vector i = vectors[i*d .. (i+1)*d]
ids:     [u64; n]        // external IDs mapping index position → user ID
```

### HNSW Graph Storage

```
graph[layer][node_id] = [neighbor_id₁, neighbor_id₂, ...]

Flat representation (cache-friendly):
  offsets: [u64; n+1]          // offsets[i] = start of node i's neighbors
  neighbors: [u32; total_edges] // all neighbor lists concatenated
```

### Write-Ahead Log (WAL)

Before mutating any index structure, write the operation to a durable log:

```
WAL entry format:
  [8 bytes: sequence_number]
  [1 byte:  operation_type]  // INSERT=1, DELETE=2, UPDATE=3
  [8 bytes: vector_id]
  [4 bytes: dimension]
  [d×4 bytes: vector data]
  [4 bytes: metadata_length]
  [n bytes: metadata (JSON/msgpack)]
  [4 bytes: CRC32 checksum]
```

On crash recovery:
1. Load last snapshot
2. Replay WAL entries after snapshot's sequence number
3. Rebuild in-memory index state

### Snapshots

Periodically serialize the full index to disk:

```
snapshot.bin:
  [header: magic, version, n_vectors, dimension, metric]
  [vector store: raw float32 array]
  [id map: vector_index → external_id]
  [hnsw graph: layer count, entry point, adjacency lists]
  [metadata store: offset index + raw bytes]
```

### Storage Backends

| Backend | Pros | Cons |
|---|---|---|
| Memory-mapped files (mmap) | OS-managed paging, fast random access | Not portable across machines |
| RocksDB | Sorted KV, compression, WAL included | Higher latency per lookup |
| Custom segment files | Full control, SIMD-aligned | Complex to implement |
| S3/object store | Infinite scale, cheap | High latency — needs local cache |

---

## 7. Filtering

The hardest unsolved problem in vector databases: combining ANN search with **metadata filters** (e.g., "find top-10 similar documents where `category = 'finance'` and `date > 2024-01-01`").

### Pre-filtering

Filter the dataset first, then run ANN on the filtered subset.

```
filtered_ids = metadata_store.query("category = 'finance'")
results = index.search(query, k=10, allowed_ids=filtered_ids)
```

- **Problem**: If the filter is very selective, the filtered subset is tiny → brute force on a small set (fine). If filter returns millions of IDs, passing them into HNSW is expensive (O(n) per node visit to check membership).
- **Implementation**: Use a bitmap (roaring bitmap) over filtered IDs; check membership O(1) during graph traversal.

### Post-filtering

Run ANN unrestricted, then filter results.

```
candidates = index.search(query, k=k*oversample_factor)
results = [c for c in candidates if metadata_filter(c)][:k]
```

- **Problem**: If the filter is selective, you may not get k results — low recall. Requires oversampling (2–10×), still no guarantee.

### In-graph filtering (HNSW with filter-aware traversal)

Modify the HNSW graph traversal to skip nodes that fail the filter. Continue expanding until k valid results are found.

```python
def filtered_search(q, k, filter_fn, ef=100):
    results = []
    while len(results) < k:
        candidates = hnsw_search(q, ef=ef)
        results = [c for c in candidates if filter_fn(c)]
        ef *= 2  # expand search if not enough results
    return results[:k]
```

- Best recall but slower when filter is very selective
- Used by Qdrant's `payload index` approach

### Filtered HNSW with ACORN (recent research)

Build predicate-aware subgraphs during index construction. Expensive to build but avoids the traversal blowup at query time.

---

## 8. Distributed Architecture

### Sharding Strategies

**Random sharding:**
- Vectors are randomly distributed across N shards
- Query fan-out: send query to all N shards, merge top-K results
- Simple; uniform load; no hot spots
- Every query touches every shard

**Centroid-based sharding (IVF-style):**
- Global k-means → assign each centroid to a shard
- Route queries to shards owning the nearest centroids
- Reduces fan-out; locality-preserving
- Shard imbalance if data is clustered

**Replica sets:**
- Each shard has R replicas for read scaling and fault tolerance
- Leader-follower: writes go to leader, replicated to followers
- Reads served from any replica

```
                    ┌─────────────┐
        Write ──►   │   Leader    │
                    └──────┬──────┘
                           │ replicate
              ┌────────────┴────────────┐
              ▼                         ▼
        ┌──────────┐             ┌──────────┐
        │ Replica1 │             │ Replica2 │
        └──────────┘             └──────────┘
              ▲                         ▲
              └──────── Read ───────────┘
```

### Distributed Query Flow

```
1. Client sends query to coordinator
2. Coordinator fans out to all shards (or relevant shards)
3. Each shard runs local ANN search → returns top-K candidates with distances
4. Coordinator merges K results from each shard → global top-K
5. Return to client
```

Merge is just a K-way merge of sorted lists: O(S · K · log(S)) where S = shards.

### Consistency Model

Most vector DBs use **eventual consistency** for writes:
- Insert → WAL → async index update
- New vectors may not be immediately searchable (propagation delay ~seconds)
- Acceptable for most use cases (recommendation, semantic search)

Strong consistency requires distributed consensus (Raft/Paxos) — expensive for high-throughput writes.

---

## 9. Components to Build From Scratch

Here's a complete map of everything you need to implement a production-grade vector database:

### 9.1 Vector Store

Responsible for storing raw float vectors and mapping them to IDs.

```
VectorStore:
  - insert(id: u64, vector: [f32; d]) → Result
  - get(id: u64) → [f32; d]
  - delete(id: u64) → Result
  - iter() → Iterator<(u64, [f32; d])>
  
Implementation:
  - Memory: Vec<f32> + HashMap<u64, usize>
  - Disk: mmap'd flat file + offset index
  - SIMD-align allocations to 32/64 bytes
```

### 9.2 Distance Engine

SIMD-accelerated distance computation — this is your hottest code path.

```
DistanceEngine:
  - cosine(a: &[f32], b: &[f32]) → f32
  - l2_squared(a: &[f32], b: &[f32]) → f32
  - dot(a: &[f32], b: &[f32]) → f32
  - batch_cosine(query: &[f32], candidates: &[[f32]]) → Vec<f32>

SIMD implementation (AVX2, processes 8 f32s at once):
  - Use _mm256_fmadd_ps for dot product accumulation
  - In Rust: use `wide` crate or `std::simd`; in C: intrinsics or BLAS (cblas_sdot)
```

```c
// Pseudocode for AVX2 dot product
float dot_avx2(float* a, float* b, int d) {
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < d; i += 8) {
        __m256 va = _mm256_load_ps(a + i);
        __m256 vb = _mm256_load_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    // horizontal sum of 8 lanes
    return hsum_avx(sum);
}
```

### 9.3 HNSW Index

The graph index — the largest and most complex component.

```
HNSWIndex:
  - build(vectors: &[(u64, Vec<f32>)]) → Result       // batch build
  - insert(id: u64, vector: &[f32]) → Result          // online insert
  - delete(id: u64) → Result                          // mark deleted (lazy)
  - search(query: &[f32], k: usize, ef: usize) → Vec<(u64, f32)>
  - search_filtered(query, k, ef, filter) → Vec<(u64, f32)>
  - serialize() → Bytes
  - deserialize(bytes: &[u8]) → Self

Internal state:
  - entry_point: u32
  - max_layer: u8
  - graph: Vec<Vec<Vec<u32>>>   // graph[layer][node] = neighbor_ids
  - params: HNSWParams { M, ef_construction, level_mult }
  - deleted: HashSet<u32>       // soft deletes
```

**Level assignment for new nodes:**
```python
def random_level(M, ml=1/ln(M)):
    # exponential distribution — most nodes at layer 0
    return floor(-ln(random()) * ml)
```

### 9.4 Metadata Store

Stores arbitrary key-value metadata attached to each vector, supports filtered queries.

```
MetadataStore:
  - put(id: u64, metadata: Map<String, Value>) → Result
  - get(id: u64) → Map<String, Value>
  - query(filter: Filter) → RoaringBitmap   // set of matching IDs
  - delete(id: u64) → Result

Filter types:
  - Eq(field, value)
  - Range(field, min, max)
  - In(field, [value1, value2, ...])
  - And(filter1, filter2)
  - Or(filter1, filter2)
  - Not(filter)

Index per field for fast lookups:
  - String fields: inverted index (HashMap<String, RoaringBitmap>)
  - Numeric fields: BTree<f64, RoaringBitmap> for range queries
```

### 9.5 WAL and Recovery

```
WAL:
  - append(entry: WALEntry) → SequenceNumber
  - flush() → Result                          // fsync to disk
  - iter_from(seq: SequenceNumber) → Iterator<WALEntry>
  - truncate_before(seq: SequenceNumber)      // after snapshot

Recovery:
  - load_snapshot() → (Index, SequenceNumber)
  - replay_wal(from_seq) → apply entries to index
```

### 9.6 Query Engine

Parses and executes search requests.

```
QueryEngine:
  - search(req: SearchRequest) → SearchResponse

SearchRequest:
  - vector: Vec<f32>
  - k: usize
  - ef: Option<usize>        // override ef_search
  - filter: Option<Filter>
  - include_metadata: bool
  - include_vector: bool

SearchResponse:
  - results: Vec<SearchResult>

SearchResult:
  - id: u64
  - score: f32
  - metadata: Option<Map>
  - vector: Option<Vec<f32>>
```

### 9.7 CRUD Layer

Orchestrates all writes through WAL → index → metadata atomically.

```
fn insert(id, vector, metadata):
  seq = wal.append(Insert(id, vector, metadata))
  wal.flush()
  vector_store.insert(id, vector)
  hnsw.insert(id, &vector)
  metadata_store.put(id, metadata)

fn delete(id):
  seq = wal.append(Delete(id))
  wal.flush()
  vector_store.delete(id)
  hnsw.delete(id)      // soft delete — mark in deleted set
  metadata_store.delete(id)

fn upsert(id, vector, metadata):
  if exists(id): delete(id)
  insert(id, vector, metadata)
```

### 9.8 API Layer

Expose the database over HTTP (REST/gRPC).

**REST endpoints:**

```
POST   /collections                    # create a collection
DELETE /collections/{name}             # drop a collection

POST   /collections/{name}/points      # upsert vectors
DELETE /collections/{name}/points      # delete by IDs

POST   /collections/{name}/points/search  # ANN search
POST   /collections/{name}/points/get     # get by IDs
```

**gRPC** is preferred for high-throughput clients (used by Qdrant, Weaviate).

### 9.9 Snapshot / Compaction

```
Snapshotter:
  - create_snapshot() → SnapshotId
    1. Freeze writes (briefly) or use copy-on-write
    2. Serialize vector store + hnsw graph + metadata to disk
    3. Record snapshot sequence number
    4. Truncate WAL entries before that sequence number

Compaction:
  - Periodically remove soft-deleted nodes from HNSW
    (requires full graph rebuild or incremental cleanup)
  - Merge small WAL segments
```

### 9.10 Collection Manager

Top-level abstraction — one index per collection (like a table).

```
CollectionManager:
  - create(name, config: CollectionConfig) → Result
  - get(name) → &Collection
  - drop(name) → Result
  - list() → Vec<CollectionInfo>

CollectionConfig:
  - dimension: usize
  - metric: DistanceMetric
  - hnsw_params: HNSWParams
  - quantization: Option<QuantizationConfig>
  - replication_factor: usize
```

---

## 10. Trade-offs and Design Decisions

### Recall vs Latency

The fundamental knob is `ef_search` (HNSW) or `nprobe` (IVF):

```
ef_search=10  →  ~90% recall, 0.5ms
ef_search=50  →  ~97% recall, 1.2ms
ef_search=200 →  ~99% recall, 4.0ms
ef_search=∞   → 100% recall, 40ms (brute force)
```

Expose this as a per-query tunable — different applications have different tolerance.

### Index Build Time vs Query Quality

HNSW with high `ef_construction` (400+) builds a better graph but takes 3–5× longer to build vs `ef_construction=100`. For most use cases, `ef_construction=200, M=16` is a safe default.

### Memory vs Throughput

Keeping vectors in RAM is fast but expensive. Options:
- **Pure in-memory**: <100M × 1536-dim vectors feasible on 256GB RAM machines
- **Memory-mapped**: OS pages in/out; p99 latency spikes on page faults
- **PQ compression**: 10–32× compression, ~5% recall loss
- **On-disk HNSW (DiskANN)**: graph on SSD, only top layers in memory

### Update-Friendliness

- **IVF**: Insertions require reassigning to nearest centroid (cheap), but the centroid assignments drift over time → periodic re-indexing needed
- **HNSW**: Insertions are incremental and cheap; deletions are soft (lazy) — dead nodes accumulate. Periodic compaction needed.
- **LSH**: Trivial updates — just rehash

### Quantization Decision

Use quantization when:
- Dataset > 10M vectors AND memory is the bottleneck
- Can accept 3–7% recall degradation

Skip quantization when:
- Dataset fits in RAM
- High-precision recall is critical (medical, legal search)

---

## 11. Real-World Systems

### FAISS (Facebook AI Similarity Search)

- C++ library (not a full database)
- Implements: Flat, IVF, HNSW, PQ, ScaNN-like variants
- Used as the indexing backend by many DBs
- GPU acceleration support
- No persistence, no API, no metadata — pure index library

### Qdrant

- Written in Rust
- HNSW + payload (metadata) filtering with SIMD distance
- On-disk HNSW option (mmap)
- gRPC + REST API
- Strong filtering via bitmap-based payload indexes
- Best-in-class filtered search performance

### Weaviate

- Written in Go
- HNSW index with module system (text2vec, img2vec built-in)
- GraphQL + REST API
- BM25 hybrid search (vector + keyword)
- Schema-based (like a document DB + vector DB)

### Chroma

- Written in Python + DuckDB
- Designed for ease of use and local development
- Not production-scale (no distributed mode)
- Great for prototyping RAG pipelines

### Pinecone

- Fully managed, proprietary
- Custom index algorithm (not public)
- Serverless tier with pod-based scaling
- Best developer experience; no ops burden
- Expensive at scale

### pgvector

- PostgreSQL extension
- HNSW and IVF indexes
- Full SQL support — joins, transactions, ACID
- Best when you already use Postgres and need <10M vectors
- Worse recall/performance than dedicated systems at scale

### Milvus

- Written in Go + C++ (FAISS backend)
- Distributed-first with Kubernetes operator
- Multiple index types (IVF, HNSW, DiskANN, ANNOY)
- Scalar filtering with inverted index
- Best for billion-scale on-prem deployments

### Comparison

| System | Language | Index | Filtering | Distributed | Best for |
|---|---|---|---|---|---|
| FAISS | C++ | IVF/HNSW/PQ | No | No | Library / research |
| Qdrant | Rust | HNSW | Excellent | Yes | Production, filtered search |
| Weaviate | Go | HNSW | Good | Yes | Hybrid search, modules |
| Chroma | Python | HNSW | Basic | No | Prototyping, RAG |
| Pinecone | Managed | Proprietary | Good | Managed | Serverless, ease of use |
| pgvector | C | HNSW/IVF | Full SQL | Via Citus | Postgres-native |
| Milvus | Go/C++ | Many | Good | Yes | Billion scale |

---

## Further Reading

- **HNSW paper**: Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using HNSW" (2018)
- **IVF-PQ**: Jégou et al., "Product Quantization for Nearest Neighbor Search" (2011)
- **DiskANN**: Jayaram et al., "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node" (2019)
- **ScaNN**: Guo et al., "Accelerating Large-Scale Inference with Anisotropic Vector Quantization" (2020)
- **ACORN** (filtered HNSW): Patel et al., "ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data" (2024)
- **ann-benchmarks**: https://ann-benchmarks.com — recall vs QPS benchmarks across all major algorithms
