# APvectorDB

A vector database with ANN (Approximate Nearest Neighbour) search structures, built in C++23.

## Prerequisites

- CMake 3.26+
- A C++23-capable compiler (clang++ or g++)
- [vcpkg](https://github.com/microsoft/vcpkg) with `VCPKG_ROOT` set in your environment

```sh
export VCPKG_ROOT=/path/to/vcpkg
```

## Build

```sh
make          # configure + build (Release by default)
make test     # run tests
make run      # build and run the executable
make clean    # remove the build directory
make rebuild  # clean + build
```

To build in debug mode:

```sh
make BUILD_TYPE=Debug
```

## Project Structure

```
.
├── src/
│   ├── main.cpp           # entry point
│   ├── storage/           # vector storage and persistence
│   ├── index/             # ANN index implementations
│   │   ├── hnsw/          # Hierarchical Navigable Small World
│   │   ├── ivf/           # Inverted File Index
│   │   └── lsh/           # Locality Sensitive Hashing
│   ├── distance/          # distance metrics (cosine, euclidean, dot product)
│   └── query/             # search interface and top-k retrieval
├── include/
│   └── vecdb/
│       ├── index.hpp      # abstract ANN index interface
│       ├── collection.hpp # top-level collection (analogous to a table)
│       └── types.hpp      # Vector<T>, Metric enum, result types
├── bindings/              # pybind11/nanobind wrappers exposing C++ to Python
│   └── module.cpp
├── python/
│   └── apvectordb/        # Python package
│       ├── __init__.py
│       └── types.py       # pure-Python helpers and type stubs
├── tests/                 # GTest unit and integration tests
│   └── python/            # Python-level tests
│       └── test_*.py
├── CMakeLists.txt
├── pyproject.toml         # scikit-build-core config for `pip install .`
├── vcpkg.json             # vcpkg dependencies
└── Makefile
```

## Project Goal

APvectorDB is an educational project focused on building a vector database from the ground up in modern C++23. The goal is to gain a deeper understanding of the data structures, indexing algorithms, and systems design principles behind vector databases and Approximate Nearest Neighbour (ANN) search.

The project also serves as an opportunity to strengthen modern C++ development skills, explore software architecture for reusable libraries, and learn how native C++ code can be packaged and exposed to Python through bindings and Python packaging tools.
