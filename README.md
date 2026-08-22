# LLM Inference Batching Simulation

This project simulates how Large Language Models (LLMs) handle multiple requests at the same time. It compares two methods of batching requests: static batching and continuous batching.

## What it does

When an LLM gets many requests, it processes them in batches to save time. 

1. **Static Batching**: The system takes a group of requests and processes them together. It waits until every single request in the group is completely finished before it starts a new group. If one request is very long and the others are short, the short ones finish early but the system still waits. This wastes time and compute power.
2. **Continuous Batching**: The system processes a group of requests, but as soon as one request finishes, it immediately brings in a new request to fill the empty spot. It never waits for the whole group to finish. This is much faster and more efficient.

This code simulates both methods so you can measure the difference.

## Files included

* `main.cpp`: The core C++ code that runs the simulation. It creates fake requests, processes them using either static or continuous batching, and records how long they take.
* `scripts/plot_latency.py`: A Python script that reads the results from the C++ program and creates graphs so you can see the performance difference.

## How to run

1. Compile the C++ code:
```bash
g++ -std=c++17 main.cpp -o main
```

2. Run the simulation. The arguments are: `mode policy lambda seed output_file num_requests pace`
```bash
./main continuous fcfs 5.0 42 results.csv 200 1
```

3. Generate the graphs using Python (requires pandas and matplotlib):
```bash
python scripts/plot_latency.py --datadir=. --outdir=.
```
