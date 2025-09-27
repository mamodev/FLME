<p align="center">
<img src="https://dn721601.ca.archive.org/0/items/unipi_logo/Logo_UNIPI_page-0001.jpg" alt="University of Pisa Logo" width="200"/>
</p>


# FLME – Federated Learning Made Easy

**FLME (Federated Learning Made Easy)** is a research and development project that aims to
provide a practical, scalable, and resource-aware framework for cross-device Federated Learning (FL).
It is both a **research platform** for exploring open problems in FL (heterogeneous data,
asynchronous training, scalability) and a **prototype framework** targeting production-level
deployments at scale.

This work originates from the **Bachelor Thesis of Marco Morozzi** in collaboration with
researchers from the [University of Pisa](https://unipi.it), 
[Massimo Torquati](http://calvados.di.unipi.it/paragroup/torquati/), and 
[Patrizio Dazzi](https://pages.di.unipi.it/dazzi/).
You can find the thesis here: [Federated Learning Made Easy (Thesis PDF)](https://github.com/mamodev/FLME/wiki/Studio-Preliminare)

---

## ✨ Motivation

Federated Learning is often centralized and synchronous: a central orchestrator coordinates
clients and enforces synchronization barriers. This does not reflect the dynamics of **real-world
cross-device FL**, where client availability is dictated by energy constraints, bandwidth,
computation power, and data readiness.

FLME introduces a new paradigm called **Resource Driven Federated Learning**, where:

- ⚡ Clients decide when to participate (based on availability of resources such as
  battery, connectivity, compute, or newly collected data).
- 🔄 Training is asynchronous and fault-tolerant by design.
- 📈 The system can scale to millions of clients with low overhead.

---

## 🎯 Project Goals

1. **An Easy Interface for FL Applications**  
   - Abstract away low-level implementation details.  
   - Enable researchers and developers to prototype and experiment with real-world FL
     setups without writing orchestration logic.  
   - Provide flexibility and personalization via a scripting interface 
     (**Python**, **Lua**) to customize training strategies.

2. **A High-Performance Federated Learning Engine**  
   - Implementation in **C++** using **coroutines + io_uring** for highly scalable and efficient
     communication.  
   - Fault-tolerant and scalable server backends capable of handling millions of clients.  
   - Include state-of-the-art FL optimization strategies **out-of-the-box**.  
   - Planned **GPU/CUDA support** for aggregation-intensive tasks.  

---

## 🧪 Research Directions

FLME investigates fundamental challenges in Federated Learning:

- Limited scalability and flexibility of synchronous FL (stragglers, barriers, inefficiency).
- Convergence of **asynchronous FL** under **non-IID data distributions** and **non-uniform
  client training parameters**.
- Simulation and empirical validation of real-world heterogeneity.

---

## 🧰 Simulation Environment – `flcdata/`

To study FL under realistic conditions, FLME provides a **simulation environment**
(`flcdata`) supporting thousands of virtual clients, where **data and system heterogeneity**
can be controlled precisely and **reproduced deterministically**.

- **Backend**: Python  
- **Frontend**: React  
- Generates synthetic FL-oriented datasets with fine control over:
  - Label distribution skew  
  - Feature distribution skew  
  - Quantity skew  
  - Concept drift  
  - Concept shift  
  (see the survey [Measuring data heterogeneity in FL](https://arxiv.org/pdf/1912.04977))  

### Running the Simulation Environment
```bash
cd flcdata
pip install -r requirements.txt
# then install frontend
npm install   # or yarn install / bun install
# preview mode (runs frontend + backend)
npm run runpreview
```

---

## 📂 Repository Structure

- `flcdata/` → Simulation environment (frontend + backend).  
- `cpp/` → C++ implementation of the production-level FL framework (currently under refactoring).  
- `paper/` → Jupyter notebooks and scripts to reproduce results from the ongoing research paper.  
- `experiments/` → Exploratory implementations and replication of other FL papers.  

---

## 🌍 Audience

- **Researchers**: Study asynchronous FL under reproducible heterogeneity conditions.  
- **Developers (future)**: Use a fault-tolerant and scalable framework to deploy federated
  applications cross-device.  

---


## 🤝 Contributing

Contributions are welcome! 🎉  
At this stage, there are no strict guidelines. If you are interested in contributing, please
contact **Marco Morozzi** directly.  

---

## 📖 References

- [Federated Learning Made Easy (Thesis)](https://dummy-thesis-link.it) – Marco Morozzi  
- [University of Pisa](https://dummy-unipi.it) – Research Collaboration  
- [Survey on Heterogeneity in Federated Learning](https://arxiv.org/pdf/1912.04977)  

---

## 👨‍💻 Authors & Acknowledgments

- **Marco Morozzi** – Developer & Researcher  
- **Prof. Massimo Torquati** – University of Pisa  
- **Prof. Patrizio Dazzi** – University of Pisa  

---
