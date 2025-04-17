Wetness Mapper

A Python-based tool designed to calculate and visualize surface wetness indices from geospatial datasets. Utilizing satellite imagery and geospatial data, this tool aids in environmental monitoring, hydrological studies, and land management by providing insights into surface moisture distribution.​
GitHub
Features

    Wetness Index Calculation: Processes geospatial data to compute wetness indices, indicating surface moisture levels.

    Data Handling: Supports reading and processing data in Parquet format for efficient storage and retrieval.

    Visualization: Generates visual representations of wetness indices to facilitate analysis and interpretation.​

Installation

    Clone the Repository

    git clone https://github.com/Apfelirne5/wetness-mapper.git
    cd wetness-mapper

    Create a Virtual Environment (Optional but Recommended)

    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

    Install Dependencies

    pip install -r requirements.txt

Usage
1. Calculating Wetness Index

The calc_naesse.py script processes geospatial data to compute the wetness index.​
GitHub+1GitHub+1

python calc_naesse.py --input data/input_data.parquet --output results/wetness_index.parquet

    --input: Path to the input Parquet file containing geospatial data.

    --output: Path to save the computed wetness index Parquet file.​

2. Visualizing Wetness Index

The calc_naesse_nb.ipynb Jupyter Notebook provides an interactive environment to visualize the computed wetness indices.​
GitHub

To run the notebook:

jupyter notebook calc_naesse_nb.ipynb

Directory Structure

wetness-mapper/
├── images/                 # Contains visual assets
│   ├── wetness_map.png     # Visualization of wetness index
│   └── histogram.png       # Histogram of wetness distribution
├── calc_naesse.py          # Script to calculate wetness index
├── calc_naesse_nb.ipynb    # Jupyter Notebook for visualization
├── read_as_parquet.py      # Utility to read data as Parquet# Wetness Mapper

A Python-based tool designed to calculate and visualize surface wetness indices from geospatial datasets.  
Utilizing satellite imagery and geospatial data, this tool aids in environmental monitoring, hydrological studies, and land management by providing insights into surface moisture distribution.

---

## Features

- **Wetness Index Calculation**: Processes data geospatial data to compute, surface moisture levels.
- **Visualization**: Generates visual representations of wetness indices to facilitate analysis and interpretation.

---

## Example Visualizations

### Example 1 – Wetness Detection Process

**Dry component**  
![Test 1 Input](images/Test_1.png)

**Wet component**  
![Test 1 Wetness](images/Test_1_wet.png)

**Detected Wet Areas**  
![Test 1 Wet Detected](images/Test_1_wet_detected.png)

---

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Apfelirne5/wetness-mapper.git
   cd wetness-mapper

├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

Contributions are welcome! Please fork the repository and submit a pull request. For major changes, open an issue first to discuss your proposed modifications.
