# Senior_Thesis_S24
This repository includes the code for Allen Wu's Senior Thesis Project at Princeton University in 2024.


# Installations Required
To run this code, installation of Stockfish 14.1 and Lichess' database from January 2019 is required as follows:

Stockfish 14.1 installation link: https://stockfishchess.org/blog/2021/stockfish-14-1/

Lichess (01/19) database installation link: https://csslab.cs.toronto.edu/datasets/#maia_kdd (install January)

Make sure that the actual file paths correctly correspond to the file paths used in the code.


# Running Clustering Script
To run the clustering script, navigate to the 'Senior Thesis' folder, and run the following command: "sbatch cluster_instructions.slurm"

Another useful command to view the current jobs in queue is as follows: "squeue -u (netid)"

Every instance a clustering script is run, the respective delta_materials and move_match data will be stored in the "to_save_folder", defined in the clustering script code in data_collection_and_cluster_code.py. Make sure to call the same folder in your aggregation code for intended results across the all player data case and filtered data cases (by elo and time).