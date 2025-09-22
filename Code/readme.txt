# Hamiltonian Neural Networks (HNN)

This repository contains code for training and evaluating Hamiltonian Neural Networks (HNN) and related baselines.  

---

## 🔧 Data Generation

### 2D Systems
Run the following script to generate data for 2D systems (e.g., **Double Well**):

```bash
python data_gen_2.py --dynamics_name "double_well"                      --q_range -2.0 2.0                      --p_range -2.0 2.0                      --sim_len 20                      --time_step 0.01
```

### 4D System (Henon–Heiles)
Run:

```bash
python data_gen_chaotic.py
```

⚠️ **Note:** The data range for the Henon–Heiles system is defined within the file.  
We cannot generate data in arbitrary ranges, since the HH system does not admit closed-form solutions in the entire phase space.  

**Noise:** The default noise level is `0.01`, but you can change it directly in the `data_gen_2.py` file.  

---

## 🎯 Training HNN

To train an HNN and save the trained model:

```bash
python run_generic_adjoint.py --dynamics_name "double_well"                               --data_folder "double_well_20"                               --gt_res 0.01                               --hid_layers "[16, 32, 16]"                               --solver_res 0.01                               --pred True                               --sim_len 7                               --solver "im"
```

- Trained models are saved in the `models/` directory.  
- Use the notebook `plot_gen_data_main.ipynb` to generate the plots shown in the paper.  

---

## 📊 Baselines

Baseline implementations are provided in the `Baselines/` folder.  

- To reproduce **Table 1 results**:  
  Open the notebook:  

  ```
  NSSNN/plot_NSSNN_baseline.ipynb
  ```  

  Load the model you need, and run the cells to reproduce the results.  

- For **SHNN baselines**:  
  Open the notebook:  

  ```
  SHNN/plot_SHNN_baseline.ipynb
  ```

---

## 📂 Repository Structure
-  scripts/    :   `data_gen.py` → Data generation script.  
                   `run_generic_adjoint.py` → Main training script for HNN.  
-  models/     →    Directory where trained models are stored.  
-  notebooks/  :   `visualize.ipynb` → Plotting notebook for generated data.  
-  Baselines/  →    Contains baseline implementations.  
