# LiSENCE Ensemble Ligand and Sequence Encoder Networks for CYP450 inhibitors Explainable Prediction 

Authors:Abena Achiaa Atwereboannah, Alantari Mugahed Ali Shawqi, ...

![image](https://github.com/user-attachments/assets/3137abef-5892-4029-a438-3da777e2d33a)

# Motivation
Adverse drug effects from Drug-Drug Interactions (DDIs) pose significant risks for patients on multiple medications. A primary cause of DDIs is the inhibition of Cytochrome P450 (CYP450) enzymes, crucial for drug metabolism. Identifying potential CYP450 inhibitors accurately is essential for drug development, yet current machine learning (ML) approaches fall short in this regard. Many ML algorithms struggle to determine the importance of individual components of input data, such as drugs’ ligand or compound SMILES strings and protein target sequences. Since data is fundamental to ML, understanding which aspects contribute most to predictions is vital for biological inferences. Current methods often overlook this aspect. This research introduces LiSENCE, an innovative AI framework with four modules: the ligand encoder network (LEN), sequence encoder network (SEN), ensemble classification module, and explainability (XAI) module. LEN and SEN, as deep learning pipelines, extract high-level features from drug ligand strings and CYP protein target sequences, respectively. These features are combined to improve prediction performance, with the XAI module providing biological interpretations. The proposed framework is fine-tuned and evaluated using two datasets: drug’s ligand/compounds SMILES strings from the PubChem database and protein target sequences from the protein data bank (PDB) with five CYP isoforms: 1A2, 2C9, 2C19, 2D6, and 3A4. 

