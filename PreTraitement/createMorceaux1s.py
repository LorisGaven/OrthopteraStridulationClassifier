import pandas as pd
from tqdm import tqdm
import os
from PIL import Image

def convertSpectro5sTo1s(path, save_path):
    # Assurer que le dossier de sauvegarde existe
    os.makedirs(save_path, exist_ok=True)
    
    # Obtenir la liste de tous les fichiers dans le dossier 'path'
    for filename in tqdm(os.listdir(path)):
        # Construire le chemin complet vers l'image
        file_path = os.path.join(path, filename)
        
        # Vérifier si le fichier est une image (par exemple, vérifier l'extension)
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                # Ouvrir l'image
                image = Image.open(file_path)
                
                # Calculer la largeur de chaque segment
                segment_width = image.width // 5

                # Extraire la partie de base du nom de fichier sans l'extension et les indices finaux
                chunk_initial = int(filename.split('_')[-2])
                base_name = '_'.join(filename.split('_')[:-2])
                
                # Diviser l'image et sauvegarder chaque segment
                for i in range(5):
                    # Définir la boîte de découpage pour chaque segment
                    left = i * segment_width
                    top = 0
                    right = left + segment_width if i < 4 else image.width
                    bottom = image.height
                    
                    # Découper l'image
                    segment = image.crop((left, top, right, bottom))
                    
                    # Définir le chemin du fichier de sortie pour le segment actuel
                    new_segment_name = f"{base_name}_{chunk_initial + i}_{chunk_initial + i+1}.png"
                    output_path = os.path.join(save_path, new_segment_name)
                    
                    # Enregistrer le segment
                    segment.save(output_path)
                    
                #print(f"Les segments de '{filename}' ont été sauvegardés avec succès.")
                
            except Exception as e:
                print(f"Une erreur est survenue lors du traitement de '{filename}': {e}")

path = 'D:\OrthopteraStridulationClassifier\Dataset acoustique insectes\Sélection morceaux audio 5s\Audible\\train_spectro'
save_path = 'D:\OrthopteraStridulationClassifier\Dataset acoustique insectes\Sélection morceaux audio 1s\Audible\\train_spectro'
convertSpectro5sTo1s(path, save_path)

path = 'D:\OrthopteraStridulationClassifier\Dataset acoustique insectes\Sélection morceaux audio 5s\Audible\\test_spectro'
save_path = 'D:\OrthopteraStridulationClassifier\Dataset acoustique insectes\Sélection morceaux audio 1s\Audible\\test_spectro'
convertSpectro5sTo1s(path, save_path)

def convertTo1s(path, save_path):
    print(f'conversion de {path}')
    df = pd.read_csv(path)
    new_df = pd.DataFrame(columns=df.columns)

    rows = df.iterrows()
    for _, row in tqdm(rows, total=len(df)):
        chunk_initial_time = row["chunk_initial_time"]
        chunk_final_time = row["chunk_final_time"]
        annotation_initial_time = row["annotation_initial_time"]
        annotation_final_time = row["annotation_final_time"]

        time_start = chunk_initial_time
        while time_start < chunk_final_time:

            if annotation_initial_time < time_start + 1 and annotation_final_time > time_start:
                nouvelle_ligne = row
                nouvelle_ligne["chunk_initial_time"] = time_start
                nouvelle_ligne["chunk_final_time"] = time_start + 1
                new_df.loc[len(new_df.index)] = nouvelle_ligne
                #new_df = pd.concat([new_df, pd.DataFrame([nouvelle_ligne])], ignore_index=True)

            time_start += 1

    new_df.to_csv(save_path, index=False)

train_path_5s = "..\Dataset acoustique insectes\CSVs morceaux audio 5s\Audible\\train_audible_recording_chunks.csv"
train_save_path = "..\Dataset acoustique insectes\CSVs morceaux audio 1s\Audible\\train_audible_recording_chunks.csv"
convertTo1s(train_path_5s, train_save_path)

train_path_5s = "..\Dataset acoustique insectes\CSVs morceaux audio 5s\Audible\\test_audible_recording_chunks.csv"
train_save_path = "..\Dataset acoustique insectes\CSVs morceaux audio 1s\Audible\\test_audible_recording_chunks.csv"
convertTo1s(train_path_5s, train_save_path)
