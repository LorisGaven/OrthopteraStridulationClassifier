import pandas as pd
import librosa
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

based_sr = 48000

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def go(path_df, wav_path_file, save_path, save_path_df):
    df = pd.read_csv(path_df)

    new_rows = []

    idx_image = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        chunk_initial_time = row["chunk_initial_time"]
        chunk_final_time = row["chunk_final_time"]
        annotation_initial_time = row["annotation_initial_time"]
        annotation_final_time = row["annotation_final_time"]
        min_frequency = row["min_frequency"]
        max_frequency = row["max_frequency"]
        code_unique = row["code_unique"]

        wav_path = f'{wav_path_file}\\{code_unique}_split_{chunk_initial_time}_{chunk_final_time}.wav'
        #print(wav_path)
        
        _, sr = librosa.load(wav_path, sr=None)
        y, _ = librosa.load(wav_path, sr=based_sr)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

        img = scale_minmax(D, 0, 255).astype(np.uint8) # Mettre les valeurs entre 0 et 255
        baseimg = Image.fromarray(img)
        
        ratio = min(sr / based_sr, 1)
        img = img[0:int(img.shape[0]*ratio), :]
        
        img = np.flip(img, axis=0)

        image = Image.fromarray(img)
        image = image.resize((baseimg.width, baseimg.height))
        n = int(chunk_final_time) - int(chunk_initial_time)
        segment_width = image.width // n

        # Diviser l'image et sauvegarder chaque segment
        for i in range(n):
            idx_image += 1
            # Définir la boîte de découpage pour chaque segment
            left = i * segment_width
            top = 0
            right = left + segment_width if i < 4 else image.width
            bottom = image.height
            
            # Découper l'image
            segment = image.crop((left, top, right, bottom))
            
            # Définir le chemin du fichier de sortie pour le segment actuel
            #new_segment_name = f'{code_unique}_split_{chunk_initial_time + i}_{chunk_initial_time + i + 1}.png'
            output_path = os.path.join(save_path, str(idx_image) + ".png")

            if annotation_initial_time < chunk_initial_time + i + 1 and annotation_final_time > chunk_initial_time + i:
                new_row = row.copy()
                new_row["chunk_initial_time"] = chunk_initial_time + i
                new_row["chunk_final_time"] = chunk_initial_time + i + 1
                new_row["annotation_initial_time"] = max(annotation_initial_time, chunk_initial_time + i)
                new_row["annotation_final_time"] = min(annotation_final_time, chunk_initial_time + i + 1)
                new_row["code_unique"] = idx_image
                new_row['sampling_rate'] = based_sr

                new_row['min_frequency'] = row['min_frequency'] / min(row['sampling_rate'] / based_sr, 1)
                new_row['max_frequency'] = row['max_frequency'] / min(row['sampling_rate'] / based_sr, 1)

                new_rows.append(new_row)

            img_width, img_height = segment.width, segment.height


            # Création de la BB
            t1 = new_row["annotation_initial_time"] - new_row["chunk_initial_time"]
            t2 = new_row["annotation_final_time"] - new_row["chunk_initial_time"]
            f1 = int(new_row['min_frequency'])
            f2 = int(new_row['max_frequency'])
            sr = new_row['sampling_rate']
            chunk_duration = new_row['chunk_final_time'] - new_row['chunk_initial_time']

            t1_pixel = int(t1 * img_width/chunk_duration)
            t2_pixel = int(t2 * img_width/chunk_duration)

            # for the frequency, the spectrogram is flipped
            f1_pixel = img_height - int(f1 * img_height/(sr/2))
            f2_pixel = img_height - int(f2 * img_height/(sr/2))

            masque = np.zeros((img_height, img_width))
            masque[f2_pixel:f1_pixel, t1_pixel:t2_pixel] = 1

            segment_np = np.array(segment)
            segment_np = segment_np * masque # Application du masque

            segment = Image.fromarray(segment_np).convert('RGB')
            # Enregistrer le segment
            segment.save(output_path)
            

    new_df = pd.concat(new_rows, axis=1).transpose()
    file_name  = path_df.split("\\")[-1]
    print(file_name)

    new_df.to_csv(f'{save_path_df}\\{file_name}', index=False)

path_df = "D:\OrthopteraStridulationClassifier\Dataset acoustique insectes\CSVs morceaux audio 5s\Audible\\train_audible_recording_chunks.csv"
wav_path_file = "D:\OrthopteraStridulationClassifier\Dataset acoustique insectes\Sélection morceaux audio 5s\Audible\\train"
save_path = "D:\OrthopteraStridulationClassifier\Dataset acoustique insectes\\newDatasetClassification\\train_spectro"
save_path_df = "D:\OrthopteraStridulationClassifier\Dataset acoustique insectes\\newDatasetClassification"

print("Création pour train")
go(path_df, wav_path_file, save_path, save_path_df)

path_df = "D:\OrthopteraStridulationClassifier\Dataset acoustique insectes\CSVs morceaux audio 5s\Audible\\test_audible_recording_chunks.csv"
wav_path_file = "D:\OrthopteraStridulationClassifier\Dataset acoustique insectes\Sélection morceaux audio 5s\Audible\\test"
save_path = "D:\OrthopteraStridulationClassifier\Dataset acoustique insectes\\newDatasetClassification\\test_spectro"
save_path_df = "D:\OrthopteraStridulationClassifier\Dataset acoustique insectes\\newDatasetClassification"

print("Création pour test")
go(path_df, wav_path_file, save_path, save_path_df)

path_df = "D:\OrthopteraStridulationClassifier\Dataset acoustique insectes\CSVs morceaux audio 5s\Audible\\val_audible_recording_chunks.csv"
wav_path_file = "D:\OrthopteraStridulationClassifier\Dataset acoustique insectes\Sélection morceaux audio 5s\Audible\\val"
save_path = "D:\OrthopteraStridulationClassifier\Dataset acoustique insectes\\newDatasetClassification\\val_spectro"
save_path_df = "D:\OrthopteraStridulationClassifier\Dataset acoustique insectes\\newDatasetClassification"

print("Création pour val")
go(path_df, wav_path_file, save_path, save_path_df)