from tqdm import tqdm
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

based_sr = 48000

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def convert_wav_to_spectrogram(wav_path, save_path):
    # Charger l'audio
    y, sr = librosa.load(wav_path, sr=based_sr)
    # Calculer le spectrogramme
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    img = scale_minmax(D, 0, 255).astype(np.uint8) # Mettre les valeurs entre 0 et 255
    img = np.flip(img, axis=0)

    im = Image.fromarray(img)
    im.save(save_path)

    
    """# Créer la figure sans les axes
    fig, ax = plt.subplots()
    im = librosa.display.specshow(D, sr=sr, x_axis=None, y_axis=None, ax=ax)
    print(type(im))
    
    # Sauvegarder le spectrogramme sans les axes comme image
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Fermer la figure pour libérer la mémoire"""

def convert_folder(input_folder, output_folder):
    # S'assurer que le dossier de sortie existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    files = os.listdir(input_folder)
    # Parcourir tous les fichiers dans le dossier d'entrée
    for filename in tqdm(files):
        if filename.endswith(".wav"):
            # Construire le chemin complet vers le fichier .wav
            wav_path = os.path.join(input_folder, filename)
            # Construire le chemin de sauvegarde pour l'image du spectrogramme
            save_path = os.path.join(output_folder, filename.replace(".wav", ".png"))
            
            # Convertir le fichier .wav en spectrogramme et le sauvegarder
            print(f"Converting {filename} to spectrogram...")
            convert_wav_to_spectrogram(wav_path, save_path)
    
    print("Conversion completed.")

# Remplacez ces chemins par vos chemins de dossiers appropriés
input_folder = "D:\OrthopteraStridulationClassifier\Dataset acoustique insectes\Sélection soundscapes 1 min"
output_folder = "D:\OrthopteraStridulationClassifier\Dataset acoustique insectes\Sélection soundscapes 1min spectro"

#input_folder = "D:\OrthopteraStridulationClassifier\Dataset acoustique insectes\Sélection morceaux audio 5s\Audible\\train"
#output_folder = "D:\OrthopteraStridulationClassifier\Dataset acoustique insectes\Sélection morceaux audio 5s\Audible\\train_spectro"

#input_folder = "DataSet/Selection_morceaux_audio_5s/Audible/Audible/val"
#output_folder = "DataSet/Selection_morceaux_audio_5s/Audible/val_spectro"

convert_folder(input_folder, output_folder)
