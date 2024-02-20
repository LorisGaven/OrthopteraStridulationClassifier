import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

yolo_df_path = "D:\OrthopteraStridulationClassifier\Dataset acoustique insectes\yolo_dataset_soudscape\df_yolo_predicted.csv"
yolo_image_path = "D:\OrthopteraStridulationClassifier\Dataset acoustique insectes\yolo_dataset_soudscape\soundscape_to_5s"

df_yolo = pd.read_csv(yolo_df_path)

soundscape_truth_df = "D:\OrthopteraStridulationClassifier\Dataset acoustique insectes\spectro1min\CSVs soundscapes 1 min\selected_soundscapes.csv"
soundscape_image_path = "D:\OrthopteraStridulationClassifier\Dataset acoustique insectes\spectro1min\Sélection soundscapes 1min spectro"

df_truth = pd.read_csv(soundscape_truth_df)

def image_to_compare(code_unique, save_path):
    df_yolo_filtered = df_yolo[df_yolo["unique_code"] == code_unique]
    df_truth_filtered = df_truth[(df_truth["code_unique"] == code_unique) & (df_truth["label_class"] == "Insecta")]

    chunk_duration = (df_truth_filtered["duree_min"] * 60 + df_truth_filtered["duree_sec"]).iloc[0]

    image = Image.open(f'{soundscape_image_path}\{code_unique}.png')
    img_width, img_height = image.width, image.height

    # On fait le truth
    image = Image.open(f'{soundscape_image_path}\{code_unique}.png')
    img_width, img_height = image.width, image.height

    plt.figure(figsize=(50, 10))

    for _, row in df_yolo_filtered.iterrows():
        label = row["predicted_label"]
        x1 = row["x1"]
        x2 = row["x2"]
        y1 = row["y1"]
        y2 = row["y2"]
        start_split = row["start_split"] * (img_width // chunk_duration)
        end_split = row["end_split"] * (img_width // chunk_duration)

        t1_pixel = x1 + start_split
        f1_pixel = y1
        f2_pixel = y2
        t2_pixel = x2 + start_split

        rect = patches.Rectangle((t1_pixel, f1_pixel), t2_pixel - t1_pixel, f2_pixel - f1_pixel, linewidth=2, edgecolor='yellow', facecolor='none')
        plt.gca().add_patch(rect)

        plt.text(t1_pixel, f1_pixel + 0.01, label, verticalalignment='bottom', horizontalalignment='left',
            color='black',  # Couleur du texte
            bbox=dict(facecolor='yellow', edgecolor='none', pad=2.0))

    plt.imshow(image)
    plt.title(f'{code_unique} predicted')
    plt.tight_layout()
    plt.savefig(f'{save_path}\{code_unique}_predicted.png')

    plt.figure(figsize=(50, 10))

    for _, row in df_truth_filtered.iterrows():
        chunk_initial_time = 0.0
        chunk_final_time = 60.0
        annotation_initial_time = row["annotation_initial_time"]
        annotation_final_time = row["annotation_final_time"]
        min_frequency = row["min_frequency"]
        max_frequency = row["max_frequency"]
        label = row["label"]

        # Create the BB
        t1 = row["annotation_initial_time"]
        t2 = row["annotation_final_time"]
        f1 = int(row['min_frequency'])
        f2 = int(row['max_frequency'])
        sr = 48000 #row['sampling_rate']
        #sr = row['sampling_rate']

        t1_pixel = int(t1 * img_width/chunk_duration)
        t2_pixel = int(t2 * img_width/chunk_duration)

        # for the frequency, the spectrogram is flipped
        f1_pixel = img_height - int(f1 * img_height/(sr/2))
        f2_pixel = img_height - int(f2 * img_height/(sr/2))

        rect = patches.Rectangle((t1_pixel, f1_pixel), t2_pixel - t1_pixel, f2_pixel - f1_pixel, linewidth=2, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)

        plt.text(t1_pixel, f2_pixel + 0.01, label, verticalalignment='bottom', horizontalalignment='left',
            color='white',  # Couleur du texte
            bbox=dict(facecolor='red', edgecolor='none', pad=2.0))

    plt.imshow(image)
    plt.title(f'{code_unique} truth')
    plt.tight_layout()
    plt.savefig(f'{save_path}\{code_unique}_truth.png')
    plt.clf()
    plt.cla()
    
    print(f'{code_unique} fini')

def image_to_compare_pillow(code_unique, save_path):
    df_yolo_filtered = df_yolo[df_yolo["unique_code"] == code_unique]
    df_truth_filtered = df_truth[(df_truth["code_unique"] == code_unique) & (df_truth["label_class"] == "Insecta")]

    chunk_duration = (df_truth_filtered["duree_min"] * 60 + df_truth_filtered["duree_sec"]).iloc[0]

    # Charger l'image une seule fois
    image_path = f'{soundscape_image_path}/{code_unique}.png'
    image = Image.open(image_path)
    img_width, img_height = image.width, image.height

    # Créer une copie pour les prédictions YOLO
    img_yolo = image.copy().convert("RGB")
    draw_yolo = ImageDraw.Draw(img_yolo)
    # Utiliser une police de caractère par défaut si nécessaire
    # Pour utiliser une police spécifique, téléchargez-la et spécifiez le chemin
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    for _, row in df_yolo_filtered.iterrows():
        label = row["predicted_label"]
        x1, x2, y1, y2 = row["x1"], row["x2"], row["y1"], row["y2"]
        start_split = row["start_split"] * (img_width // chunk_duration)
        end_split = row["end_split"] * (img_width // chunk_duration)

        t1_pixel = int(x1 + start_split)
        f1_pixel = int(y1)
        f2_pixel = int(y2)
        t2_pixel = int(x2 + start_split)

        # Dessiner le rectangle jaune
        draw_yolo.rectangle([t1_pixel, f1_pixel, t2_pixel, f2_pixel], outline="yellow", width=2)
        # Ajouter le texte
        draw_yolo.text((t1_pixel, f1_pixel), label, fill="black", font=font)

    # Enregistrer l'image avec les prédictions YOLO
    img_yolo.save(f'{save_path}/{code_unique}_predicted.png')

    # Créer une copie pour les annotations de vérité terrain
    img_truth = image.copy().convert("RGB")
    draw_truth = ImageDraw.Draw(img_truth)

    for _, row in df_truth_filtered.iterrows():
        label = row["label"]
        t1_pixel = int(row["annotation_initial_time"] * img_width / chunk_duration)
        t2_pixel = int(row["annotation_final_time"] * img_width / chunk_duration)
        f1_pixel = img_height - int(row['min_frequency'] * img_height / (24000))  # 24000 est la moitié de sr pour un spectrogramme jusqu'à 48000 Hz
        f2_pixel = img_height - int(row['max_frequency'] * img_height / (24000))

        # Dessiner le rectangle rouge
        draw_truth.rectangle([t1_pixel, f2_pixel, t2_pixel, f1_pixel], outline="red", width=2)
        # Ajouter le texte
        draw_truth.text((t1_pixel, f2_pixel), label, fill="white", font=font)

    # Enregistrer l'image avec les annotations de vérité terrain
    img_truth.save(f'{save_path}/{code_unique}_truth.png')

    print(f'{code_unique} fini')

code_uniques = df_truth["code_unique"].unique()
save_path = r"D:\OrthopteraStridulationClassifier\Dataset acoustique insectes\yolo_dataset_soudscape\test_result"

for code_unique in tqdm(code_uniques):
    image_to_compare(code_unique, save_path)