from transformers import pipeline
from IPython.display import display
from pathlib import Path
from PIL import Image

# Cria o objeto pipeline
modelo = 'openai/clip-vit-large-patch14'
classifier = pipeline('zero-shot-image-classification', model=modelo)

def mostrar_imagem(imagem):
    imagem = imagem.copy()
    imagem.thumbnail((250, 250)) # Tamanho da imagem
    display(imagem)

animais = [Image.open(arquivos) for arquivos in Path('imagens/animals').iterdir()]

classes = [
    'wolf',
    'tiger',
    'dog',
    'cat',
    'puma',
    'turtle',
    'bird',
]

for imagem  in animais :
    predicao = classificador(imagem, candidate_labels=classes)
    print('---predicao ---')
    
    for p in predicao:
        label = p['label']
        score = p['score']
        score_ajustado = f'{100 * score:.2f}%'
        print(f'{label} : {score_ajustado}')
    mostrar_imagem(imagem)    





