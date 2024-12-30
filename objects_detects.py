from transformers import pipeline
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patches

# Cria o objeto pipeline
detector = pipeline('object-detection', model = 'facebook/detr-resnet-50')
# Imprime as imagens
imagem = Image.open(Path('imagens/cidades/city.jpg'))
deteccao = detector(imagem)

# Exibe a imagem
fig, ax = plt.subplots(figsize = (16,16))
ax.imshow(imagem)

# Exibe a detecção
det = deteccao[0]
box = det['box']

origem = (box['xmin'], box['ymin'])
largura = box['xmax'] - box['xmin']
altura =  box['ymax'] - box['ymin']

# Desenha o retangulo
rect = patches.Rectangle(
    origem, 
    largura, 
    altura, 
    linewidth = 3 , 
    edgecolor = 'red',
    facecolor = 'none'
    )

# Adiciona o retangulo
ax.add_patch(rect)
texto = f'{det["label"]} {100 * det["score"] :.2f}%'

# Adiciona o texto
ax.text(box['xmin'], box['ymin'] - 15, 
        texto, 
        bbox={'facecolor': 'red', 'alpha': 0.8})


# Exibe a imagem
plt.show()