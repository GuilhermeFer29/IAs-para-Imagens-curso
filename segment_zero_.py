from transformers import pipeline
from IPython.display import display
from pathlib import Path
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch
import matplotlib.pyplot as plt

# Cria o objeto pipeline
modelo = 'CIDAS/clipseg-rd64-refined'
segmentador = pipeline('image-segmentation', model=modelo)

# Imprime as imagens
imagem = Image.open(Path('imagens/cidades/city.jpg'))
processor = CLIPSegProcessor.from_pretrained(modelo)
model= CLIPSegForImageSegmentation.from_pretrained(modelo)

# Segmenta a imagem
prompts = ['street', 'cars', 'traffic light']
inputs= processor(text=prompts, images=[imagem] * len(prompts),
padding = True,
return_tensors='pt',) 

# Realiza a segmentação
with torch.no_grad():
    outputs = model(**inputs)  
    
predicoes  = outputs.logits.unsqueeze(1) 
segmentacao = []

# Cria o dicionario
for i , label in enumerate(prompts):
    d = {'label': label, 'mask': torch.sigmoid(predicoes[i][0]).numpy()}
    segmentacao.append(d)
segmentacao    

# Exibe as imagens
fig, axes = plt.subplots(nrows = 1 , ncols = len(prompts)+ 1, figsize= (16,4))

axes[0].imshow(imagem)

# Exibe as segmentações
for i, segmento in enumerate(segmentacao):
    axes[i+1].imshow(segmento['mask'], cmap='viridis' )
    axes[i+1].set_title(segmento['label'])