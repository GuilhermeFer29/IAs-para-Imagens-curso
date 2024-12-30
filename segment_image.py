from transformers import pipeline
from IPython.display import display
from pathlib import Path
from PIL import Image

modelo = 'nvidia/segformer-b1-finetuned-cityscapes-1024-1024'

segment = pipeline('image-segmentation', model=modelo)

imagem = Image.open(Path('imagens/cidades/city.jpg'))

segmentacao = segment(imagem)
segmentacao