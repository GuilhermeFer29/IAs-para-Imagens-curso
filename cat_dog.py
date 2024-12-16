from datasets import load_dataset
from IPython.display import display
from transformers import pipeline


# Carrega o dataset Cat_and_Dog
database = load_dataset('Bingsu/Cat_and_Dog', split='test')

# Variavel line recebe 10 imagens aleatorias
line = database.shuffle()[:10]
model = 'akahana/vit-base-cats-vs-dogs'
  
# Função definer o tamanho da imagem
def show_image(image):
    image = image.copy()
    image.thubnail((250, 250)) # Tamanho da imagem
    display(image)

    
classifier = pipeline(   'image-classification', model=model)

# Imprime as imagens
for image in line['image']:  
    result = classifier(image)
    print(result)  
    show_image(image)