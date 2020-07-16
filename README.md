## Mask Detector 

Aplicativo de **visao computacional** e  **deep learning** para identificar se a pessoa esta utilizando mascara ou nao.

Foi desenvolvido em **python 3.7**, utilizando o detector de rosto MTCNN e TFLite para inferencia do modelo.


O aplicativo executa inferencia em tempo real pela webcam.

## Desenvolvimento
 
 O algoritmo foi desenvolvido em duas etapas:

1. Deteccao do rosto para extrair a area de inferencia do modelo
2.  Classificacao da imagem, para saber se esta ou nao de mascara
	


### Instalacao

Para instalar as dependencias do projeto execute o seguinte comando:

	pip install -r requirements



### Rodando o aplicativo

Para executar o aplicativo, utilize o seguinte o comando:

	python app.py


## Exemplo:

Neste trecho de codigo temos os principais parametros para executar o aplicativo:

```python
if __name__ == "__main__":    
    app = App(model='./model.tflite', label='./labels.txt', camera_id=0, size=480)  
    app.inference()
```


**model :** 

Caminho do modelo de classificacao

**label :** 

Caminho dos rotulos de classificacao

**camera_id :** 

Indice da camera

**size :** 

Redimensionamento da imagem apresentada em tempo real



**Obs:** Acesse a pasta "exemplo" para ver as imagens classificadas pelo modelo.