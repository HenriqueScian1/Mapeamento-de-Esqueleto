# 🧍‍♂️ Mapeamento de Esqueleto — Reconhecimento de Gestos e Ações com Python e OpenCV

## 📘 Visão Geral

O **mapeamento de esqueleto** (ou *estimativa da pose humana*) é a etapa responsável por **detectar e localizar as articulações do corpo humano** em uma imagem ou vídeo, permitindo compreender ações, gestos e posturas.
Essa técnica é amplamente utilizada em:

* Interação humano-computador
* Vigilância inteligente
* Jogos e esportes com visão computacional
* Tradução de linguagem de sinais (Libras)
* Reconhecimento de exercícios físicos

---

## 🧠 Modelos Utilizados

### 🔹 Modelo MPII

* Site: [http://human-pose.mpi-inf.mpg.de/](http://human-pose.mpi-inf.mpg.de/)
* Produz **15 pontos-chave** do corpo.
* Treinado com 25 mil imagens e 40 mil pessoas em 410 atividades humanas.
* Cada imagem possui **anotações de articulações corporais**.
* Ideal para exercícios e movimentos de corpo inteiro.

**Pontos do modelo MPII:**

```
0 - Cabeça
1 - Pescoço
2 - Ombro Direito
3 - Cotovelo Direito
4 - Pulso Direito
5 - Ombro Esquerdo
6 - Cotovelo Esquerdo
7 - Pulso Esquerdo
8 - Quadril Direito
9 - Joelho Direito
10 - Tornozelo Direito
11 - Quadril Esquerdo
12 - Joelho Esquerdo
13 - Tornozelo Esquerdo
14 - Peito
15 - Fundo
```

---

### 🔹 Modelo COCO (CAFFE)

* Site: [http://cocodataset.org/#keypoints-2018](http://cocodataset.org/#keypoints-2018)
* Produz **18 pontos-chave** (17 articulações + fundo).
* Baseado no *COCO Keypoints Challenge 2016*.
* Contém mais de 200.000 imagens e 250.000 pessoas anotadas.
* Muito usado em aplicações de reconhecimento de gestos e movimentos.

**Pontos do modelo COCO:**

```
0 - Nariz
1 - Pescoço
2 - Ombro Direito
3 - Cotovelo Direito
4 - Pulso Direito
5 - Ombro Esquerdo
6 - Cotovelo Esquerdo
7 - Pulso Esquerdo
8 - Quadril Direito
9 - Joelho Direito
10 - Tornozelo Direito
11 - Quadril Esquerdo
12 - Joelho Esquerdo
13 - Tornozelo Esquerdo
14 - Olho Direito
15 - Olho Esquerdo
16 - Orelha Direita
17 - Orelha Esquerda
18 - Fundo
```

---

## 🧩 Arquitetura da Rede Neural (VGGNet + OpenPose)

A detecção do esqueleto é feita usando uma **CNN (Rede Neural Convolucional)** baseada na **VGGNet**, com múltiplos estágios de previsão.

### Estrutura do processo:

1. **Entrada da imagem**
   A imagem é processada por 10 camadas convolucionais da VGGNet.

2. **Previsão dos mapas de confiança (Confidence Maps)**
   Indicam a **probabilidade da presença** de cada parte do corpo em cada pixel.

3. **Previsão dos mapas de afinidade (Part Affinity Fields - PAFs)**
   Representam **relações entre pares de partes** (ex: braço direito = ombro + cotovelo + punho).

4. **Inferência dos pontos-chave**
   Um algoritmo de inferência “ganancioso” (*greedy*) combina os pontos e afinidades para reconstruir o esqueleto 2D completo.

---

## 🗺️ Estrutura da Saída (Modelo COCO)

A rede retorna uma matriz 4D com a seguinte estrutura:

| Dimensão | Descrição                                               |
| -------- | ------------------------------------------------------- |
| 1        | ID da imagem (batch)                                    |
| 2        | Índice do ponto-chave (mapas de confiança + afinidades) |
| 3        | Altura do mapa (posição vertical)                       |
| 4        | Largura do mapa (posição horizontal)                    |

Para o modelo COCO:

* 18 mapas de confiança (pontos)
* 1 mapa de fundo
* 19 pares de afinidade × 2 (x e y)
* **Total: 57 mapas de saída**

---

## 🧭 Interpretação dos Mapas

### ✅ **Mapas de Confiança**

Imagens em tons de cinza com valores altos nas regiões onde a probabilidade de encontrar um ponto-chave é maior.

### 🔗 **Mapas de Afinidade (PAF)**

Campos vetoriais 2D que codificam a **direção e força da conexão** entre dois pontos (ex: ombro → cotovelo).

Exemplo de conexões:

* Pescoço ↔ Ombro Direito
* Ombro Direito ↔ Cotovelo Direito
* Cotovelo Direito ↔ Pulso Direito
* Quadril ↔ Joelho ↔ Tornozelo

Essas ligações definem o **esqueleto completo**.

---

## ⚙️ Fluxo do Mapeamento

1. Carregar a rede neural treinada (ex: `pose_deploy_linevec.prototxt` e `pose_iter_440000.caffemodel`).
2. Fazer o *forward pass* da imagem pelo modelo (`net.forward()` no OpenCV).
3. Extrair os mapas de confiança e afinidade.
4. Localizar os picos máximos (pontos de maior probabilidade).
5. Reconstruir as conexões válidas (usando os mapas de afinidade).
6. Desenhar o esqueleto sobre a imagem.

---

## 🧰 Exemplo Simplificado (Python + OpenCV)

```python
import cv2
import numpy as np

# Carregar modelo COCO
protoFile = "pose_deploy_linevec.prototxt"
weightsFile = "pose_iter_440000.caffemodel"
nPoints = 18

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# Ler imagem
frame = cv2.imread("pessoa.jpg")
frameWidth = frame.shape[1]
frameHeight = frame.shape[0]

# Pré-processar
inpBlob = cv2.dnn.blobFromImage(frame, 1.0/255, (368,368),
                                (0,0,0), swapRB=False, crop=False)
net.setInput(inpBlob)

# Forward
output = net.forward()

# Extração dos pontos
points = []
for i in range(nPoints):
    probMap = output[0, i, :, :]
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    x = (frameWidth * point[0]) / output.shape[3]
    y = (frameHeight * point[1]) / output.shape[2]
    if prob > 0.1:
        points.append((int(x), int(y)))
    else:
        points.append(None)
```

---

## 🧩 Aplicações Diretas

* **Reconhecimento de Exercícios (ex.: Polichinelo)**
  O código analisa a posição dos pontos para detectar braços e pernas em movimento.

* **Tradução de Libras (Linguagem de Sinais)**
  Usa pontos da mão e corpo para identificar letras e gestos.

* **Análise de Postura e Movimento**
  Detecta alinhamento corporal, inclinações e desequilíbrios.

---

## 🏁 Conclusão

O mapeamento de esqueleto é a base para sistemas de **reconhecimento de gestos e ações humanas**.
Combinando os **mapas de confiança e afinidade** obtidos por redes neurais convolucionais, é possível reconstruir com precisão a estrutura corporal e aplicar em diversas áreas da visão computacional.
