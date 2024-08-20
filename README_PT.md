
# Sistema de Recomendação de Filmes usando LightFM

Este projeto é um sistema simples de recomendação de filmes construído usando a biblioteca [LightFM](https://making.lyst.com/lightfm/docs/home.html). Ele usa o [conjunto de dados MovieLens](https://grouplens.org/datasets/movielens/) para treinar um modelo e depois recomenda filmes aos usuários com base em suas preferências.

## Iniciando

### Pré-requisitos

Antes de começar, certifique-se de ter os seguintes pacotes Python instalados:

- `numpy`
- `lightfm`

Você pode instalá-los usando pip:

```bash
pip install numpy lightfm
```

### Explicação do Código

```python
import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# Buscar e formatar os dados
data = fetch_movielens(min_rating=4.0)

# Imprimir os dados de treinamento e teste
print(repr(data['train']))
print(repr(data['test']))

# Criar o modelo
model = LightFM(loss='warp')

# Treinar o modelo
model.fit(data['train'], epochs=30, num_threads=2)

def sample_recommendation(model, data, user_ids):
    
    # Número de usuários e filmes nos dados de treinamento
    n_users, n_items = data['train'].shape

    # Gerar recomendações para cada usuário informado
    for user_id in user_ids:
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        scores = model.predict(user_id, np.arange(n_items))
        top_items = data['item_labels'][np.argsort(-scores)]

        # Imprimir os resultados
        print("Usuário %s" % user_id)
        print("     Positivos conhecidos:")
        for x in known_positives[:3]:
            print("         %s" % x)
        print("     Recomendados:")
        for x in top_items[:3]:
            print("     %s" % x)

# Amostra de recomendações para usuários específicos
sample_recommendation(model, data, [3, 25, 450])
```

### Como Executar

1. Certifique-se de ter todas as bibliotecas necessárias instaladas.
2. Copie o código acima para um arquivo Python (por exemplo, `sistema_recomendacao.py`).
3. Execute o arquivo Python no seu terminal ou IDE.
4. O programa irá buscar o conjunto de dados MovieLens, treinar um modelo de recomendação e fornecer recomendações de filmes para usuários especificados.

