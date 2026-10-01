# Projeto Prático de Buscas e Árvores de Decisão

Projeto de Inteligência Artificial focado na implementação e comparação de buscas heurísticas e buscas locais.

## Problemas e algoritmos

### N-Puzzle — buscas heurísticas

São utilizados N-Puzzles, incluindo o 8-Puzzle, com geração aleatória de estados iniciais e verificação de solvabilidade.

Algoritmos:

- **Busca Gulosa**
- **A***
- **IDA***

As buscas utilizam a **Distância de Manhattan** como heurística:

`h(n) = soma das distâncias Manhattan de cada peça até sua posição objetivo`

#### Busca Gulosa

Prioriza os estados usando somente:

`f(n) = h(n)`

Assim, considera apenas a estimativa de distância até o objetivo.

#### A*

Combina o custo já percorrido e a heurística:

`f(n) = g(n) + h(n)`

onde `g(n)` representa o custo desde o estado inicial e `h(n)` a estimativa restante.

#### IDA*

O **IDA* (Iterative Deepening A*)** combina a avaliação `f(n) = g(n) + h(n)` com aprofundamento iterativo.

Em vez de manter toda a fronteira da busca em memória como o A*, o algoritmo realiza buscas em profundidade utilizando um limite para `f(n)`. Quando o limite é insuficiente para encontrar a solução, ele é aumentado e uma nova iteração é realizada.

A principal diferença experimental em relação ao A* é, portanto, a forma de exploração e o uso de memória: o IDA* usa muito menos memória, mas pode reexpandir estados em diferentes iterações.

Os experimentos registram:

- tempo de execução;
- quantidade de movimentos;
- nós expandidos;
- caminho completo da solução;
- indicação de solução encontrada.

### TSP — buscas locais

O Problema do Caixeiro Viajante utiliza instâncias baseadas no TSPLIB.

Algoritmos:

- **Hill Climbing (Subida de Encosta)**
- **Simulated Annealing (SA)**
- **SA com Reaquecimento**

O conversor de `.tsp` calcula a matriz de distâncias a partir das coordenadas das cidades e gera um JSON utilizado pelos experimentos.

#### Hill Climbing

Parte de uma rota inicial e procura vizinhos com menor custo, terminando quando não encontra uma melhoria.

#### Simulated Annealing

Além de aceitar melhorias, pode aceitar temporariamente soluções piores de acordo com a temperatura:

`P = exp(-Δ/T)`

A temperatura é reduzida durante a execução.

#### SA com Reaquecimento

É uma variação do Simulated Annealing. Quando a temperatura atinge o limite mínimo, ela pode ser reiniciada para a temperatura inicial, permitindo continuar a exploração de outras regiões do espaço de soluções.

A quantidade de reaquecimentos realizados é registrada no CSV.

## Experimentos

### Experimentos heurísticos

É possível definir uma lista de tamanhos e quantidades de tabuleiros, por exemplo:

```python
CONFIGURACOES = [
    (3, 100),
    (4, 10)
]
```

Isso gera 100 tabuleiros 3×3 e 10 tabuleiros 4×4. Cada tabuleiro solúvel é executado uma vez por cada algoritmo.

Os resultados incluem seed, solução, movimentos, nós expandidos e tempo.

### Experimentos locais

Para o TSP, o problema original pode ser copiado em memória e reduzido progressivamente:

`16 → 15 → 14 → ... → 2 cidades`

A cada tamanho, os algoritmos podem ser executados N vezes. O problema original não é alterado.

São registrados custo, tempo, seed, caminho e quantidade de reaquecimentos.

## Resultados

Os arquivos ficam na pasta:

```text
resultados/
```

Os experimentos podem concatenar resultados ou sobrescrever o CSV.

No experimento heurístico, o padrão é concatenar:

```bash
python experimento_buscas_heuristicas.py
```

Para sobrescrever:

```bash
python experimento_buscas_heuristicas.py --modo sobrescrever
```

## Estrutura

```text
projeto-buscas/
├── problemas/
│   └── *.tsp
├── problemas_json/
│   └── *.json
├── resultados/
│   ├── resultados_tsp.csv
│   └── resultados_heuristicas.csv
├── experimentos/
├── notebooks/
└── README.md
```

## Conversor TSP

O conversor recebe o nome do arquivo `.tsp`, procura-o em `./problemas/` e gera o JSON em `./problemas_json/`.

Exemplo:

```bash
node conversor.js ulysses16.tsp
```

Se o JSON já existir, ele é sobrescrito.

## Visualização

Os notebooks permitem comparar:

- tempo de execução;
- custo das soluções;
- nós expandidos;
- quantidade de movimentos;
- quantidade de reaquecimentos;
- médias e desvios padrão;
- melhores e piores caminhos do TSP.

O notebook do TSP também pode exibir o mapa original das cidades e as rotas encontradas.

## Fluxo

```text
N-Puzzle ──→ Gulosa / A* / IDA* ──→ CSV ──→ Gráficos

TSP ──→ .tsp ──→ conversor ──→ JSON
                              └──→ HC / SA / SA+Reaquecimento
                                      └──→ CSV ──→ Gráficos
```

## Métricas

### Tempo
Tempo necessário para executar cada busca, em milissegundos.

### Custo
Custo total da rota encontrada no TSP.

### Movimentos
Número de movimentos da solução do N-Puzzle.

### Nós expandidos
Quantidade de estados/nós efetivamente expandidos durante a busca.

### Reaquecimentos
Quantidade de vezes que a temperatura foi reiniciada no SA com Reaquecimento.

## Observações

As buscas locais possuem comportamento estocástico. Por isso, uma única execução não é suficiente para caracterizar seu desempenho; os experimentos podem ser repetidos e analisados por médias e dispersões.

Uma solução de busca local com baixo custo também não deve ser automaticamente considerada ótima. O objetivo é comparar o comportamento dos algoritmos.

## Tecnologias

- Python
- NumPy
- Pandas
- Matplotlib
- JSON
- CSV
- Jupyter Notebook / Google Colab
- Node.js

## Integrantes

- Ana Clara Oliveira
- Antônio Lucas Nascimento
- Enzo Guimarães
- Gabriel Gonçalves
