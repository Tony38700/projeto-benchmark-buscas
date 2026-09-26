import json
import csv
import time
import numpy as np


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO_TSP = "ulysses16.json"

ARQUIVO_RESULTADOS = "resultados_tsp.csv"

# Quantidade de vezes que cada algoritmo será executado
# para cada quantidade de cidades.
N_EXECUCOES = 10

# Cidade inicial da rota
INICIO = 0

# Seed principal do experimento.
#
# Se for None:
#   cada execução terá resultados diferentes.
#
# Se for um número:
#   o experimento poderá ser reproduzido
#   exatamente em outra execução.
SEED_EXPERIMENTO = 12345


# ============================================================
# CARREGAR PROBLEMA ORIGINAL
# ============================================================

with open(
    "./problemas_json/" + ARQUIVO_TSP,
    "r"
) as arquivo:

    tsp_original = json.load(arquivo)


# ============================================================
# FUNÇÕES DE CUSTO
# ============================================================

def custo_rota(
    rota,
    distancias
):

    custo = 0

    for i in range(
        len(rota) - 1
    ):

        custo += distancias[
            rota[i]
        ][
            rota[i + 1]
        ]

    return custo


# ============================================================
# GERAR VIZINHO
# ============================================================

def gerar_vizinho(
    rota,
    rng
):

    vizinho = rota.copy()

    i, j = rng.choice(
        np.arange(
            1,
            len(vizinho) - 1
        ),
        size=2,
        replace=False
    )

    vizinho[i], vizinho[j] = (
        vizinho[j],
        vizinho[i]
    )

    return vizinho


# ============================================================
# HILL CLIMBING
# ============================================================

def subida_de_encosta(
    rota_inicial,
    distancias,
    iter_max=600
):

    rota_atual = rota_inicial.copy()

    custo_atual = custo_rota(
        rota_atual,
        distancias
    )

    # Com apenas 2 cidades, não existe
    # nenhuma troca possível entre cidades internas.
    if len(rota_atual) <= 3:
        return (
            rota_atual,
            custo_atual
        )

    for _ in range(iter_max):

        vizinhos = []

        for i in range(
            1,
            len(rota_atual) - 1
        ):

            for j in range(
                i + 1,
                len(rota_atual) - 1
            ):

                vizinho = rota_atual.copy()

                vizinho[i], vizinho[j] = (
                    vizinho[j],
                    vizinho[i]
                )

                vizinhos.append(
                    vizinho
                )

        melhor_vizinho = min(
            vizinhos,
            key=lambda rota:
                custo_rota(
                    rota,
                    distancias
                )
        )

        custo_vizinho = custo_rota(
            melhor_vizinho,
            distancias
        )

        if custo_vizinho >= custo_atual:

            return (
                rota_atual,
                custo_atual
            )

        rota_atual = melhor_vizinho

        custo_atual = custo_vizinho

    return (
        rota_atual,
        custo_atual
    )


# ============================================================
# SIMULATED ANNEALING
# ============================================================

def simulated_annealing(
    rota_inicial,
    distancias,
    rng,
    temperatura_inicial=1000,
    taxa_resfriamento=0.995,
    temperatura_minima=1e-3,
    iter_max=600
):

    rota_atual = rota_inicial.copy()

    custo_atual = custo_rota(
        rota_atual,
        distancias
    )

    if len(rota_atual) <= 3:
        return (
            rota_atual,
            custo_atual
        )

    melhor_rota = rota_atual.copy()

    melhor_custo = custo_atual

    temperatura = temperatura_inicial

    for _ in range(iter_max):

        if temperatura <= temperatura_minima:
            break

        vizinho = gerar_vizinho(
            rota_atual,
            rng
        )

        custo_vizinho = custo_rota(
            vizinho,
            distancias
        )

        delta = (
            custo_vizinho -
            custo_atual
        )

        if (
            delta < 0
            or
            rng.random() <
            np.exp(
                -delta /
                temperatura
            )
        ):

            rota_atual = vizinho

            custo_atual = custo_vizinho

            if custo_atual < melhor_custo:

                melhor_rota = (
                    rota_atual.copy()
                )

                melhor_custo = (
                    custo_atual
                )

        temperatura *= (
            taxa_resfriamento
        )

    return (
        melhor_rota,
        melhor_custo
    )


# ============================================================
# SIMULATED ANNEALING COM REAQUECIMENTO
# ============================================================

def simulated_annealing_reaquecimento(
    rota_inicial,
    distancias,
    rng,
    temperatura_inicial=1000,
    taxa_resfriamento=0.995,
    temperatura_minima=1e-3,
    iter_max=600,
    reaquecimento=3
):

    rota_atual = rota_inicial.copy()

    custo_atual = custo_rota(
        rota_atual,
        distancias
    )

    if len(rota_atual) <= 3:
        return (
            rota_atual,
            custo_atual,
            0
        )

    melhor_rota = rota_atual.copy()

    melhor_custo = custo_atual

    temperatura = temperatura_inicial

    reaquecimentos_realizados = 0

    for _ in range(iter_max):

        # ----------------------------------------------------
        # Reaquecimento
        # ----------------------------------------------------

        if temperatura <= temperatura_minima:

            if (
                reaquecimentos_realizados
                >= reaquecimento
            ):
                break

            temperatura = temperatura_inicial

            reaquecimentos_realizados += 1

        # ----------------------------------------------------
        # Gerar vizinho
        # ----------------------------------------------------

        vizinho = gerar_vizinho(
            rota_atual,
            rng
        )

        custo_vizinho = custo_rota(
            vizinho,
            distancias
        )

        delta = (
            custo_vizinho -
            custo_atual
        )

        # ----------------------------------------------------
        # Critério de aceitação
        # ----------------------------------------------------

        if (
            delta < 0
            or
            rng.random() <
            np.exp(
                -delta /
                temperatura
            )
        ):

            rota_atual = vizinho

            custo_atual = custo_vizinho

            if custo_atual < melhor_custo:

                melhor_rota = (
                    rota_atual.copy()
                )

                melhor_custo = (
                    custo_atual
                )

        # ----------------------------------------------------
        # Resfriamento
        # ----------------------------------------------------

        temperatura *= (
            taxa_resfriamento
        )

    return (
        melhor_rota,
        melhor_custo,
        reaquecimentos_realizados
    )


# ============================================================
# REDUZIR PROBLEMA EM MEMÓRIA
# ============================================================

def reduzir_problema(
    problema,
    quantidade_cidades
):

    cidades = (
        problema["cities"]
        [:quantidade_cidades]
    )

    coordenadas = (
        problema["coordinates"]
        [:quantidade_cidades]
    )

    matriz_original = (
        problema["distance_matrix"]
    )

    matriz = [
        linha[:quantidade_cidades]
        for linha in
        matriz_original[
            :quantidade_cidades
        ]
    ]

    problema_reduzido = {
        "name": problema.get(
            "name",
            "TSP"
        ),

        "cities": cidades.copy(),

        "coordinates": coordenadas.copy(),

        "distance_matrix": matriz
    }

    return problema_reduzido


# ============================================================
# GERAR ROTA INICIAL
# ============================================================

def gerar_rota_inicial(
    cidades,
    inicio,
    rng
):

    rota = list(cidades)

    rota.remove(inicio)

    rng.shuffle(rota)

    rota = (
        [inicio]
        +
        rota
        +
        [inicio]
    )

    return rota


# ============================================================
# GERADOR DE SEEDS
# ============================================================

if SEED_EXPERIMENTO is None:

    rng_seeds = np.random.default_rng()

else:

    rng_seeds = np.random.default_rng(
        SEED_EXPERIMENTO
    )


# ============================================================
# PREPARAR RESULTADOS
# ============================================================

resultados = []

cabecalho = [
    "quantidade_cidades",
    "iteracao",
    "execucao",
    "algoritmo",
    "seed",
    "custo",
    "tempo_ms",
    "reaquecimentos"
]


# ============================================================
# INFORMAÇÕES
# ============================================================

quantidade_original = len(
    tsp_original["cities"]
)

print("=" * 70)
print("EXPERIMENTO - BUSCAS LOCAIS NO TSP")
print("=" * 70)

print(
    f"\nProblema original: "
    f"{tsp_original.get('name', 'TSP')}"
)

print(
    f"Cidades iniciais: "
    f"{quantidade_original}"
)

print(
    f"Execuções por tamanho: "
    f"{N_EXECUCOES}"
)

print(
    f"Seed do experimento: "
    f"{SEED_EXPERIMENTO}"
)

print(
    f"Arquivo de resultados: "
    f"{ARQUIVO_RESULTADOS}"
)


# ============================================================
# EXPERIMENTO
# ============================================================

iteracao = 1

for quantidade_cidades in range(
    quantidade_original,
    1,
    -1
):

    # --------------------------------------------------------
    # Cria uma cópia independente em memória.
    # --------------------------------------------------------

    problema = reduzir_problema(
        tsp_original,
        quantidade_cidades
    )

    cidades = problema["cities"]

    distancias = (
        problema["distance_matrix"]
    )

    # --------------------------------------------------------
    # N execuções
    # --------------------------------------------------------

    for execucao in range(
        1,
        N_EXECUCOES + 1
    ):

        # ====================================================
        # SEEDS DESTA EXECUÇÃO
        # ====================================================

        # Uma seed determina a rota inicial.
        #
        # Outras duas determinam o comportamento
        # aleatório dos algoritmos SA.
        #
        # Assim, os algoritmos não compartilham
        # o mesmo gerador aleatório.

        seed_rota = int(
            rng_seeds.integers(
                0,
                2**63 - 1
            )
        )

        seed_sa = int(
            rng_seeds.integers(
                0,
                2**63 - 1
            )
        )

        seed_sa_r = int(
            rng_seeds.integers(
                0,
                2**63 - 1
            )
        )

        rng_rota = np.random.default_rng(
            seed_rota
        )

        rng_sa = np.random.default_rng(
            seed_sa
        )

        rng_sa_r = np.random.default_rng(
            seed_sa_r
        )

        # ====================================================
        # ROTA INICIAL
        # ====================================================

        # Os três algoritmos recebem exatamente
        # esta mesma rota inicial.

        rota_inicial = (
            gerar_rota_inicial(
                cidades,
                INICIO,
                rng_rota
            )
        )

        # ====================================================
        # HILL CLIMBING
        # ====================================================

        inicio_tempo = (
            time.perf_counter()
        )

        (
            rota_hc,
            custo_hc
        ) = subida_de_encosta(
            rota_inicial,
            distancias
        )

        tempo_hc = (
            time.perf_counter() -
            inicio_tempo
        ) * 1000.0

        resultados.append({
            "quantidade_cidades":
                quantidade_cidades,

            "iteracao":
                iteracao,

            "execucao":
                execucao,

            "algoritmo":
                "Hill Climbing",

            "seed":
                seed_rota,

            "custo":
                custo_hc,

            "tempo_ms":
                tempo_hc,

            "reaquecimentos":
                0
        })

        # ====================================================
        # SIMULATED ANNEALING
        # ====================================================

        inicio_tempo = (
            time.perf_counter()
        )

        (
            rota_sa,
            custo_sa
        ) = simulated_annealing(
            rota_inicial,
            distancias,
            rng_sa
        )

        tempo_sa = (
            time.perf_counter() -
            inicio_tempo
        ) * 1000.0

        resultados.append({
            "quantidade_cidades":
                quantidade_cidades,

            "iteracao":
                iteracao,

            "execucao":
                execucao,

            "algoritmo":
                "Simulated Annealing",

            "seed":
                seed_sa,

            "custo":
                custo_sa,

            "tempo_ms":
                tempo_sa,

            "reaquecimentos":
                0
        })

        # ====================================================
        # SA COM REAQUECIMENTO
        # ====================================================

        inicio_tempo = (
            time.perf_counter()
        )

        (
            rota_sa_r,
            custo_sa_r,
            reaquecimentos
        ) = simulated_annealing_reaquecimento(
            rota_inicial,
            distancias,
            rng_sa_r
        )

        tempo_sa_r = (
            time.perf_counter() -
            inicio_tempo
        ) * 1000.0

        resultados.append({
            "quantidade_cidades":
                quantidade_cidades,

            "iteracao":
                iteracao,

            "execucao":
                execucao,

            "algoritmo":
                "SA com Reaquecimento",

            "seed":
                seed_sa_r,

            "custo":
                custo_sa_r,

            "tempo_ms":
                tempo_sa_r,

            "reaquecimentos":
                reaquecimentos
        })

    iteracao += 1


# ============================================================
# SALVAR CSV
# ============================================================

with open(
    ARQUIVO_RESULTADOS,
    "w",
    newline="",
    encoding="utf-8"
) as arquivo:

    escritor = csv.DictWriter(
        arquivo,
        fieldnames=cabecalho
    )

    escritor.writeheader()

    escritor.writerows(
        resultados
    )


# ============================================================
# FINALIZAÇÃO
# ============================================================

print(
    "\n" +
    "=" * 70
)

print(
    "EXPERIMENTO FINALIZADO"
)

print(
    "=" * 70
)

print(
    f"Total de registros: "
    f"{len(resultados)}"
)

print(
    f"Resultados salvos em: "
    f"{ARQUIVO_RESULTADOS}"
)
