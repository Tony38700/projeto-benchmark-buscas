import csv
import heapq
import time
import numpy as np
import os


# ============================================================
# CONFIGURAÇÕES
# ============================================================

# (tamanho_do_tabuleiro, quantidade_de_tabuleiros)
CONFIGURACOES = [
    #(3, 100),
    (4, 1),
]

ARQUIVO_RESULTADOS = "resultados_heuristicas.csv"

# Seed mestre do experimento.
#
# Use um número para tornar o experimento reproduzível.
# Use None para gerar um experimento diferente a cada execução.
SEED_EXPERIMENTO = None

# Valor usado para representar o espaço vazio.
VAZIO = 0


# ============================================================
# HEURÍSTICAS
# ============================================================

def heuristica_manhattan(estado, objetivo, n):
    """
    Soma das distâncias Manhattan de cada peça até sua
    posição no estado objetivo.
    O espaço vazio não é considerado.
    """

    posicoes_objetivo = {}

    for indice, valor in enumerate(objetivo):
        if valor != VAZIO:
            posicoes_objetivo[valor] = (
                indice // n,
                indice % n
            )

    distancia = 0

    for indice, valor in enumerate(estado):

        if valor == VAZIO:
            continue

        linha = indice // n
        coluna = indice % n

        linha_obj, coluna_obj = posicoes_objetivo[valor]

        distancia += (
            abs(linha - linha_obj)
            +
            abs(coluna - coluna_obj)
        )

    return distancia


# ============================================================
# SOLUBILIDADE
# ============================================================

def contar_inversoes(estado):
    """
    Conta as inversões ignorando o espaço vazio.
    """

    valores = [
        valor
        for valor in estado
        if valor != VAZIO
    ]

    inversoes = 0

    for i in range(len(valores)):
        for j in range(i + 1, len(valores)):

            if valores[i] > valores[j]:
                inversoes += 1

    return inversoes


def eh_soluvel(estado, n):
    """
    Verifica a solubilidade de um puzzle N x N.

    Para largura ímpar:
        inversões devem ser pares.

    Para largura par:
        a paridade das inversões depende da linha do vazio
        contada a partir de baixo.
    """

    inversoes = contar_inversoes(estado)

    if n % 2 == 1:
        return inversoes % 2 == 0

    indice_vazio = estado.index(VAZIO)

    linha_vazio = indice_vazio // n

    linha_do_fundo = n - linha_vazio

    if linha_do_fundo % 2 == 0:
        return inversoes % 2 == 1

    return inversoes % 2 == 0


# ============================================================
# GERAÇÃO DE TABULEIROS
# ============================================================

def gerar_tabuleiro_aleatorio(n, rng):
    """
    Posiciona todas as peças aleatoriamente.
    O tabuleiro só é aceito posteriormente se for solúvel.
    """

    estado = np.arange(n * n)

    rng.shuffle(estado)

    return tuple(int(valor) for valor in estado)


def gerar_tabuleiros_soluveis(n, quantidade, rng):
    """
    Gera estados aleatórios até obter a quantidade solicitada
    de tabuleiros solúveis.
    """

    tabuleiros = []

    while len(tabuleiros) < quantidade:

        estado = gerar_tabuleiro_aleatorio(
            n,
            rng
        )

        if eh_soluvel(estado, n):
            tabuleiros.append(estado)

    return tabuleiros


# ============================================================
# MOVIMENTOS
# ============================================================

def gerar_vizinhos(estado, n):
    """
    Gera todos os estados obtidos movendo o espaço vazio
    para uma posição adjacente.
    """

    indice_vazio = estado.index(VAZIO)

    linha = indice_vazio // n
    coluna = indice_vazio % n

    movimentos = []

    if linha > 0:
        movimentos.append(
            indice_vazio - n
        )

    if linha < n - 1:
        movimentos.append(
            indice_vazio + n
        )

    if coluna > 0:
        movimentos.append(
            indice_vazio - 1
        )

    if coluna < n - 1:
        movimentos.append(
            indice_vazio + 1
        )

    for novo_indice in movimentos:

        vizinho = list(estado)

        vizinho[indice_vazio], vizinho[novo_indice] = (
            vizinho[novo_indice],
            vizinho[indice_vazio]
        )

        yield tuple(vizinho)


# ============================================================
# ESTADO OBJETIVO
# ============================================================

def criar_objetivo(n):
    return tuple(range(1, n * n)) + (VAZIO,)


# ============================================================
# RECONSTRUIR CAMINHO
# ============================================================

def reconstruir_caminho(
    pais,
    estado_inicial,
    estado_final
):

    caminho = []

    atual = estado_final

    while atual != estado_inicial:

        caminho.append(atual)

        atual = pais[atual]

    caminho.append(estado_inicial)

    caminho.reverse()

    return caminho


# ============================================================
# BUSCA GULOSA
# ============================================================

def busca_gulosa(
    estado_inicial,
    objetivo,
    n
):

    fila = []

    contador = 0

    h = heuristica_manhattan(
        estado_inicial,
        objetivo,
        n
    )

    heapq.heappush(
        fila,
        (
            h,
            contador,
            estado_inicial
        )
    )

    visitados = {
        estado_inicial
    }

    pais = {}

    while fila:

        _, _, atual = heapq.heappop(
            fila
        )

        if atual == objetivo:

            caminho = reconstruir_caminho(
                pais,
                estado_inicial,
                objetivo
            )

            return caminho

        for vizinho in gerar_vizinhos(
            atual,
            n
        ):

            if vizinho in visitados:
                continue

            visitados.add(vizinho)

            pais[vizinho] = atual

            contador += 1

            h = heuristica_manhattan(
                vizinho,
                objetivo,
                n
            )

            heapq.heappush(
                fila,
                (
                    h,
                    contador,
                    vizinho
                )
            )

    return None


# ============================================================
# A*
# ============================================================

def busca_a_estrela(
    estado_inicial,
    objetivo,
    n
):

    fila = []

    contador = 0

    g_inicial = 0

    h_inicial = heuristica_manhattan(
        estado_inicial,
        objetivo,
        n
    )

    heapq.heappush(
        fila,
        (
            g_inicial + h_inicial,
            contador,
            estado_inicial
        )
    )

    custos = {
        estado_inicial: 0
    }

    pais = {}

    while fila:

        _, _, atual = heapq.heappop(
            fila
        )

        if atual == objetivo:

            caminho = reconstruir_caminho(
                pais,
                estado_inicial,
                objetivo
            )

            return caminho

        g_atual = custos[atual]

        for vizinho in gerar_vizinhos(
            atual,
            n
        ):

            novo_g = g_atual + 1

            if (
                vizinho not in custos
                or
                novo_g < custos[vizinho]
            ):

                custos[vizinho] = novo_g

                pais[vizinho] = atual

                h = heuristica_manhattan(
                    vizinho,
                    objetivo,
                    n
                )

                f = novo_g + h

                contador += 1

                heapq.heappush(
                    fila,
                    (
                        f,
                        contador,
                        vizinho
                    )
                )

    return None


# ============================================================
# IDA*
# ============================================================

def ida_estrela(
    estado_inicial,
    objetivo,
    n
):

    limite = heuristica_manhattan(
        estado_inicial,
        objetivo,
        n
    )

    caminho = [
        estado_inicial
    ]

    def busca(profundidade, limite_atual):

        estado = caminho[-1]

        h = heuristica_manhattan(
            estado,
            objetivo,
            n
        )

        f = profundidade + h

        if f > limite_atual:
            return f

        if estado == objetivo:
            return True

        menor_limite = float("inf")

        for vizinho in gerar_vizinhos(
            estado,
            n
        ):

            if vizinho in caminho:
                continue

            caminho.append(vizinho)

            resultado = busca(
                profundidade + 1,
                limite_atual
            )

            if resultado is True:
                return True

            if resultado < menor_limite:
                menor_limite = resultado

            caminho.pop()

        return menor_limite

    while True:

        resultado = busca(
            0,
            limite
        )

        if resultado is True:
            return caminho.copy()

        if resultado == float("inf"):
            return None

        limite = resultado


# ============================================================
# EXECUTAR UMA BUSCA
# ============================================================

def executar_busca(
    nome,
    funcao,
    estado_inicial,
    objetivo,
    n
):

    inicio = time.perf_counter()

    caminho = funcao(
        estado_inicial,
        objetivo,
        n
    )

    tempo = (
        time.perf_counter()
        -
        inicio
    ) * 1000.0

    if caminho is None:

        movimentos = None

        resolvido = False

    else:

        movimentos = len(caminho) - 1

        resolvido = True

    return {
        "algoritmo": nome,
        "resolvido": resolvido,
        "movimentos": movimentos,
        "tempo_ms": tempo
    }


# ============================================================
# EXECUÇÃO DO EXPERIMENTO
# ============================================================

print("=" * 70)
print("EXPERIMENTO - BUSCAS HEURÍSTICAS NO 8-PUZZLE")
print("=" * 70)

print(
    f"\nConfigurações: {CONFIGURACOES}"
)

print(
    f"Seed do experimento: "
    f"{SEED_EXPERIMENTO}"
)

print(
    f"Arquivo de resultados: "
    f"{ARQUIVO_RESULTADOS}"
)


if SEED_EXPERIMENTO is None:

    rng = np.random.default_rng()

else:

    rng = np.random.default_rng(
        SEED_EXPERIMENTO
    )


resultados = []

cabecalho = [
    "tamanho_puzzle",
    "quantidade_tabuleiros",
    "execucao",
    "algoritmo",
    "seed",
    "resolvido",
    "movimentos",
    "tempo_ms"
]


# ============================================================
# PROCESSAR CADA CONFIGURAÇÃO
# ============================================================

for n, quantidade in CONFIGURACOES:

    print("\n" + "=" * 70)

    print(
        f"PUZZLE {n}x{n}"
    )

    print(
        f"Tabuleiros: {quantidade}"
    )

    print("=" * 70)

    objetivo = criar_objetivo(n)

    # --------------------------------------------------------
    # Gerar somente tabuleiros solúveis
    # --------------------------------------------------------

    tabuleiros = gerar_tabuleiros_soluveis(
        n,
        quantidade,
        rng
    )

    # --------------------------------------------------------
    # Resolver cada tabuleiro
    # --------------------------------------------------------

    for indice, estado_inicial in enumerate(
        tabuleiros,
        start=1
    ):

        seed_tabuleiro = int(
            rng.integers(
                0,
                2**63 - 1
            )
        )

        print(
            f"Tabuleiro "
            f"{indice}/{quantidade}"
        )

        # ====================================================
        # GULOSA
        # ====================================================

        resultado = executar_busca(
            "Gulosa",
            busca_gulosa,
            estado_inicial,
            objetivo,
            n
        )

        resultados.append({
            "tamanho_puzzle": n,
            "quantidade_tabuleiros": quantidade,
            "execucao": indice,
            "algoritmo": resultado["algoritmo"],
            "seed": seed_tabuleiro,
            "resolvido": resultado["resolvido"],
            "movimentos": resultado["movimentos"],
            "tempo_ms": resultado["tempo_ms"]
        })

        # ====================================================
        # A*
        # ====================================================

        resultado = executar_busca(
            "A*",
            busca_a_estrela,
            estado_inicial,
            objetivo,
            n
        )

        resultados.append({
            "tamanho_puzzle": n,
            "quantidade_tabuleiros": quantidade,
            "execucao": indice,
            "algoritmo": resultado["algoritmo"],
            "seed": seed_tabuleiro,
            "resolvido": resultado["resolvido"],
            "movimentos": resultado["movimentos"],
            "tempo_ms": resultado["tempo_ms"]
        })

        # ====================================================
        # IDA*
        # ====================================================

        resultado = executar_busca(
            "IDA*",
            ida_estrela,
            estado_inicial,
            objetivo,
            n
        )

        resultados.append({
            "tamanho_puzzle": n,
            "quantidade_tabuleiros": quantidade,
            "execucao": indice,
            "algoritmo": resultado["algoritmo"],
            "seed": seed_tabuleiro,
            "resolvido": resultado["resolvido"],
            "movimentos": resultado["movimentos"],
            "tempo_ms": resultado["tempo_ms"]
        })


# ============================================================
# SALVAR CSV
# ============================================================

arquivo_existe = os.path.exists(ARQUIVO_RESULTADOS)

with open(
    ARQUIVO_RESULTADOS,
    "a",
    newline="",
    encoding="utf-8"
) as arquivo:

    escritor = csv.DictWriter(
        arquivo,
        fieldnames=cabecalho
    )

    if not arquivo_existe:
        escritor.writeheader()

    escritor.writerows(resultados)


# ============================================================
# RESULTADOS FINAIS
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENTO FINALIZADO")
print("=" * 70)

print(
    f"Total de registros: "
    f"{len(resultados)}"
)

print(
    f"Resultados salvos em: "
    f"{ARQUIVO_RESULTADOS}"
)

print("\nResumo:")

for n, quantidade in CONFIGURACOES:

    subconjunto = [
        resultado
        for resultado in resultados
        if resultado["tamanho_puzzle"] == n
    ]

    print(
        f"\n{n}x{n} "
        f"({quantidade} tabuleiros)"
    )

    for algoritmo in [
        "Gulosa",
        "A*",
        "IDA*"
    ]:

        registros = [
            resultado
            for resultado in subconjunto
            if resultado["algoritmo"] == algoritmo
        ]

        tempos = [
            resultado["tempo_ms"]
            for resultado in registros
        ]

        movimentos = [
            resultado["movimentos"]
            for resultado in registros
            if resultado["movimentos"] is not None
        ]

        print(
            f"  {algoritmo}: "
            f"tempo médio = "
            f"{np.mean(tempos):.3f} ms | "
            f"movimentos médios = "
            f"{np.mean(movimentos):.2f}"
        )
