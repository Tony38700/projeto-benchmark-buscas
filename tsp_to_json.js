const fs = require('fs');
const path = require('path');

// ============================================================
// CONFIGURAÇÃO
// ============================================================

const inputFile = process.argv[2];

const inputDirectory = './problemas';
const outputDirectory = './problemas_json';

if (!inputFile) {
    console.error('Uso: node tsp_to_json.js arquivo.tsp');
    process.exit(1);
}

// O usuário fornece apenas o nome do arquivo.
// O script procura automaticamente dentro de ./problemas.
const inputPath = path.join(inputDirectory, path.basename(inputFile));

if (!fs.existsSync(inputPath)) {
    console.error(`Arquivo não encontrado: ${inputPath}`);
    process.exit(1);
}

const content = fs.readFileSync(inputPath, 'utf8');
const lines = content.split(/\r?\n/);

// ============================================================
// LEITURA DO CABEÇALHO
// ============================================================

const header = {};

for (const line of lines) {
    const trimmed = line.trim();

    if (trimmed === 'EOF') {
        break;
    }

    const match = trimmed.match(/^([^:]+)\s*:\s*(.*)$/);

    if (match) {
        const key = match[1].trim();
        const value = match[2].trim();

        header[key] = value;
    }
}

const name = header.NAME || path.basename(inputPath, path.extname(inputPath));

const dimension = Number(header.DIMENSION);
const edgeWeightType = header.EDGE_WEIGHT_TYPE || null;
const edgeWeightFormat = header.EDGE_WEIGHT_FORMAT || null;

if (!dimension || dimension <= 0) {
    throw new Error('DIMENSION não encontrada ou inválida.');
}

// ============================================================
// LOCALIZAR SEÇÕES
// ============================================================

function findSection(sectionName) {
    const index = lines.findIndex(
        (line) => line.trim().toUpperCase() === sectionName,
    );

    if (index === -1) {
        return null;
    }

    const values = [];

    for (let i = index + 1; i < lines.length; i++) {
        const line = lines[i].trim();

        if (!line || line === 'EOF') {
            continue;
        }

        const upper = line.toUpperCase();

        if (upper.endsWith('_SECTION') || upper === 'EOF') {
            break;
        }

        values.push(line);
    }

    return values;
}

// ============================================================
// CONVERTER TEXTO EM NÚMEROS
// ============================================================

function numericValues(section) {
    if (!section) {
        return [];
    }

    return section.join(' ').split(/\s+/).filter(Boolean).map(Number);
}

// ============================================================
// MATRIZ VAZIA
// ============================================================

function createMatrix(n) {
    return Array.from({ length: n }, () => Array(n).fill(0));
}

// ============================================================
// MATRIZ EXPLÍCITA
// ============================================================

function buildExplicitMatrix(values, n, format) {
    const matrix = createMatrix(n);
    let index = 0;

    const f = (format || 'FULL_MATRIX').toUpperCase();

    // --------------------------------------------------------
    // FULL_MATRIX
    // --------------------------------------------------------

    if (f === 'FULL_MATRIX') {
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                matrix[i][j] = values[index++];
            }
        }

        return matrix;
    }

    // --------------------------------------------------------
    // LOWER_DIAG_ROW
    // --------------------------------------------------------

    if (f === 'LOWER_DIAG_ROW') {
        for (let i = 0; i < n; i++) {
            for (let j = 0; j <= i; j++) {
                const value = values[index++];

                matrix[i][j] = value;
                matrix[j][i] = value;
            }
        }

        return matrix;
    }

    // --------------------------------------------------------
    // LOWER_ROW
    // --------------------------------------------------------

    if (f === 'LOWER_ROW') {
        for (let i = 1; i < n; i++) {
            for (let j = 0; j < i; j++) {
                const value = values[index++];

                matrix[i][j] = value;
                matrix[j][i] = value;
            }
        }

        return matrix;
    }

    // --------------------------------------------------------
    // UPPER_DIAG_ROW
    // --------------------------------------------------------

    if (f === 'UPPER_DIAG_ROW') {
        for (let i = 0; i < n; i++) {
            for (let j = i; j < n; j++) {
                const value = values[index++];

                matrix[i][j] = value;
                matrix[j][i] = value;
            }
        }

        return matrix;
    }

    // --------------------------------------------------------
    // UPPER_ROW
    // --------------------------------------------------------

    if (f === 'UPPER_ROW') {
        for (let i = 0; i < n - 1; i++) {
            for (let j = i + 1; j < n; j++) {
                const value = values[index++];

                matrix[i][j] = value;
                matrix[j][i] = value;
            }
        }

        return matrix;
    }

    // --------------------------------------------------------
    // LOWER_DIAG_COL
    // --------------------------------------------------------

    if (f === 'LOWER_DIAG_COL') {
        for (let j = 0; j < n; j++) {
            for (let i = j; i < n; i++) {
                const value = values[index++];

                matrix[i][j] = value;
                matrix[j][i] = value;
            }
        }

        return matrix;
    }

    // --------------------------------------------------------
    // LOWER_COL
    // --------------------------------------------------------

    if (f === 'LOWER_COL') {
        for (let j = 0; j < n - 1; j++) {
            for (let i = j + 1; i < n; i++) {
                const value = values[index++];

                matrix[i][j] = value;
                matrix[j][i] = value;
            }
        }

        return matrix;
    }

    // --------------------------------------------------------
    // UPPER_DIAG_COL
    // --------------------------------------------------------

    if (f === 'UPPER_DIAG_COL') {
        for (let j = 0; j < n; j++) {
            for (let i = 0; i <= j; i++) {
                const value = values[index++];

                matrix[i][j] = value;
                matrix[j][i] = value;
            }
        }

        return matrix;
    }

    // --------------------------------------------------------
    // UPPER_COL
    // --------------------------------------------------------

    if (f === 'UPPER_COL') {
        for (let j = 1; j < n; j++) {
            for (let i = 0; i < j; i++) {
                const value = values[index++];

                matrix[i][j] = value;
                matrix[j][i] = value;
            }
        }

        return matrix;
    }

    throw new Error(`EDGE_WEIGHT_FORMAT não suportado: ${format}`);
}

// ============================================================
// DISTÂNCIA EUCLIDIANA
// ============================================================

function euclideanDistance(a, b) {
    const dx = a.x - b.x;
    const dy = a.y - b.y;

    return Math.sqrt(dx * dx + dy * dy);
}

// ============================================================
// DISTÂNCIA EUC_2D DO TSPLIB
// ============================================================

function euc2dDistance(a, b) {
    return Math.round(euclideanDistance(a, b));
}

// ============================================================
// DISTÂNCIA CEIL_2D
// ============================================================

function ceil2dDistance(a, b) {
    return Math.ceil(euclideanDistance(a, b));
}

// ============================================================
// DISTÂNCIA ATT
// ============================================================

function attDistance(a, b) {
    const dx = a.x - b.x;
    const dy = a.y - b.y;

    const rij = Math.sqrt((dx * dx + dy * dy) / 10.0);

    const tij = Math.round(rij);

    if (tij < rij) {
        return tij + 1;
    }

    return tij;
}

// ============================================================
// DISTÂNCIA GEO
// ============================================================

function geoToRadians(value) {
    const degrees = Math.floor(value);
    const minutes = value - degrees;

    return (Math.PI * (degrees + (5.0 * minutes) / 3.0)) / 180.0;
}

function geoDistance(a, b) {
    const latitudeA = geoToRadians(a.x);
    const longitudeA = geoToRadians(a.y);

    const latitudeB = geoToRadians(b.x);
    const longitudeB = geoToRadians(b.y);

    const RRR = 6378.388;

    const q1 = Math.cos(longitudeA - longitudeB);
    const q2 = Math.cos(latitudeA - latitudeB);
    const q3 = Math.cos(latitudeA + latitudeB);

    const dij = Math.floor(
        RRR * Math.acos(0.5 * ((1 + q1) * q2 - (1 - q1) * q3)) + 1,
    );

    return dij;
}

// ============================================================
// LER COORDENADAS
// ============================================================

function readCoordinates() {
    let section = findSection('NODE_COORD_SECTION');

    if (!section) {
        section = findSection('DISPLAY_DATA_SECTION');
    }

    if (!section) {
        return null;
    }

    const coordinates = [];

    for (const line of section) {
        const parts = line.trim().split(/\s+/);

        if (parts.length < 3) {
            continue;
        }

        const id = Number(parts[0]);
        const x = Number(parts[1]);
        const y = Number(parts[2]);

        if (Number.isNaN(id) || Number.isNaN(x) || Number.isNaN(y)) {
            continue;
        }

        coordinates.push({
            id: id,
            x: x,
            y: y,
        });
    }

    coordinates.sort((a, b) => a.id - b.id);

    if (coordinates.length !== dimension) {
        throw new Error(
            `Foram encontradas ${coordinates.length} coordenadas, ` +
                `mas DIMENSION indica ${dimension}.`,
        );
    }

    return coordinates;
}

// ============================================================
// MATRIZ A PARTIR DAS COORDENADAS
// ============================================================

function buildCoordinateMatrix(coordinates, type) {
    const matrix = createMatrix(dimension);

    const t = (type || '').toUpperCase();

    for (let i = 0; i < dimension; i++) {
        for (let j = i + 1; j < dimension; j++) {
            let distance;

            switch (t) {
                case 'EUC_2D':
                    distance = euc2dDistance(coordinates[i], coordinates[j]);
                    break;

                case 'CEIL_2D':
                    distance = ceil2dDistance(coordinates[i], coordinates[j]);
                    break;

                case 'ATT':
                    distance = attDistance(coordinates[i], coordinates[j]);
                    break;

                case 'GEO':
                    distance = geoDistance(coordinates[i], coordinates[j]);
                    break;

                case 'EUC_3D':
                    throw new Error('EUC_3D ainda não está implementado.');

                case 'MAN_2D':
                    distance = Math.round(
                        Math.abs(coordinates[i].x - coordinates[j].x) +
                            Math.abs(coordinates[i].y - coordinates[j].y),
                    );
                    break;

                case 'MAX_2D':
                    distance = Math.round(
                        Math.max(
                            Math.abs(coordinates[i].x - coordinates[j].x),
                            Math.abs(coordinates[i].y - coordinates[j].y),
                        ),
                    );
                    break;

                default:
                    throw new Error(
                        `EDGE_WEIGHT_TYPE não suportado para coordenadas: ${type}`,
                    );
            }

            matrix[i][j] = distance;
            matrix[j][i] = distance;
        }
    }

    return matrix;
}

// ============================================================
// PROCESSAMENTO
// ============================================================

const edgeWeightSection = findSection('EDGE_WEIGHT_SECTION');

const coordinates = readCoordinates();

let distanceMatrix;

// Se existe matriz explícita, ela tem prioridade.
if (edgeWeightSection) {
    const values = numericValues(edgeWeightSection);

    distanceMatrix = buildExplicitMatrix(values, dimension, edgeWeightFormat);
} else if (coordinates) {
    distanceMatrix = buildCoordinateMatrix(coordinates, edgeWeightType);
} else {
    throw new Error(
        'O arquivo não possui EDGE_WEIGHT_SECTION ' +
            'nem coordenadas utilizáveis.',
    );
}

// ============================================================
// PREPARAR DADOS PARA PYTHON
// ============================================================

const cities = Array.from({ length: dimension }, (_, i) => i);

const result = {
    name: name,
    type: header.TYPE || 'TSP',
    dimension: dimension,

    edge_weight_type: edgeWeightType,
    edge_weight_format: edgeWeightFormat,

    cities: cities,

    coordinates: coordinates
        ? coordinates.map((city, index) => ({
              id: index,
              original_id: city.id,
              x: city.x,
              y: city.y,
          }))
        : null,

    distance_matrix: distanceMatrix,
};

// ============================================================
// SALVAR JSON
// ============================================================

// Cria ./problemas_json caso ainda não exista.
fs.mkdirSync(outputDirectory, {
    recursive: true,
});

// Mantém o mesmo nome do arquivo,
// trocando apenas .tsp por .json.
const outputFile = path.join(
    outputDirectory,
    `${path.basename(inputPath, path.extname(inputPath))}.json`,
);

// writeFileSync sobrescreve automaticamente
// caso o arquivo já exista.
fs.writeFileSync(outputFile, JSON.stringify(result, null, 2), 'utf8');

// ============================================================
// RESULTADO
// ============================================================

console.log('Conversão concluída.');
console.log(`Arquivo de entrada: ${inputPath}`);
console.log(`Arquivo de saída: ${outputFile}`);
console.log(`Problema: ${name}`);
console.log(`Cidades: ${dimension}`);
console.log(`Tipo de distância: ${edgeWeightType}`);

if (edgeWeightFormat) {
    console.log(`Formato da matriz: ${edgeWeightFormat}`);
}

console.log('Matriz de distâncias gerada para uso no Python.');
