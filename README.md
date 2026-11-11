# Coeficiente de Variação com OpenMP

Exercício final de **Programação Paralela**.

Programa em C++ que calcula o **coeficiente de variação (CV)** da estatura e do peso de 1000 pessoas, com os dados agrupados em **intervalos de classe** (8 cm para a estatura e 4 kg para o peso), usando **OpenMP** para paralelizar os cálculos. O programa também compara o tempo da versão sequencial com o da versão paralela.

**Autor:** Felipe Oliveira Alves

## Conteúdo do repositório

| Arquivo | Descrição |
|---|---|
| `desafioopenmp.cpp` | Código-fonte (versões sequencial e paralela, gerador de dados e benchmark) |
| `dados.txt` | Arquivo de exemplo com 1000 pessoas (dados sintéticos) |
| `relatorio_openmp.pdf` | Relatório: partes paralelizadas, resultados e análise de desempenho |

## Requisitos

- Compilador C++17 com suporte a OpenMP (testado com `g++`).
- No Windows, use `g++` via MinGW-w64 ou MSYS2. O MSVC não serve, porque seu OpenMP 2.0 não suporta as reduções de mínimo/máximo e de vetor usadas no código.

## Compilação

```bash
g++ -O2 -fopenmp -std=c++17 desafioopenmp.cpp -o desafioopenmp
```

## Execução

```bash
# 1) Gerar um arquivo de teste com 1000 pessoas (opcional, já existe um dados.txt)
./desafioopenmp --gerar dados.txt

# 2) Calcular o CV e comparar sequencial x paralelo
./desafioopenmp dados.txt

# 3) Benchmark com volume maior (replica os dados R vezes)
./desafioopenmp dados.txt --replicar 10000
./desafioopenmp dados.txt --replicar 50000
```

A opção `--replicar` repete os 1000 registros R vezes. O CV não muda, só aumenta o volume de trabalho, o que permite ver o ganho do paralelismo.

Para limitar o número máximo de threads testadas:

```bash
# Linux / macOS / Git Bash
OMP_NUM_THREADS=4 ./desafioopenmp dados.txt --replicar 10000
```

```powershell
# Windows (PowerShell)
$env:OMP_NUM_THREADS=4; .\desafioopenmp.exe dados.txt --replicar 10000
```

### Formato do arquivo de entrada

Uma pessoa por linha, com a estatura (cm) e o peso (kg) separados por espaço:

```
165.0 75.7
172.5 68.3
```

## Como funciona

1. **Leitura** do arquivo (sequencial).
2. **Mínimo e máximo** de cada variável, para definir as classes: o limite inferior da primeira classe é o mínimo arredondado para baixo, e o número de classes é a parte inteira de `(máx - li) / h`, mais 1.
3. **Histograma** de frequências por classe.
4. **Média ponderada** pelos pontos médios: `μ = Σ(xi · fi) / N`.
5. **Desvio padrão populacional**: `σ = √( Σ fi · (xi − μ)² / N )`.
6. **Coeficiente de variação**: `CV = (σ / μ) × 100 %`.

### O que é paralelizado

| Etapa | Diretiva |
|---|---|
| Mínimo e máximo | `parallel for` com `reduction(min, max)` |
| Histograma de frequências | `parallel for` com `reduction(+ : f[:k])` |
| Soma ponderada (média) | `parallel for` com `reduction(+)` |
| Soma dos desvios quadráticos | `parallel for` com `reduction(+)` |

Para **N < 100.000** o programa usa direto a versão sequencial (`LIMIAR_PAR`), porque o custo de entrar nas regiões paralelas supera o trabalho útil. A cada execução, o programa confere que as versões sequencial e paralela dão o mesmo CV.

## Resultados

Com o `dados.txt` de exemplo (1000 pessoas):

| Variável | Classe | Média | Desvio padrão | CV |
|---|---|---|---|---|
| Estatura | 8 cm | 170,3440 cm | 9,4425 cm | **5,5432 %** |
| Peso | 4 kg | 70,4240 kg | 12,4147 kg | **17,6285 %** |

> O `dados.txt` é **sintético** (gerado pelo próprio programa, semente 42), então esses valores valem para esse arquivo. Com outros dados, o CV será diferente.

### Desempenho

Medido em um AMD Ryzen 7 9800X3D (8 núcleos, 16 threads) com Windows 11 25H2. O tempo é a mediana de 7 blocos de execuções, para estatura e peso juntos, sem a leitura do arquivo.

| N | 1 thread | 2 threads | 4 threads | 8 threads | 16 threads |
|---|---|---|---|---|---|
| 1.000 | 0,98x | 0,98x | 1,00x | 0,98x | 0,98x |
| 10 milhões | 1,00x | 1,95x | 3,68x | 4,80x | **5,27x** |
| 50 milhões | 0,99x | 1,25x | 2,37x | 3,54x | **3,73x** |

O speedup é o tempo sequencial dividido pelo tempo paralelo. Com 1000 pessoas o programa roda o código sequencial, por isso fica em torno de 1,0x. Com volumes grandes há ganho real, mas ele é limitado pelos 8 núcleos físicos e, para 50 milhões, provavelmente pela largura de banda da memória. A análise completa está em [`relatorio_openmp.pdf`](relatorio_openmp.pdf).

## Observações

- Os tempos variam com a máquina, o sistema operacional e a carga no momento da execução. Para comparar versões, rode mais de uma vez.
- O timer do `omp_get_wtime()` em alguns sistemas tem resolução de ~1 ms. Por isso o benchmark mede blocos de execuções e divide pelo número de execuções do bloco.
