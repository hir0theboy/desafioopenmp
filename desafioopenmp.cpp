    /*-------------------------------------------------------------------------------------------------------------------------
    * File Name  : desafioopenmp.cpp
    * Author     : Felipe Oliveira Alves
    * Institution: IESB
    * Course     : Parallel and Distributed Programming
    * Objective  : Calcular o coeficiente de variacao (CV) da estatura (classes de 8 cm) e do peso (classes de 4 kg)
    *              de 1000 pessoas, usando dados agrupados em intervalos de classe e OpenMP para paralelizar os
    *              calculos. Compara o tempo da versao sequencial com o da versao paralela.
    * Semester   : 2026/2
    * Version    : 1.1
    *-------------------------------------------------------------------------------------------------------------------------
    *
    * COMO COMPILAR:
    *   g++ -O2 -fopenmp -std=c++17 desafioopenmp.cpp -o desafioopenmp
    *
    * COMO EXECUTAR:
    *   1) Gerar um arquivo de teste com 1000 pessoas (uma por linha: "estatura peso"):
    *        ./desafioopenmp --gerar dados.txt
    *
    *   2) Calcular o CV e comparar sequencial x paralelo:
    *        ./desafioopenmp dados.txt
    *
    *   3) Benchmark com volume maior (replica os dados R vezes; o CV nao muda, so o volume de trabalho):
    *        ./desafioopenmp dados.txt --replicar 10000
    *
    *   4) Escolher o numero maximo de threads (Linux/macOS):
    *        OMP_NUM_THREADS=4 ./desafioopenmp dados.txt --replicar 10000
    *      No Windows (PowerShell):
    *        $env:OMP_NUM_THREADS=4; .\desafioopenmp.exe dados.txt --replicar 10000
    *
    * FORMATO DO ARQUIVO DE ENTRADA:
    *   Cada linha contem a estatura (cm) e o peso (kg) de uma pessoa, separados por espaco. Ex.: 165.0 75.7
    *-------------------------------------------------------------------------------------------------------------------------*/

    #include <omp.h>

    #include <algorithm>
    #include <cmath>
    #include <cstdio>
    #include <cstdlib>
    #include <cstring>
    #include <fstream>
    #include <iostream>
    #include <random>
    #include <string>
    #include <vector>

    using namespace std;

    struct Resultado {
        double li;      // limite inferior da primeira classe
        int k;          // número de classes
        double media;
        double desvio;
        double cv;      // em %
        vector<long> freq;
    };

    // ---------------------------------------------------------------- sequencial
    Resultado calcular_seq(const vector<double>& v, double h) {
        Resultado r;
        const long n = (long)v.size();

        double mn = v[0], mx = v[0];
        for (long i = 1; i < n; i++) {
            mn = min(mn, v[i]);
            mx = max(mx, v[i]);
        }
        r.li = floor(mn);
        r.k = (int)floor((mx - r.li) / h) + 1;
        r.freq.assign(r.k, 0);

        for (long i = 0; i < n; i++) r.freq[(int)((v[i] - r.li) / h)]++;

        double soma = 0.0;
        for (int c = 0; c < r.k; c++) {
            double xi = r.li + (c + 0.5) * h;
            soma += xi * r.freq[c];
        }
        r.media = soma / n;

        double ss = 0.0;
        for (int c = 0; c < r.k; c++) {
            double xi = r.li + (c + 0.5) * h;
            ss += r.freq[c] * (xi - r.media) * (xi - r.media);
        }
        r.desvio = sqrt(ss / n);
        r.cv = r.desvio / r.media * 100.0;
        return r;
    }

    // ------------------------------------------------------------------ paralela
    // Abaixo deste N o custo de criar/sincronizar threads supera o trabalho util,
    // entao os laços rodam em uma unica thread (clausula if do OpenMP).
    const long LIMIAR_PAR = 100000;

    Resultado calcular_par(const vector<double>& v, double h) {
        Resultado r;
        const long n = (long)v.size();
        const double* d = v.data();
        const bool usar = n >= LIMIAR_PAR;

        // 1) mínimo e máximo: reduções min/max
        double mn = d[0], mx = d[0];
    #pragma omp parallel for if(usar) reduction(min : mn) reduction(max : mx)
        for (long i = 0; i < n; i++) {
            mn = min(mn, d[i]);
            mx = max(mx, d[i]);
        }
        r.li = floor(mn);
        r.k = (int)floor((mx - r.li) / h) + 1;
        r.freq.assign(r.k, 0);

        // 2) histograma (frequências): cada thread acumula num vetor privado e o
        //    OpenMP soma todos no final (redução de array) -> sem race condition
        long* f = r.freq.data();
        const int k = r.k;
        const double li = r.li;
    #pragma omp parallel for if(usar) reduction(+ : f[:k])
        for (long i = 0; i < n; i++) f[(int)((d[i] - li) / h)]++;

        // 3) soma ponderada para a média
        double soma = 0.0;
    #pragma omp parallel for if(usar) reduction(+ : soma)
        for (int c = 0; c < k; c++) {
            double xi = li + (c + 0.5) * h;
            soma += xi * f[c];
        }
        r.media = soma / n;

        // 4) soma dos quadrados dos desvios para o desvio padrão
        double ss = 0.0;
        const double mu = r.media;
    #pragma omp parallel for if(usar) reduction(+ : ss)
        for (int c = 0; c < k; c++) {
            double xi = li + (c + 0.5) * h;
            ss += f[c] * (xi - mu) * (xi - mu);
        }
        r.desvio = sqrt(ss / n);
        r.cv = r.desvio / r.media * 100.0;
        return r;
    }

    // --------------------------------------------------------------------- util
    void gerar(const string& arq, int n) {
        mt19937 gen(42);
        normal_distribution<double> alt(170.0, 9.0), peso(70.0, 12.0);
        ofstream out(arq);
        out.setf(ios::fixed);
        out.precision(1);
        for (int i = 0; i < n; i++) {
            double a = clamp(alt(gen), 140.0, 205.0);
            double p = clamp(peso(gen), 40.0, 130.0);
            out << a << " " << p << "\n";
        }
    }

    // Executa fn() 'reps' vezes (apos 1 execucao de aquecimento) e devolve a MEDIANA
    // dos tempos, que e mais robusta a ruido e a timers de baixa resolucao.
    template <class F>
    double medir(F fn, int reps, Resultado& out) {
        out = fn();  // aquecimento (cache, page faults, criacao do pool de threads)
        vector<double> t(reps);
        for (int i = 0; i < reps; i++) {
            double t0 = omp_get_wtime();
            out = fn();
            t[i] = omp_get_wtime() - t0;
        }
        nth_element(t.begin(), t.begin() + reps / 2, t.end());
        return t[reps / 2];
    }

    void mostrar(const char* nome, const char* un, const Resultado& r, double h) {
        printf("\n%s (classes de %.0f %s)\n", nome, h, un);
        printf("  classe            ponto medio   frequencia\n");
        for (int c = 0; c < r.k; c++) {
            double a = r.li + c * h;
            printf("  [%6.1f, %6.1f)  %8.1f      %ld\n", a, a + h, a + h / 2, r.freq[c]);
        }
        printf("  media = %.4f %s | desvio = %.4f %s | CV = %.4f %%\n", r.media, un,
            r.desvio, un, r.cv);
    }

    int main(int argc, char** argv) {
        if (argc >= 3 && !strcmp(argv[1], "--gerar")) {
            gerar(argv[2], 1000);
            printf("Arquivo %s gerado.\n", argv[2]);
            return 0;
        }
        if (argc < 2) {
            fprintf(stderr, "Uso: %s dados.txt [--replicar R] | %s --gerar dados.txt\n",
                    argv[0], argv[0]);
            return 1;
        }

        long replicar = 1;
        for (int i = 2; i + 1 < argc; i++)
            if (!strcmp(argv[i], "--replicar")) replicar = atol(argv[i + 1]);

        // ----- leitura (sequencial: I/O não compensa paralelizar)
        vector<double> alt, peso;
        ifstream in(argv[1]);
        double a, p;
        while (in >> a >> p) {
            alt.push_back(a);
            peso.push_back(p);
        }
        printf("Pessoas lidas: %zu\n", alt.size());
        if (alt.empty()) return 1;

        if (replicar > 1) {
            size_t n0 = alt.size();
            alt.reserve(n0 * replicar);
            peso.reserve(n0 * replicar);
            for (long r = 1; r < replicar; r++)
                for (size_t i = 0; i < n0; i++) {
                    alt.push_back(alt[i]);
                    peso.push_back(peso[i]);
                }
            printf("Dados replicados %ldx -> N = %zu\n", replicar, alt.size());
        }

        const double HA = 8.0, HP = 4.0;
        Resultado ra, rp, sa, sp;

        // ----- resultados (versão paralela) e conferência com a sequencial
        ra = calcular_par(alt, HA);
        rp = calcular_par(peso, HP);
        sa = calcular_seq(alt, HA);
        sp = calcular_seq(peso, HP);
        mostrar("ESTATURA", "cm", ra, HA);
        mostrar("PESO", "kg", rp, HP);
        printf("\nConferencia seq x par: dif CV estatura = %.2e | dif CV peso = %.2e\n",
            fabs(ra.cv - sa.cv), fabs(rp.cv - sp.cv));

        // ----- desempenho
        int reps = (int)max(5L, min(2000L, 20000000L / (long)alt.size()));
        printf("\n=== Desempenho (mediana de %d execucoes, estatura+peso) ===\n", reps);

        auto seq = [&] {
            Resultado x = calcular_seq(alt, HA);
            Resultado y = calcular_seq(peso, HP);
            return y.cv + x.cv > 0 ? x : y;
        };
        Resultado tmp;
        double ts = medir(seq, reps, tmp);
        printf("Sequencial: %.6f s\n", ts);

        int maxt = omp_get_max_threads();
        vector<int> lista;
        for (int t = 1; t < maxt; t *= 2) lista.push_back(t);
        lista.push_back(maxt);  // sempre testa o maximo disponivel

        printf("threads  tempo(s)    speedup  eficiencia\n");
        for (int t : lista) {
            omp_set_num_threads(t);
            auto par = [&] {
                Resultado x = calcular_par(alt, HA);
                Resultado y = calcular_par(peso, HP);
                return y.cv + x.cv > 0 ? x : y;
            };
            double tp = medir(par, reps, tmp);
            printf("%5d    %.6f   %6.2fx   %6.1f%%\n", t, tp, ts / tp, 100.0 * ts / tp / t);
        }
        return 0;
    }