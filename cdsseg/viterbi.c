/* Viterbi decoding over a sparse state graph whose edges can be gated by per-position motif masks.
 * Built and called by cdsseg/hmm.py (ctypes). Scores are normalised at every position, so double
 * precision holds over whole chromosomes; back-pointers take N * S bytes. */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

int viterbi(int64_t n, int n_states, const uint8_t *emits_cds, int n_edges, const int32_t *src,
            const int32_t *dst, const double *logp, const int32_t *motif, const uint8_t *masks,
            const float *em_cds, const double *init, uint8_t *path) {
    double *v = malloc(sizeof(double) * n_states), *w = malloc(sizeof(double) * n_states);
    uint8_t *bp = malloc((size_t)n * n_states);
    if (!v || !w || !bp) { free(v); free(w); free(bp); return -1; }
    for (int s = 0; s < n_states; s++) v[s] = init[s] + (emits_cds[s] ? em_cds[0] : 0.0);
    for (int64_t t = 1; t < n; t++) {
        uint8_t *b = bp + (size_t)t * n_states;
        for (int s = 0; s < n_states; s++) { w[s] = -INFINITY; b[s] = 0; }
        for (int e = 0; e < n_edges; e++) {
            if (motif[e] >= 0 && !masks[(size_t)motif[e] * n + t]) continue;
            double x = v[src[e]] + logp[e];
            if (x > w[dst[e]]) { w[dst[e]] = x; b[dst[e]] = (uint8_t)src[e]; }
        }
        double best = -INFINITY;
        for (int s = 0; s < n_states; s++) {
            if (emits_cds[s]) w[s] += em_cds[t];
            if (w[s] > best) best = w[s];
        }
        for (int s = 0; s < n_states; s++) v[s] = w[s] - best;  /* normalised scores of position t */
    }
    int s = 0;
    for (int k = 1; k < n_states; k++) if (v[k] > v[s]) s = k;
    path[n - 1] = (uint8_t)s;
    for (int64_t t = n - 1; t > 0; t--) { s = bp[(size_t)t * n_states + s]; path[t - 1] = (uint8_t)s; }
    free(v); free(w); free(bp);
    return 0;
}
