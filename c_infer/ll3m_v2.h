/* v2 (RoPE / GQA / SwiGLU / RMSNorm) helpers used by ll3m.c */
#ifndef LL3M_V2_H
#define LL3M_V2_H

#include "ll3m.h"

/* Extended config fields live in ClmModel extras (see ll3m.c). */
int clm_model_load_clm2(ClmModel *m, FILE *f, const char *path);
void clm_forward_v2(const ClmModel *m, ClmState *s);
int clm_state_create_v2(ClmState *s, const ClmModel *m, int token_cap);

#endif
