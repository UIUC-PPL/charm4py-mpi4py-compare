#include <stdint.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define NMAX 64

typedef struct {
  uint64_t  LCG_seed;
  uint64_t  LCG_A[NMAX];
} random_draw_t;

void     LCG_init(random_draw_t *);
uint64_t LCG_next(uint64_t, random_draw_t *);
void     LCG_get_chunk(uint64_t *, uint64_t *, int, int, uint64_t);
void     LCG_jump(uint64_t, uint64_t, random_draw_t *);
uint64_t random_draw(double, random_draw_t *);


static uint64_t  LCG_a = 6364136223846793005;
static uint64_t  LCG_c = 1442695040888963407;
static uint64_t  LCG_seed_init = 27182818285; //used to (re)set seed 

#ifdef __OPENMP
#pragma omp threadprivate (LCG_seed, LCG_A)
#endif

random_draw_t *LCG_Create()
{
  random_draw_t *lcg = malloc(sizeof(random_draw_t));
  return lcg;
}

void LCG_Destroy(void *to_destroy)
{
  free(to_destroy);
}

/* for a range of 0 to size-i, find chunk assigned to calling thread */
void LCG_get_chunk(uint64_t *start, uint64_t *end, int tid, int nthreads, uint64_t size) {
    uint64_t chunk, remainder;
    chunk = size/nthreads;
    remainder = size - chunk*nthreads;
 
    if ((uint64_t)tid < remainder) {
      *start = tid*(chunk+1);
      *end   = *start + chunk;
    }
    else {
      *start = remainder*(chunk+1) + (tid-remainder)*chunk;
      *end   = *start + chunk -1;
    }
    return;
}
 
 
static uint64_t tail(uint64_t x) {
  uint64_t x2 = x;
  if (!x) return x;
  uint64_t result = 1;
  while (x>>=1) result <<=1;
  return (x2 - result);
}  
 
/* Sum(i=1,2^k) a^i */
static uint64_t SUMPOWER(int k, random_draw_t *parm) {
  if (!k) return LCG_a;
  return SUMPOWER(k-1, parm)*(1+parm->LCG_A[k-1]);
}
 
static int LOG(uint64_t n) {
  int result = 0;
  while (n>>=1) result++;
  return(result);
}
 
/* Sum(i=1,n) a^i, with n arbitrary */
static uint64_t SUMK(uint64_t n, random_draw_t *parm) {
  if (n==0) return(0);
  uint64_t HEAD = SUMPOWER(LOG(n),parm);
  uint64_t TAILn = tail(n);
  if (TAILn==0) return(HEAD);
  return(HEAD + (parm->LCG_A[LOG(n)])*SUMK(TAILn,parm));
}
 
uint64_t LCG_next(uint64_t bound, random_draw_t *parm) {
  parm->LCG_seed = LCG_a*parm->LCG_seed + LCG_c;
  return (parm->LCG_seed%bound);
}
 
void LCG_init(random_draw_t *parm){
 
  int i;
 
  parm->LCG_seed = LCG_seed_init;
  parm->LCG_A[0] = LCG_a;
  for (i=1; i<NMAX; i++) {
    parm->LCG_A[i] = parm->LCG_A[i-1]*parm->LCG_A[i-1];
  }
  return;
}
 
void LCG_jump(uint64_t m, uint64_t bound, random_draw_t *parm){
 
  int i, index, LCG_power[NMAX];
  uint64_t mm, s_part;

  for (i=0; i<NMAX; i++) LCG_power[i] = 0; 
  parm->LCG_seed = LCG_seed_init;
  
  /* Catch two special cases */
  switch (m) {
  case 0: return;
  case 1: LCG_next(bound, parm); return;
  }
 
  mm = m;
  index = 0;
  while (mm) {
    LCG_power[index++] = mm&1;
    mm >>=1;
  }
 
  s_part = 1;
  for (i=0; i<index; i++) if (LCG_power[i]) s_part *= parm->LCG_A[i];
  parm->LCG_seed = s_part*parm->LCG_seed + (SUMK(m-1,parm)+1)*LCG_c;
  return;
}

uint64_t random_draw(double mu, random_draw_t *parm)
{
  const double   two_pi      = 2.0*3.14159265358979323846;
  const uint64_t rand_max    = ULLONG_MAX;
  const double   rand_div    = 1.0/ULLONG_MAX;
  const uint64_t denominator = UINT_MAX;

  static double   z0, z1;
  double          u0, u1, sigma;
  static uint64_t numerator;
  static uint64_t i0, i1;

  if (mu>=1.0) {
    /* set std dev equal to 15% of average; ensures result will never be negative       */
    sigma = mu*0.15;  
    u0 = LCG_next(rand_max, parm) * rand_div;
    u1 = LCG_next(rand_max, parm) * rand_div;

    z0 = sqrt(-2.0 * log(u0)) * cos(two_pi * u1);
    z1 = sqrt(-2.0 * log(u0)) * sin(two_pi * u1);
    return (uint64_t) (z0 * sigma + mu+0.5);
  }
  else {
    /* we need to pick two integers whose quotient approximates mu; set one to UINT_MAX */
    numerator = (uint32_t) (mu*(double)denominator);
    i0 = LCG_next(denominator, parm); /* don't use value, but must call LCG_next twice  */
    i1 = LCG_next(denominator, parm);
    return ((uint64_t)(i1<=numerator));
  }

}
