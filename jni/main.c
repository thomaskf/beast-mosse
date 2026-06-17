#include <complex.h>
#include <fftw3.h>
#include <stdlib.h>
#include <math.h>
#include <jni.h>
#include <string.h>
#include <assert.h>

#include "rfftw.h"
#include "quasse-eqs-fftC.h"
#include "mosse-eqs-fftC.h"

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>

#define R_D__0 (give_log ? -INFINITY : 0.)
#define M_LN_SQRT_2PI 0.918938533204672741780329736406
#define M_1_SQRT_2PI 0.398942280401432677939946059934

JNIEXPORT jlong JNICALL Java_mosse_MosseDistribution_makeMosseFFT(
    JNIEnv *env, jobject thisObject, jint nx, jdouble dx, jintArray array_nd,
    jint flags) {

  jsize n_fft = (int)((*env)->GetArrayLength(env, array_nd));
  assert(n_fft);

  jint *const_array_body = (*env)->GetIntArrayElements(env, array_nd, 0);
  int *local_nd = malloc(sizeof(int) * n_fft);
  assert(local_nd);
  // copy java array to c array
  memcpy(local_nd, const_array_body, sizeof(int) * n_fft);
  (*env)->ReleaseIntArrayElements(env, array_nd, const_array_body, 0);

  mosse_fft *obj = make_mosse_fft(n_fft, nx, dx, local_nd, flags);

  return (jlong)(obj);
}

// JNIEXPORT void JNICALL Java_mosse_MosseDistribution_propagateT(JNIEnv *env, jobject thisObject, jlong obj_ptr) {
//    mosse_fft *obj = (mosse_fft *)(obj_ptr);
//    int idx = 0;
//    propagate_t_mosse(obj, idx);
// }

JNIEXPORT jdoubleArray JNICALL Java_mosse_MosseDistribution_getX(JNIEnv *env, jobject thisObject, jlong obj_ptr) {
   mosse_fft *obj = (mosse_fft*)(obj_ptr);
   // convert c array to java array and return it
   int size = (obj->max_nd) * (obj->nx);
   jdoubleArray result = (*env)->NewDoubleArray(env, size);
   // move c array structure to java structure 
   (*env)->SetDoubleArrayRegion(env, result, 0, size, obj->x); 
   return result;

}

JNIEXPORT jdoubleArray JNICALL Java_mosse_MosseDistribution_doIntegrateMosse(
    JNIEnv *env, jobject thisObject, jlong mosse_ptr, jdoubleArray vars, jdoubleArray lambda, jdoubleArray mu,
    jdoubleArray r, jdouble a,
    jdouble drift, jdouble diffusion, jdoubleArray Q,
    jdoubleArray eVal, jdoubleArray eVec, jdoubleArray iEvec, jboolean useEigen,
    jdoubleArray eQCache, jlong eigenGeneration,
    jint nt, jdouble dt, jint pad_left, jint pad_right) {

  mosse_fft *obj = (mosse_fft *)(mosse_ptr);
  assert(obj);

  int nkl = pad_left;
  int nkr = pad_right;
  int ndat = (int)((*env)->GetArrayLength(env, lambda));
  double c_dt = dt;
  int c_nt = nt;
  double c_drift = drift;
  double c_diffusion = diffusion;
  int i, idx, nd;

  int len_vars = (int)((*env)->GetArrayLength(env, vars));
  nd = len_vars / obj->nx;

  /* Fill reusable per-plan buffers with a single GetDoubleArrayRegion each.
   * No malloc/free, no GetArrayElements/Release pinning, no double-copy:
   * GetDoubleArrayRegion does one bulk JVM->C copy directly into our buffer.
   */
  jsize n_lambda = (jsize)ndat;
  jsize n_mu     = (jsize)((*env)->GetArrayLength(env, mu));
  jsize n_r      = (jsize)((*env)->GetArrayLength(env, r));
  jsize n_vars   = (jsize)len_vars;
  jsize n_Q      = (jsize)((*env)->GetArrayLength(env, Q));

  (*env)->GetDoubleArrayRegion(env, lambda, 0, n_lambda, obj->lambda_buf);
  (*env)->GetDoubleArrayRegion(env, mu,     0, n_mu,     obj->mu_buf);
  (*env)->GetDoubleArrayRegion(env, r,      0, n_r,      obj->r_buf);
  (*env)->GetDoubleArrayRegion(env, vars,   0, n_vars,   obj->vars_buf);

  /* Q is just 16 doubles — copy into a local stack-resident view via
   * gsl_matrix_view_array. Use a small stack buffer to avoid touching the
   * per-plan allocator. */
  double c_Q[16];
  assert(n_Q == 16);
  (*env)->GetDoubleArrayRegion(env, Q, 0, n_Q, c_Q);

  idx = lookup(nd, obj->nd, obj->n_fft);
  if (idx < 0) {
    printf("Failed to find nd = %d\n", nd);
    abort();
  }

  qf_copy_x_mosse(obj, obj->vars_buf, nd, 1);

  obj->lambda = obj->lambda_buf;
  obj->mu     = obj->mu_buf;
  {
    gsl_matrix_view Q_view = gsl_matrix_view_array(c_Q, 4, 4);
    gsl_matrix_memcpy(obj->Q, &Q_view.matrix);
  }
  obj->r  = obj->r_buf;
  obj->a  = (double)a;
  obj->dt = c_dt;

  /* Eigendata + eQ_cache copy, gated by Java's generation counter.
   * These arrays (eVal/eVec/iEvec/eQCache) are constant across all branch
   * calls within one likelihood eval. Java bumps `eigenGeneration` whenever
   * buildEigenDecomp / buildEQCache rebuilds them (transitionMatricesDirty
   * path in MosseTreeLikelihood.traverseFull). On cache hit, the four
   * GetDoubleArrayRegion copies (plus their safepoint cost) are skipped —
   * the eQCache copy alone is ~520 KB at high resolution, so this saves
   * gigabytes of JNI memcpy per MCMC step at 200-thread concurrency.
   *
   * obj->eigen_generation starts at -1 (see make_mosse_fft) so the first
   * call from each thread always copies. Switched from Get*Elements + memcpy
   * to GetDoubleArrayRegion for the same single-trip bulk-copy gain we
   * already use for lambda/mu/r/vars.
   */
  if (eigenGeneration != obj->eigen_generation) {
    if (useEigen && eVal != NULL && eVec != NULL && iEvec != NULL) {
      (*env)->GetDoubleArrayRegion(env, eVal,  0,  4, obj->eVal);
      (*env)->GetDoubleArrayRegion(env, eVec,  0, 16, obj->eVec);
      (*env)->GetDoubleArrayRegion(env, iEvec, 0, 16, obj->iEvec);
      obj->useEigen = 1;
    } else {
      obj->useEigen = 0;
    }
    if (eQCache != NULL) {
      jsize n_cache = (*env)->GetArrayLength(env, eQCache);
      (*env)->GetDoubleArrayRegion(env, eQCache, 0, n_cache, obj->eQ_cache);
      obj->eQ_cache_valid = 1;
    } else {
      obj->eQ_cache_valid = 0;
    }
    obj->eigen_generation = eigenGeneration;
  }
  /* else: cache hit — obj->useEigen, obj->eVal/eVec/iEvec, obj->eQ_cache /
   * eQ_cache_valid already current for this generation. */

  {
    const double *cl = obj->lambda_buf, *cm = obj->mu_buf;
    for (i = 0; i < ndat; i++) {
      obj->z[i]  = exp(c_dt * (cl[i] - cm[i]));
      obj->zz[i] = exp(c_dt * (cl[i] + cm[i]));
    }
  }

  qf_setup_kern_mosse(obj, c_drift, c_diffusion, c_dt, nkl, nkr);

  do_integrate_mosse(obj, c_nt, idx);

  /* obj->lambda/mu/r point at the persistent lambda_buf/mu_buf/r_buf
   * and will be overwritten on the next call; no NULL reset needed. */

  int size = obj->nx * nd;

  // copy to java array
  jdoubleArray j_result = (*env)->NewDoubleArray(env, size);
  (*env)->SetDoubleArrayRegion(env, j_result, 0, size, obj->x);

  /* lambda_buf, mu_buf, r_buf, vars_buf are persistent per-plan buffers
   * owned by the mosse_fft object; they are freed in mosseFinalize, not here. */
  return j_result;
}

int lookup(int x, int *v, int len) {
  int i, idx=-1;
  for ( i = 0; i < len; i++ )
    if ( v[i] == x ) {
      idx = i;
      break;
    }

  return idx;
}

double pnorm_cdf(double x, double mu, double sigma) {
    if (isnan(x) || isnan(mu) || isnan(sigma)) return NAN;
    if (sigma <= 0.0) {
        if (x < mu) return 0.0;
        return 1.0;
    }
    double z = (x - mu) / sigma;
    return 0.5 * erfc(-z / M_SQRT2);
}

/* Probability mass in a cell centered at x with width dx */
double normal_cell_mass(double x_center, double dx, double mu, double sigma) {
    double half = 0.5 * dx;
    double left  = x_center - half;
    double right = x_center + half;
    double a = pnorm_cdf(right, mu, sigma) - pnorm_cdf(left, mu, sigma);
    return (a < 0.0) ? 0.0 : a;
}

double dnorm(double x, double mu, double sigma, int give_log) {

  if (isnan(x) || isnan(mu) || isnan(sigma)) {
    return x + mu + sigma;
  }
  if (!isfinite(sigma)) {
    return R_D__0;
  }
  if (!isfinite(x) && mu == x) {
    return NAN; /* x-mu is NaN */
  }
  if (sigma <= 0) {
    if (sigma < 0) {
      printf("Error!\n");
      return NAN;
    }
    /* sigma == 0 */
    return (x == mu) ? INFINITY : R_D__0;
  }
  x = (x - mu) / sigma;

  if (!isfinite(x)) {
    return R_D__0;
  }
  return (give_log ? -(M_LN_SQRT_2PI + 0.5 * x * x + log(sigma))
                   : M_1_SQRT_2PI * exp(-0.5 * x * x) / sigma);
  /* M_1_SQRT_2PI = 1 / sqrt(2 * pi) */
}

rfftw_plan_real *make_rfftw_plan_real(int nd, int nx, int dir, double *x,
                                      fftw_complex *y, int flags) {
  rfftw_plan_real *obj = (rfftw_plan_real *)calloc(1, sizeof(rfftw_plan_real));
  int ny = ((int)floor(nx / 2)) + 1;
  int xstride, xdist, ystride, ydist;

  if (dir == DIR_COLS) {
    xstride = 1;
    ystride = 1;
    xdist = nx;
    ydist = ny;
  } else { /* if ( dir == DIR_ROWS ) { // assume ROWS silently */
    xstride = nd;
    ystride = nd;
    xdist = 1;
    ydist = 1;
  }

  obj->nd = nd;
  obj->nx = nx;
  obj->ny = ny;
  obj->x = x;
  obj->y = y;
  obj->dir = dir;

  /* Consider FFTW_PATIENT, or even FFTW_EXHAUSTIVE */
  obj->plan_f = fftw_plan_many_dft_r2c(1, &nx, nd, obj->x, NULL, xstride, xdist,
                                       obj->y, NULL, ystride, ydist, flags);
  obj->plan_b = fftw_plan_many_dft_c2r(1, &nx, nd, obj->y, NULL, ystride, ydist,
                                       obj->x, NULL, xstride, xdist, flags);
  return obj;
}

void convolve(rfftw_plan_real *obj, fftw_complex *fy) {
  int nx = obj->nx, ny = obj->ny, nd = obj->nd, i, j;
  int nxd = nx * nd;
  double *x = obj->x;
  fftw_complex *y = obj->y;

  fftw_execute(obj->plan_f);

  for (i = 0; i < nd; i++)
    for (j = 0; j < ny; j++, y++)
      (*y) *= fy[j];

  fftw_execute(obj->plan_b);

  /* Normalize by nx and clamp tiny negative values from FFT round-off to 0,
   * so downstream sums over D stay non-negative. */
  for (i = 0; i < nxd; i++) {
    double v = x[i] / nx;
    x[i] = (v < 0.0) ? 0.0 : v;
  }
}

/* This does the memory allocation and plans the FFT transforms */
mosse_fft *make_mosse_fft(int n_fft, int nx, double dx, int *nd, int flags) {
  mosse_fft *obj = calloc(1, sizeof(mosse_fft));
  int ny = (((int)floor(nx / 2)) + 1);
  int i, max_nd = 1;
  for (i = 0; i < n_fft; i++)
    if (nd[i] > max_nd)
      max_nd = nd[i];

  obj->n_fft = n_fft;
  obj->nx = nx;
  obj->ny = ny;
  obj->dx = dx;
  obj->nd = nd;

  obj->x = fftw_malloc(max_nd * nx * sizeof(double));
  obj->y = fftw_malloc(max_nd * (ny + 1) * sizeof(fftw_complex));
  obj->max_nd = max_nd;

  obj->z = (double *)calloc(nx, sizeof(double));
  obj->zz = (double *)calloc(nx, sizeof(double));
  obj->wrk = (double *)calloc(nx, sizeof(double));
  obj->wrkd = (double *)calloc(max_nd * nx, sizeof(double));

  obj->Q  = gsl_matrix_alloc(4, 4);
  obj->eQ = gsl_matrix_alloc(4, 4);
  obj->Qx = gsl_matrix_alloc(4, 4);

  /* Eigenvalues, Eigenvectors, and Inverses of Eigenvectors */
  obj->eVal     = (double *)calloc(4,  sizeof(double));
  obj->eVec     = (double *)calloc(16, sizeof(double));
  obj->iEvec    = (double *)calloc(16, sizeof(double));
  obj->useEigen = 0;

  /* Per-bin eQ cache (a==0 fast path): one 4x4 matrix per bin, flat row-major. */
  obj->eQ_cache       = (double *)calloc(nx * 16, sizeof(double));
  obj->eQ_cache_valid = 0;

  obj->xt = (double *)calloc(nx * max_nd, sizeof(double));

  obj->fft = (rfftw_plan_real **)calloc(n_fft, sizeof(rfftw_plan_real *));

  for (i = 0; i < n_fft; i++) {
    obj->fft[i] =
        make_rfftw_plan_real(nd[i], nx, DIR_COLS, obj->x, obj->y, flags);
  }

  /* Brownian kernel */
  obj->kern_x = fftw_malloc(nx * sizeof(double));
  obj->kern_y = fftw_malloc((ny + 1) * sizeof(fftw_complex));
  obj->kernel =
      make_rfftw_plan_real(1, nx, DIR_COLS, obj->kern_x, obj->kern_y, flags);

  /* Pre-allocated reusable buffers for JNI array copy-in. These let
   * doIntegrateMosse fill them via a single GetDoubleArrayRegion call per
   * argument instead of malloc/Get*Elements/memcpy/Release every branch. */
  obj->lambda_buf = (double *)calloc(nx,          sizeof(double));
  obj->mu_buf     = (double *)calloc(nx,          sizeof(double));
  obj->r_buf      = (double *)calloc(nx,          sizeof(double));
  obj->vars_buf   = (double *)calloc(nx * max_nd, sizeof(double));

  /* Kernel-setup cache starts invalid; first qf_setup_kern_mosse call
   * will populate the cache and the FFT. */
  obj->kern_valid     = 0;
  obj->kern_drift     = 0.0;
  obj->kern_diffusion = 0.0;
  obj->kern_dt        = 0.0;
  obj->kern_nkl       = 0;
  obj->kern_nkr       = 0;

  /* Eigendata generation starts at -1 so the first call from each thread
   * (whatever generation Java passes in, including 0) always copies. */
  obj->eigen_generation = -1L;

  return obj;
}

JNIEXPORT void JNICALL Java_mosse_MosseDistribution_mosseFinalize(JNIEnv *env, jobject thisObject, jlong obj_ptr) {
  mosse_fft *obj = (mosse_fft *)(obj_ptr);

  int i;
  /* Rprintf("Cleaning up\n"); */

  for (i = 0; i < obj->n_fft; i++) {
    fftw_destroy_plan(obj->fft[i]->plan_f);
    fftw_destroy_plan(obj->fft[i]->plan_b);
  }
  free(obj->fft);
  free(obj->nd);

  fftw_free(obj->x);
  fftw_free(obj->y);

  free(obj->z);
  free(obj->zz);
  free(obj->wrk);
  free(obj->wrkd);

  gsl_matrix_free(obj->Q);
  gsl_matrix_free(obj->eQ);
  gsl_matrix_free(obj->Qx);

  free(obj->eVal);
  free(obj->eVec);
  free(obj->iEvec);
  free(obj->eQ_cache);

  free(obj->lambda_buf);
  free(obj->mu_buf);
  free(obj->r_buf);
  free(obj->vars_buf);

  fftw_destroy_plan(obj->kernel->plan_f);
  fftw_destroy_plan(obj->kernel->plan_b);

  fftw_free(obj->kern_x);
  fftw_free(obj->kern_y);

  free(obj);
}

void qf_copy_x_mosse(mosse_fft *obj, double *x, int nd, int copy_in) {
  int i, n = obj->nx * nd;
  double *fft_x = obj->x;
  if (copy_in) {
    for (i = 0; i < n; i++) {
      fft_x[i] = x[i];
    }
  } else {
    for (i = 0; i < n; i++) {
      x[i] = fft_x[i];
    }
  }
}

void qf_setup_kern_mosse(mosse_fft *obj, double drift, double diffusion,
                         double dt, int nkl, int nkr) {
  /* Short-circuit: if the inputs match what we last used to build the kernel,
   * skip the kernel buffer reconstruction and FFT entirely. drift/diffusion/dt
   * and the pad widths are constant across all branch calls within one
   * likelihood evaluation, so this saves an O(nx) loop + an FFT per branch. */
  if (obj->kern_valid
      && obj->kern_drift     == drift
      && obj->kern_diffusion == diffusion
      && obj->kern_dt        == dt
      && obj->kern_nkl       == nkl
      && obj->kern_nkr       == nkr) {
    /* obj->nkl/nkr/npad/ndat/drift/diffusion already reflect cached state */
    return;
  }

  const int nx = obj->nx;
  int i;
  double x, *kern_x = obj->kern_x, tot = 0, dx = obj->dx;
  double mean, sd;

  obj->nkl = nkl;
  obj->nkr = nkr;
  obj->npad = nkl + 1 + nkr;
  obj->ndat = nx - obj->npad;
  obj->drift = drift;
  obj->diffusion = diffusion;

  tot = 0;
  mean = -dt * drift;
  sd = sqrt(dt * diffusion);

  for (i = 0, x = 0; i <= nkr; i++, x += dx)
    tot += kern_x[i] = normal_cell_mass(x, dx, mean, sd);
  for (i = nkr + 1; i < nx - nkl; i++)
    kern_x[i] = 0;
  for (i = nx - nkl, x = -nkl * dx; i < nx; i++, x += dx)
    tot += kern_x[i] = normal_cell_mass(x, dx, mean, sd);    

  for (i = 0; i <= nkr; i++)
    kern_x[i] /= tot;
  for (i = nx - nkl; i < nx; i++)
    kern_x[i] /= tot;

  fftw_execute(obj->kernel->plan_f);

  /* Mark cache as valid for the inputs we just used. */
  obj->kern_valid     = 1;
  obj->kern_drift     = drift;
  obj->kern_diffusion = diffusion;
  obj->kern_dt        = dt;
  obj->kern_nkl       = nkl;
  obj->kern_nkr       = nkr;
}

void do_integrate_mosse(mosse_fft *obj, int nt, int idx) {
  int i, nkl = obj->nkl;
  for (i = 0; i < nt; i++) {
    propagate_t_mosse(obj, idx);
    propagate_x_mosse(obj, idx);
    if (isnan(obj->x[nkl])) {
      printf("Integration failure at step %d\n", i);
      abort();
    }
  }
}

/* Lower level functions */
void propagate_t_mosse(mosse_fft *obj, int idx) {
  int ix, id, ik, nx = obj->nx, ndat = obj->ndat, nd = obj->nd[idx],
                  nk = nd - 1;
  double *vars = obj->x, *dd = obj->wrk;
  double *xt = obj->xt; /* transposed buffer: xt[ix * nd + id] */
  double e, tmp1, tmp2, lambda_x, mu_x, z_x;
  double r_x;
  double dt = obj->dt;

  for (ix = 0; ix < ndat; ix++) {
    lambda_x = obj->lambda[ix];
    mu_x = obj->mu[ix];
    z_x = obj->z[ix];
    e = vars[ix];

    /* Update the E values */
    tmp1 = mu_x - lambda_x * e;
    tmp2 = z_x * (e - 1);
    vars[ix] = (tmp1 + tmp2 * mu_x) / (tmp1 + tmp2 * lambda_x);

    tmp1 =
        (lambda_x - mu_x) / (z_x * lambda_x - mu_x + (1 - z_x) * lambda_x * e);
    /* Here is the D scaling factor */
    dd[ix] = z_x * tmp1 * tmp1;
  }

  /* Transpose D values: x[id*nx+ix] -> xt[ix*nd+id], apply scaling */
  for (id = 1; id < nd; id++) {
    double *src = obj->x + nx * id;
    for (ix = 0; ix < ndat; ix++) {
      double v = src[ix];
      xt[ix * nd + id] = (v < 0) ? 0.0 : v * dd[ix];
    }
  }

  for (ix = 0; ix < ndat; ix++) {
    double *F = &xt[ix * nd + 1];
    double *out = &xt[ix * nd + 1];
    double F_loc[4];
    for (ik = 0; ik < nk; ik++) F_loc[ik] = F[ik];

    if (dd[ix] == 0.0) {
      for (id = 0; id < nk; id++) out[id] = 0.0;
      continue;
    }

    r_x = obj->r[ix];
    z_x = obj->zz[ix];
    if (obj->a != 0.0) {
      tmp1 = r_x * dt + obj->a * log(dd[ix] * z_x);
    } else {
      tmp1 = r_x * dt;
    }

    /* Build eQ = exp(Q * tmp1) */
    if (obj->useEigen) {
      const double *eQ_src;
      double eQ_loc[16];

      if (obj->eQ_cache_valid) {
        eQ_src = &obj->eQ_cache[ix * 16];
      } else {
        double eL[4];
        for (int k = 0; k < 4; k++)
          eL[k] = exp(obj->eVal[k] * tmp1);
        for (int i = 0; i < 4; i++)
          for (int j = 0; j < 4; j++) {
            double sum = 0.0;
            for (int k = 0; k < 4; k++)
              sum += obj->eVec[i * 4 + k] * eL[k] * obj->iEvec[k * 4 + j];
            eQ_loc[i * 4 + j] = sum;
          }
        eQ_src = eQ_loc;
      }

      /* Apply eQ^T to F */
      for (id = 0; id < nk; id++) {
        double sum = 0.0;
        for (ik = 0; ik < nk; ik++)
          sum += eQ_src[ik * 4 + id] * F_loc[ik];
        out[id] = sum;
      }
    } else {
      gsl_matrix_memcpy(obj->Qx, obj->Q);
      gsl_matrix_scale(obj->Qx, tmp1);
      gsl_linalg_exponential_ss(obj->Qx, obj->eQ, GSL_MODE_DEFAULT);

      for (id = 0; id < nk; id++) {
        double sum = 0.0;
        for (ik = 0; ik < nk; ik++)
          sum += gsl_matrix_get(obj->eQ, ik, id) * F_loc[ik];
        out[id] = sum;
      }
    }
  }

  /* Transpose back; clamp negatives from eigendecomposition round-off so
   * downstream sums stay non-negative. */
  for (id = 1; id < nd; id++) {
    double *dst = obj->x + nx * id;
    for (ix = 0; ix < ndat; ix++) {
      double v = xt[ix * nd + id];
      dst[ix] = (v < 0.0) ? 0.0 : v;
    }
  }
}

void propagate_x_mosse(mosse_fft *obj, int idx) {
  double *x, *dd, *wrk = obj->wrk;
  int i, id, nx = obj->nx;
  int nkl = obj->nkl, nkr = obj->nkr, npad = obj->npad;
  int nd = obj->nd[idx];

  x = obj->x + 0;
  for (i = 0; i < nkl; i++)
    wrk[i] = x[i];
  for (i = nx - npad - nkr; i < nx - npad; i++)
    wrk[i] = x[i];

  convolve(obj->fft[idx], obj->kern_y);

  x = obj->x + 0;
  for (i = 0; i < nkl; i++)
    x[i] = wrk[i];
  for (i = nx - npad - nkr; i < nx - npad; i++)
    x[i] = wrk[i];
  for (i = nx - npad; i < nx; i++)
    x[i] = 0;

  for (id = 1; id < nd; id++) {
    x = obj->x + (obj->nx) * id;
    dd = obj->wrkd + (obj->nx) * id;
    for (i = 0; i < nkl; i++)
      x[i] = dd[i];
    for (i = nx - npad - nkr; i < nx - npad; i++)
      x[i] = dd[i];
    for (i = nx - npad; i < nx; i++)
      x[i] = 0;
  }
}
